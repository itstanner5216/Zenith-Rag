/**
 * rag-index.ts — MCP tool for indexing a codebase into the RAG store.
 *
 * MCP surface (input schema): { path: string } only.
 *
 * Configuration (chunk size, batch size, file extensions, exclude
 * patterns, store id, etc.) comes from env / lib/config.ts — NOT from
 * tool parameters. This keeps the tool surface narrow per Zenith-MCP
 * conventions: hosts shouldn't have to plumb every knob through the
 * tool schema, and Claude shouldn't have to think about chunk sizes.
 *
 * Backend selection:
 *   - If RAG_API_BASE is set in the environment, the tool calls the
 *     running zenith-rag HTTP server (POST /v1/vector_stores/:id/embeddings/batch).
 *     This is the recommended deployment — the server owns the DB
 *     connection and embedding budget.
 *   - Otherwise, the tool falls back to direct-DB writes via
 *     createVectorDB({databaseUrl}). This still requires DATABASE_URL
 *     and the LiteLLM creds in the MCP host's environment.
 *   - RAG_INDEX_MODE can pin one mode explicitly: "http" or "direct".
 */

import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";
import * as crypto from "crypto";
import { glob } from "glob";
import type { ToolServer, ToolContext, ToolResult } from "./types.js";
import { errorMessage } from "./types.js";
import { loadConfig, type RagConfig } from "../lib/config.js";
import { createVectorDB, type VectorStoreDB } from "../lib/db-adapter.js";
import {
    ensureSchema,
    getExistingSourceHashes,
    deleteEmbeddingsForDocs,
    insertEmbeddingBatch,
    type EmbeddingInsertItem,
} from "../lib/schema.js";
import { LiteLLMEmbeddingService } from "../lib/vector-store.js";
import { chunkFile, type ChunkResult } from "../lib/chunker.js";
import { resolveProjectRoot, getProjectId } from "../utils/project-scope.js";

// ---------------------------------------------------------------------------
// Tool input schema — strictly { path } per the user's "no extra params"
// directive. Indexing knobs live in env / lib/config.ts.
// ---------------------------------------------------------------------------

const IndexParamsSchema = z.object({
    path: z.string().describe(
        "Project directory to index. Must be within the MCP host's allowed " +
        "directories. Files are auto-discovered using the configured glob " +
        "patterns and exclude rules."
    ),
});

type IndexParams = z.infer<typeof IndexParamsSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fileHash(content: string): string {
    return crypto.createHash("sha256").update(content).digest("hex");
}

interface IndexStats {
    filesDiscovered: number;
    filesSkipped: number;
    filesIndexed: number;
    chunksCreated: number;
    errors: string[];
}

type Backend =
    | { kind: "http"; apiBase: string; apiKey: string }
    | { kind: "direct"; db: VectorStoreDB };

/**
 * Resolve which backend (HTTP-callback vs direct-DB) the indexer
 * should use. RAG_INDEX_MODE pins it explicitly; otherwise we prefer
 * HTTP whenever a server URL is configured (drop-in deployment) and
 * fall back to direct-DB only when DATABASE_URL is the only thing
 * available.
 */
function resolveBackend(cfg: RagConfig): Backend {
    const mode = (process.env.RAG_INDEX_MODE ?? "").toLowerCase();
    const apiBase = process.env.RAG_API_BASE ?? cfg.apiBase;
    const apiKey = process.env.RAG_API_KEY ?? cfg.apiKey;

    if (mode === "http") {
        if (!apiBase) {
            throw new Error(
                "RAG_INDEX_MODE=http requires RAG_API_BASE (or apiBase in config) to be set."
            );
        }
        return { kind: "http", apiBase, apiKey };
    }

    if (mode === "direct") {
        if (!cfg.databaseUrl) {
            throw new Error(
                "RAG_INDEX_MODE=direct requires DATABASE_URL to be set."
            );
        }
        return { kind: "direct", db: createVectorDB({ databaseUrl: cfg.databaseUrl }) };
    }

    // Auto-detect: prefer HTTP when a server URL is configured, fall
    // back to direct-DB only when only DATABASE_URL is available.
    if (apiBase) {
        return { kind: "http", apiBase, apiKey };
    }
    if (cfg.databaseUrl) {
        return { kind: "direct", db: createVectorDB({ databaseUrl: cfg.databaseUrl }) };
    }

    throw new Error(
        "rag_index requires either RAG_API_BASE (HTTP-callback mode) or DATABASE_URL (direct-DB mode)."
    );
}

/**
 * Convert a chunk + computed embedding into the wire shape used by
 * the HTTP batch endpoint. The HTTP server accepts pre-computed
 * embeddings so we don't pay for re-embedding on the server side.
 */
interface HttpBatchItem {
    content: string;
    semantic_content: string;
    lexical_content: string;
    embedding: number[];
    metadata: Record<string, unknown>;
}

async function postBatchHttp(
    apiBase: string,
    apiKey: string,
    storeId: string,
    items: HttpBatchItem[],
    projectId: string
): Promise<void> {
    const url = `${apiBase.replace(/\/$/, "")}/v1/vector_stores/${storeId}/embeddings/batch`;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify({ embeddings: items, project_id: projectId }),
        signal: AbortSignal.timeout(120_000),
    });
    if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`HTTP batch insert failed (${resp.status}): ${body.slice(0, 500)}`);
    }
}

async function deleteDocsHttp(
    apiBase: string,
    apiKey: string,
    storeId: string,
    projectId: string,
    docPaths: string[]
): Promise<void> {
    if (docPaths.length === 0) return;
    // The Python-parity DELETE endpoint takes a single doc_path query
    // param per call, so for bulk deletes we issue one request per
    // path. This is the same shape the rag-upload CLI uses.
    const headers: Record<string, string> = {};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    for (const docPath of docPaths) {
        const url = new URL(`${apiBase.replace(/\/$/, "")}/v1/vector_stores/${storeId}/embeddings`);
        url.searchParams.set("doc_path", docPath);
        url.searchParams.set("project_id", projectId);
        const resp = await fetch(url.toString(), {
            method: "DELETE",
            headers,
            signal: AbortSignal.timeout(30_000),
        });
        if (!resp.ok) {
            // Don't abort the whole index — log and move on.
            const body = await resp.text();
            console.warn(`[rag_index] HTTP delete failed for ${docPath} (${resp.status}): ${body.slice(0, 200)}`);
        }
    }
}

interface SourceMapEntry {
    content_hash: string;
    chunk_count: number;
}

async function fetchSourcesHttp(
    apiBase: string,
    apiKey: string,
    storeId: string,
    projectId: string
): Promise<Map<string, SourceMapEntry>> {
    const url = new URL(`${apiBase.replace(/\/$/, "")}/v1/vector_stores/${storeId}/sources`);
    url.searchParams.set("project_id", projectId);
    const headers: Record<string, string> = {};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const resp = await fetch(url.toString(), {
        headers,
        signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) {
        throw new Error(`HTTP sources fetch failed (${resp.status})`);
    }
    const body = (await resp.json()) as {
        sources?: Record<string, { content_hash: string; chunk_count: number }>;
    };
    const map = new Map<string, SourceMapEntry>();
    for (const [docPath, entry] of Object.entries(body.sources ?? {})) {
        if (entry && typeof entry.content_hash === "string") {
            map.set(docPath, {
                content_hash: entry.content_hash,
                chunk_count: entry.chunk_count ?? 0,
            });
        }
    }
    return map;
}

// ---------------------------------------------------------------------------
// Tool registration
// ---------------------------------------------------------------------------

export function register(server: ToolServer, ctx: ToolContext) {
    server.registerTool<IndexParams>(
        "rag_index",
        {
            title: "RAG Index",
            description:
                "Index a project directory for RAG search. Discovers files via the " +
                "configured glob patterns, chunks them with symbol-awareness, generates " +
                "embeddings, and writes them to the vector store. Incremental — files " +
                "whose content hash hasn't changed are skipped.",
            inputSchema: IndexParamsSchema,
            annotations: { readOnlyHint: false, idempotentHint: true, destructiveHint: false },
        },
        async (params: IndexParams): Promise<ToolResult> => {
            const cfg = loadConfig();

            // Validate the input path against MCP host's allowed dirs.
            // ctx.validatePath resolves symlinks, normalises, and throws
            // if the resolved path escapes the sandbox. This is the
            // canonical way to accept a path argument in a Zenith-MCP
            // tool — never trust the raw input.
            let validatedPath: string;
            try {
                validatedPath = await ctx.validatePath(params.path);
            } catch (err: unknown) {
                return {
                    content: [{
                        type: "text",
                        text: `rag_index path validation failed: ${errorMessage(err)}`,
                    }],
                };
            }

            const root = resolveProjectRoot(validatedPath, {
                allowedDirectories: ctx.getAllowedDirectories(),
            }) ?? validatedPath;
            const projectId = getProjectId(root) ?? root;
            const storeId = cfg.defaultStoreId;

            // Pick backend up front so failures (missing env, etc.)
            // are reported clearly rather than mid-index.
            let backend: Backend;
            try {
                backend = resolveBackend(cfg);
            } catch (err: unknown) {
                return {
                    content: [{
                        type: "text",
                        text: `rag_index backend selection failed: ${errorMessage(err)}`,
                    }],
                };
            }

            const stats: IndexStats = {
                filesDiscovered: 0,
                filesSkipped: 0,
                filesIndexed: 0,
                chunksCreated: 0,
                errors: [],
            };

            try {
                // In direct mode, ensure the schema exists. The HTTP
                // server is responsible for its own bootstrap.
                if (backend.kind === "direct") {
                    await ensureSchema(backend.db, cfg.embeddingDim);
                }

                // 1. Discover files using the configured patterns/excludes.
                const allExtensions = new Set([
                    ...cfg.codeExtensions,
                    ...cfg.defaultExtensions,
                    ...cfg.configExtensions,
                ]);
                const patterns = Array.from(allExtensions).map((ext) => `**/*.${ext}`);
                const excludePatterns = [
                    "**/node_modules/**",
                    "**/dist/**",
                    "**/build/**",
                    "**/.git/**",
                    "**/vendor/**",
                    "**/target/**",
                    "**/coverage/**",
                ];

                const matched: string[] = [];
                for (const pattern of patterns) {
                    const hits = await glob(pattern, {
                        cwd: root,
                        absolute: true,
                        ignore: excludePatterns,
                        nodir: true,
                    });
                    matched.push(...hits);
                }
                const uniqueFiles = [...new Set(matched)];
                stats.filesDiscovered = uniqueFiles.length;

                if (!uniqueFiles.length) {
                    return {
                        content: [{
                            type: "text",
                            text:
                                `No files matched the configured patterns under ${root}. ` +
                                `Configured extensions: ${[...allExtensions].join(", ")}.`,
                        }],
                    };
                }

                // 2. Get existing per-doc hashes for incremental sync.
                const existingHashes: Map<string, SourceMapEntry> =
                    backend.kind === "http"
                        ? await fetchSourcesHttp(backend.apiBase, backend.apiKey, storeId, projectId)
                        : await getExistingSourceHashes(backend.db, storeId, projectId);

                // 3. Filter to files that need (re)indexing.
                const filesToProcess: { filePath: string; content: string; hash: string; relPath: string }[] = [];
                for (const filePath of uniqueFiles) {
                    try {
                        const content = await fs.readFile(filePath, "utf-8");
                        const hash = fileHash(content);
                        const relPath = path.relative(root, filePath);

                        if (existingHashes.get(relPath)?.content_hash === hash) {
                            stats.filesSkipped++;
                            continue;
                        }
                        filesToProcess.push({ filePath, content, hash, relPath });
                    } catch (err: unknown) {
                        stats.errors.push(`Read error ${filePath}: ${errorMessage(err)}`);
                    }
                }

                if (!filesToProcess.length) {
                    return {
                        content: [{
                            type: "text",
                            text: [
                                `Index is up to date for ${root}`,
                                `* ${stats.filesDiscovered} files discovered`,
                                `* ${stats.filesSkipped} files unchanged (skipped)`,
                                `* 0 files needed reindexing`,
                            ].join("\n"),
                        }],
                    };
                }

                // 4. Chunk + embed each file. Always async — chunkFile
                // returns Promise<ChunkResult[]>.
                const embedder = new LiteLLMEmbeddingService(
                    cfg.litellmBase,
                    cfg.litellmKey,
                    cfg.embeddingModel
                );

                const allItems: HttpBatchItem[] = [];
                const docsToReindex: string[] = [];
                const batchSize = cfg.embedBatchSize;

                for (const { filePath, content, hash, relPath } of filesToProcess) {
                    try {
                        const chunks: ChunkResult[] = await chunkFile(filePath, {
                            maxChunkChars: cfg.codeChunkMaxChars,
                            content,
                            repoRoot: root,
                            projectId,
                        });
                        if (chunks.length === 0) continue;

                        docsToReindex.push(relPath);

                        for (let i = 0; i < chunks.length; i += batchSize) {
                            const batchChunks = chunks.slice(i, i + batchSize);
                            const texts = batchChunks.map((c) => c.semanticContent || c.content);
                            const embeddings = await embedder.generateBatch(texts);

                            for (let j = 0; j < batchChunks.length; j++) {
                                const chunk = batchChunks[j];
                                allItems.push({
                                    content: chunk.content,
                                    semantic_content: chunk.semanticContent,
                                    lexical_content: chunk.lexicalContent,
                                    embedding: embeddings[j],
                                    metadata: {
                                        ...chunk.metadata,
                                        // Defensive override: rag-index is the canonical owner of
                                        // content_hash for incremental-sync purposes. The chunker
                                        // also writes a hash to chunk.metadata, but tying the
                                        // stored hash to rag-index's local `hash` variable
                                        // guarantees the comparison at line ~348
                                        // (existingHashes.get(relPath)?.content_hash === hash)
                                        // can never silently break if chunker behavior drifts.
                                        content_hash: hash,
                                        project_id: projectId,
                                        repo_root: root,
                                    },
                                });
                            }
                        }

                        stats.filesIndexed++;
                    } catch (err: unknown) {
                        stats.errors.push(`Chunk/embed error ${filePath}: ${errorMessage(err)}`);
                    }
                }

                stats.chunksCreated = allItems.length;

                // 5. Stale-doc delete + bulk insert via the chosen backend.
                if (backend.kind === "http") {
                    if (docsToReindex.length > 0) {
                        await deleteDocsHttp(
                            backend.apiBase,
                            backend.apiKey,
                            storeId,
                            projectId,
                            docsToReindex
                        );
                    }
                    if (allItems.length > 0) {
                        // Chunk into reasonable HTTP batches so a single
                        // request body doesn't grow unbounded.
                        const HTTP_BATCH = 100;
                        for (let i = 0; i < allItems.length; i += HTTP_BATCH) {
                            await postBatchHttp(
                                backend.apiBase,
                                backend.apiKey,
                                storeId,
                                allItems.slice(i, i + HTTP_BATCH),
                                projectId
                            );
                        }
                    }
                } else {
                    if (docsToReindex.length > 0) {
                        await deleteEmbeddingsForDocs(backend.db, storeId, projectId, docsToReindex);
                    }
                    if (allItems.length > 0) {
                        const inserts: EmbeddingInsertItem[] = allItems.map((it) => ({
                            content: it.content,
                            semantic_content: it.semantic_content,
                            lexical_content: it.lexical_content,
                            embedding: it.embedding,
                            metadata: it.metadata,
                        }));
                        await insertEmbeddingBatch(backend.db, storeId, inserts);
                    }
                }

                // 6. Report
                const report = [
                    `Indexed ${root} (backend: ${backend.kind})`,
                    "",
                    `Stats:`,
                    `* ${stats.filesDiscovered} files discovered`,
                    `* ${stats.filesSkipped} files unchanged (skipped)`,
                    `* ${stats.filesIndexed} files indexed`,
                    `* ${stats.chunksCreated} chunks created & embedded`,
                    ...(stats.errors.length > 0
                        ? [
                              "",
                              `${stats.errors.length} errors:`,
                              ...stats.errors.slice(0, 10).map((e) => `  * ${e}`),
                              ...(stats.errors.length > 10
                                  ? [`  ... and ${stats.errors.length - 10} more`]
                                  : []),
                          ]
                        : []),
                ];

                return {
                    content: [{ type: "text", text: report.join("\n") }],
                };
            } catch (err) {
                return {
                    content: [{
                        type: "text",
                        text: `RAG indexing failed: ${errorMessage(err)}`,
                    }],
                };
            } finally {
                if (backend.kind === "direct") {
                    await backend.db.close();
                }
            }
        }
    );
}
