// ---------------------------------------------------------------------------
// http.ts — Zenith-Rag HTTP server (Express)
//
// Provides the same surface as the Python pgvector server so existing
// clients (and the rag-upload CLI in non-direct-db mode) drop in
// without changes:
//
//   POST   /v1/vector_stores/:store_id/search
//   POST   /v1/vector_stores/:store_id/embeddings/batch
//   DELETE /v1/vector_stores/:store_id/embeddings
//   GET    /v1/vector_stores/:store_id/stats
//   GET    /v1/vector_stores/:store_id/sources
//   GET    /health   (no auth)
//
// All /v1 routes are gated by bearer-token auth using lib/server-auth.ts;
// /health is intentionally unauthed so probes and the CLI's connectivity
// check still work without a key.
//
// Run via: `npm run start:api` (compiled) or `npm run dev:api` (tsx).
// ---------------------------------------------------------------------------

import express, { type Request, type Response } from "express";
import type { Server } from "http";
import { loadConfig, toVectorStoreConfig } from "../lib/config.js";
import { createVectorDB, type VectorStoreDB } from "../lib/db-adapter.js";
import {
    ensureSchema,
    insertEmbeddingBatch,
    deleteEmbeddingsForDocs,
    ALLOWED_FILTER_KEYS,
    type EmbeddingInsertItem,
} from "../lib/schema.js";
import {
    VectorStoreSearchEngine,
    LiteLLMEmbeddingService,
    createSearchEngine,
} from "../lib/vector-store.js";
import { authMiddleware, requireApiKey } from "../lib/server-auth.js";

export interface ServerHandle {
    app: express.Application;
    db: VectorStoreDB;
    close: () => Promise<void>;
}

// Per-item shape accepted by the batch insert endpoint. Both the
// legacy and the Python-parity wire shapes deliver an array of these.
interface BatchEmbeddingInput {
    content: string;
    embedding?: number[];
    semantic_content?: string;
    lexical_content?: string;
    metadata?: Record<string, unknown>;
}

export function createServer(): ServerHandle {
    const app = express();
    app.use(express.json({ limit: "50mb" }));

    const cfg = loadConfig();

    // Resolve API key. requireApiKey throws if no key is set AND
    // allowNoAuthForTests is false. When the test bypass IS set we
    // intentionally accept an empty string (means "no auth").
    const apiKey = requireApiKey(cfg.allowNoAuthForTests);

    // Conditional auth middleware — only mounted when there's a key
    if (apiKey) {
        app.use("/v1", authMiddleware(apiKey));
    } else {
        // No API key set AND allowNoAuthForTests=true. Loud warning so
        // a misconfigured prod deployment isn't silently wide-open.
        console.warn(
            "[ZENITH-RAG][AUTH] Bearer auth is DISABLED. " +
            "Anyone who can reach the /v1 endpoints can read AND write to the vector store. " +
            "Set ZENITH_RAG_API_KEY (or PGVECTOR_API_KEY / API_KEY) and unset ALLOW_NO_AUTH_FOR_TESTS to enable auth."
        );
    }

    const db = createVectorDB({ databaseUrl: cfg.databaseUrl });
    const embedder = new LiteLLMEmbeddingService(
        cfg.litellmBase,
        cfg.litellmKey,
        cfg.embeddingModel
    );

    // Per-store engine cache — each engine carries a persistent
    // BM25Cache so popular stores don't pay the index-build cost on
    // every request. This mirrors Python's _bm25_cache (main.py:96)
    // keyed by (vector_store_id, repo_root, project_id), but here we
    // only need to key by storeId because BM25Cache itself further
    // shards by (storeId, projectId, repoRoot) internally.
    //
    // Building per-request would correctly route the storeId override
    // but rebuild the BM25 index from Postgres every search — the
    // observable behavior the Python baseline explicitly avoids.
    const engineByStore = new Map<string, VectorStoreSearchEngine>();
    const baseConfig = toVectorStoreConfig(cfg);
    function getOrCreateEngine(storeId: string): VectorStoreSearchEngine {
        const cached = engineByStore.get(storeId);
        if (cached) return cached;
        const engine = createSearchEngine(db, baseConfig, { storeId });
        engineByStore.set(storeId, engine);
        return engine;
    }

    // Health check (no auth) — mirrors Python baseline: SELECT 1 to
    // verify DB connectivity and returns {status, timestamp, database}.
    app.get("/health", async (_req: Request, res: Response) => {
        let dbOk = true;
        try {
            await db.query("SELECT 1");
        } catch {
            dbOk = false;
        }

        const status = dbOk ? "healthy" : "degraded";
        res.json({
            status,
            timestamp: Math.floor(Date.now() / 1000),
            database: dbOk ? "connected" : "unreachable",
        });
    });

    // POST /v1/vector_stores/:store_id/search
    app.post("/v1/vector_stores/:store_id/search", async (req: Request, res: Response) => {
        try {
            const { store_id } = req.params;
            const {
                query, limit, filters, content_type,
                lexical_only, return_metadata, return_raw_content,
            } = req.body ?? {};
            if (typeof query !== "string" || query.length === 0) {
                res.status(400).json({ error: "query (string) required" });
                return;
            }

            const mergedFilters: Record<string, string> = { ...(filters ?? {}) };
            if (typeof content_type === "string" && content_type.length > 0) {
                mergedFilters.content_type = content_type;
            }

            // Whitelist validation — reject malformed/malicious filter keys at the
            // HTTP boundary so the engine never sees them. The engine has its own
            // defense-in-depth check, but we want a clean 400 for callers rather
            // than a 500 from deep inside the search pipeline.
            for (const key of Object.keys(mergedFilters)) {
                if (!ALLOWED_FILTER_KEYS.has(key)) {
                    res.status(400).json({
                        error: `Invalid filter key '${key}'. Allowed: ${Array.from(ALLOWED_FILTER_KEYS).join(", ")}`,
                    });
                    return;
                }
            }

            const storeEngine = getOrCreateEngine(store_id);
            const result = await storeEngine.search({
                query,
                limit: typeof limit === "number" ? limit : cfg.defaultLimit,
                filters: mergedFilters,
                lexicalOnly: lexical_only === true,
                returnMetadata: return_metadata !== false,
                returnRawContent: return_raw_content === true,
            });
            res.json(result);
        } catch (err) {
            console.error("Search error:", err);
            res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
        }
    });

    // POST /v1/vector_stores/:store_id/embeddings/batch
    //
    // Body shape matches Python EmbeddingBatchCreateRequest:
    //   { "embeddings": [ { content, embedding?, semantic_content?, lexical_content?, metadata? }, ... ],
    //     "project_id": "..."  // optional; merged into per-item metadata if present }
    //
    // The legacy "items" key is also accepted for forward-compat with
    // any client that already moved to that name, but new clients should
    // prefer "embeddings" since that is the documented Python shape.
    //
    // If a per-item `embedding` array is supplied, it is used verbatim
    // (so callers may pre-embed with their own model). If absent, the
    // server falls back to LiteLLMEmbeddingService.generateBatch over
    // the semantic_content (or content).
    app.post(
        "/v1/vector_stores/:store_id/embeddings/batch",
        async (req: Request, res: Response) => {
            try {
                const { store_id } = req.params;
                const body = (req.body ?? {}) as {
                    embeddings?: BatchEmbeddingInput[];
                    items?: BatchEmbeddingInput[];
                    project_id?: string;
                };
                const incoming: BatchEmbeddingInput[] | undefined = body.embeddings ?? body.items;
                const project_id = body.project_id;

                if (!Array.isArray(incoming) || incoming.length === 0) {
                    res.status(400).json({ error: "embeddings array required" });
                    return;
                }

                // Generate embeddings for any items that don't carry one.
                // We embed in a single batched call when needed so this
                // endpoint stays O(1) HTTP roundtrips per request.
                const missingIdxs: number[] = [];
                const missingTexts: string[] = [];
                for (let i = 0; i < incoming.length; i++) {
                    const item = incoming[i];
                    if (!Array.isArray(item.embedding) || item.embedding.length === 0) {
                        missingIdxs.push(i);
                        missingTexts.push(item.semantic_content ?? item.content);
                    }
                }

                let generated: number[][] = [];
                if (missingTexts.length > 0) {
                    generated = await embedder.generateBatch(missingTexts);
                }

                const insertItems: EmbeddingInsertItem[] = incoming.map((item, idx) => {
                    let vec: number[];
                    if (Array.isArray(item.embedding) && item.embedding.length > 0) {
                        vec = item.embedding;
                    } else {
                        const slot = missingIdxs.indexOf(idx);
                        vec = generated[slot];
                    }
                    return {
                        content: item.content,
                        semantic_content: item.semantic_content ?? item.content,
                        lexical_content: item.lexical_content ?? item.content,
                        embedding: vec,
                        metadata: {
                            ...(item.metadata ?? {}),
                            ...(project_id ? { project_id } : {}),
                        },
                    };
                });

                await insertEmbeddingBatch(db, store_id, insertItems);

                // Invalidate the BMX+ cache so the next search picks up
                // the freshly-inserted embeddings. Mirrors the delete
                // endpoint's invalidation and the Python baseline's
                // post-write cache bust.
                if (project_id) {
                    getOrCreateEngine(store_id).invalidateBmxCache(store_id, project_id);
                } else {
                    getOrCreateEngine(store_id).invalidateBmxCache(store_id);
                }

                res.json({ inserted: incoming.length });
            } catch (err) {
                console.error("Batch insert error:", err);
                res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
            }
        }
    );

    // DELETE /v1/vector_stores/:store_id/embeddings
    //
    // Python-baseline shape: query params doc_path and project_id, with
    // the rule "at least one filter required" (main.py:1546-1550).
    // We also accept a JSON body shape ({ project_id, doc_path,
    // doc_paths }) for clients that prefer body params, but query
    // params win when both are supplied so existing CLIs that send
    // ?doc_path= keep working unchanged.
    app.delete("/v1/vector_stores/:store_id/embeddings", async (req: Request, res: Response) => {
        try {
            const { store_id } = req.params;

            const body = (req.body ?? {}) as {
                project_id?: unknown;
                doc_path?: unknown;
                doc_paths?: unknown;
            };

            const project_id =
                typeof req.query.project_id === "string"
                    ? req.query.project_id
                    : typeof body.project_id === "string"
                    ? body.project_id
                    : undefined;

            const docPaths: string[] = [];
            if (typeof req.query.doc_path === "string" && req.query.doc_path.length > 0) {
                docPaths.push(req.query.doc_path);
            } else if (typeof body.doc_path === "string" && body.doc_path.length > 0) {
                docPaths.push(body.doc_path);
            } else if (Array.isArray(body.doc_paths)) {
                for (const p of body.doc_paths) {
                    if (typeof p === "string" && p.length > 0) docPaths.push(p);
                }
            }

            // Python parity: at least one of doc_path / project_id required
            if (!project_id && docPaths.length === 0) {
                res.status(400).json({
                    error: "At least one filter (doc_path or project_id) is required",
                });
                return;
            }

            // Schema helper requires a project_id and an explicit doc_path
            // list. Two cases to handle:
            //   1. doc_paths supplied (with or without project_id) — bulk
            //      delete by doc_path within the project (or all projects
            //      if no project_id was given — handled by raw SQL below).
            //   2. project_id only — wipe every chunk for that project.
            let deleted = 0;
            if (docPaths.length > 0 && project_id) {
                deleted = await deleteEmbeddingsForDocs(db, store_id, project_id, docPaths);
            } else if (docPaths.length > 0 && !project_id) {
                // Delete-by-doc_path across all projects in the store.
                const placeholders = docPaths.map((_, i) => `$${i + 2}`).join(", ");
                const result = await db.query<{ count: string }>(
                    `WITH deleted AS (
                         DELETE FROM embeddings
                         WHERE vector_store_id = $1
                           AND metadata->>'doc_path' IN (${placeholders})
                         RETURNING id
                     )
                     SELECT COUNT(*)::text AS count FROM deleted`,
                    [store_id, ...docPaths]
                );
                deleted = parseInt(result[0]?.count ?? "0", 10);
            } else {
                // project_id only — wipe everything for that project.
                const result = await db.query<{ count: string }>(
                    `WITH deleted AS (
                         DELETE FROM embeddings
                         WHERE vector_store_id = $1
                           AND metadata->>'project_id' = $2
                         RETURNING id
                     )
                     SELECT COUNT(*)::text AS count FROM deleted`,
                    [store_id, project_id]
                );
                deleted = parseInt(result[0]?.count ?? "0", 10);
            }

            // Invalidate cached BMX+ index for this store/project — the
            // next search rebuilds from the now-smaller corpus. If we
            // didn't have a project_id, drop the whole-store cache.
            if (project_id) {
                getOrCreateEngine(store_id).invalidateBmxCache(store_id, project_id);
            } else {
                getOrCreateEngine(store_id).invalidateBmxCache(store_id);
            }

            res.json({ deleted });
        } catch (err) {
            console.error("Delete error:", err);
            res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
        }
    });

    // GET /v1/vector_stores/:store_id/stats
    app.get("/v1/vector_stores/:store_id/stats", async (req: Request, res: Response) => {
        try {
            const { store_id } = req.params;
            const project_id = typeof req.query.project_id === "string"
                ? req.query.project_id
                : undefined;

            const params: unknown[] = [store_id];
            let whereClause = "WHERE vector_store_id = $1";
            if (project_id) {
                whereClause += " AND metadata->>'project_id' = $2";
                params.push(project_id);
            }

            const countRows = await db.query<{ count: string }>(
                `SELECT COUNT(*)::text AS count FROM embeddings ${whereClause}`,
                params
            );
            const typeRows = await db.query<{ content_type: string | null; count: string }>(
                `SELECT metadata->>'content_type' AS content_type, COUNT(*)::text AS count
                 FROM embeddings ${whereClause}
                 GROUP BY metadata->>'content_type'`,
                params
            );

            res.json({
                total_chunks: parseInt(countRows[0]?.count ?? "0", 10),
                by_content_type: Object.fromEntries(
                    typeRows.map((r) => [r.content_type ?? "unknown", parseInt(r.count, 10)])
                ),
            });
        } catch (err) {
            console.error("Stats error:", err);
            res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
        }
    });

    // GET /v1/vector_stores/:store_id/sources
    // Used by rag-upload --sync and --status to get the per-document
    // hash/chunk-count map needed for incremental sync planning.
    app.get("/v1/vector_stores/:store_id/sources", async (req: Request, res: Response) => {
        try {
            const { store_id } = req.params;
            const project_id = typeof req.query.project_id === "string"
                ? req.query.project_id
                : undefined;

            const params: unknown[] = [store_id];
            let sourcesWhere = "WHERE vector_store_id = $1";
            if (project_id) {
                sourcesWhere += " AND metadata->>'project_id' = $2";
                params.push(project_id);
            }

            const rows = await db.query<{
                doc_path: string | null;
                content_hash: string | null;
                chunk_count: string;
            }>(
                `SELECT
                     metadata->>'doc_path'      AS doc_path,
                     metadata->>'content_hash'  AS content_hash,
                     COUNT(*)::text             AS chunk_count
                 FROM embeddings
                 ${sourcesWhere}
                 GROUP BY metadata->>'doc_path', metadata->>'content_hash'`,
                params
            );

            // Only emit rows that have BOTH doc_path and content_hash.
            // A null content_hash means the row was inserted before the
            // hash was added (or by a non-conforming writer) and is
            // useless for sync planning — including it would cause
            // rag-upload --sync to mis-classify the doc as "new" on
            // every run because no local hash will ever equal `null`.
            // Better to omit it and let the next ingest re-stamp the
            // hash.
            const sources: Record<string, { content_hash: string; chunk_count: number }> = {};
            for (const row of rows) {
                if (row.doc_path && row.content_hash) {
                    sources[row.doc_path] = {
                        content_hash: row.content_hash,
                        chunk_count: parseInt(row.chunk_count, 10),
                    };
                }
            }

            res.json({ sources });
        } catch (err) {
            console.error("Sources error:", err);
            res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
        }
    });

    return {
        app,
        db,
        close: async () => {
            await db.close();
        },
    };
}

export async function startServer(port?: number, host?: string): Promise<Server> {
    const cfg = loadConfig();
    const handle = createServer();

    // Idempotent schema bootstrap so a fresh DB comes up automatically.
    await ensureSchema(handle.db, cfg.embeddingDim);

    const finalPort = port ?? cfg.port;
    const finalHost = host ?? cfg.host;

    const server = handle.app.listen(finalPort, finalHost, () => {
        console.log(`Zenith-RAG server listening on http://${finalHost}:${finalPort}`);
    });

    // Graceful shutdown so docker stop / kubectl rollouts don't drop
    // an in-flight transaction.
    //
    // Order:
    //   1. Stop accepting new connections AND wait for in-flight
    //      requests to drain (this is the part http.Server.close
    //      actually does — calling it without awaiting the callback
    //      was the bug in the previous implementation, which would
    //      race the DB pool teardown ahead of unfinished requests).
    //   2. Close the DB pool.
    //   3. Exit.
    //
    // A 10-second hard cap protects against hung clients keeping
    // sockets open forever; `server.closeAllConnections()` is also
    // available on Node 18.2+ if we ever need to truly force-close.
    //
    // shuttingDown latch makes the handler idempotent — multiple
    // SIGTERMs in flight (e.g. supervisord retries) won't try to
    // close the DB pool twice (pg's pool.end is not idempotent).
    let shuttingDown = false;
    const SHUTDOWN_TIMEOUT_MS = 10_000;
    const shutdown = async (signal: string): Promise<void> => {
        if (shuttingDown) return;
        shuttingDown = true;
        console.log(`Received ${signal} — shutting down...`);

        const closeServer = new Promise<void>((resolve, reject) => {
            server.close((err?: Error) => (err ? reject(err) : resolve()));
        });

        const timeout = new Promise<void>((_resolve, reject) =>
            setTimeout(
                () =>
                    reject(
                        new Error(
                            `Server did not close within ${SHUTDOWN_TIMEOUT_MS}ms — forcing exit`
                        )
                    ),
                SHUTDOWN_TIMEOUT_MS
            )
        );

        try {
            await Promise.race([closeServer, timeout]);
        } catch (err) {
            console.warn(`Shutdown wait failed: ${err instanceof Error ? err.message : String(err)}`);
        }

        try {
            await handle.close();
        } catch (err) {
            console.warn(`DB pool close failed: ${err instanceof Error ? err.message : String(err)}`);
        }

        process.exit(0);
    };
    process.on("SIGTERM", () => void shutdown("SIGTERM"));
    process.on("SIGINT", () => void shutdown("SIGINT"));

    return server;
}

// CLI entry — run when invoked directly via `node dist/server/http.js`
// (or `tsx server/http.ts`). Uses import.meta.url comparison instead of
// `require.main === module` since this is an ESM module.
const isDirectInvocation =
    process.argv[1] && import.meta.url === `file://${process.argv[1]}`;

if (isDirectInvocation) {
    startServer().catch((err) => {
        console.error("Failed to start Zenith-RAG server:", err);
        process.exit(1);
    });
}
