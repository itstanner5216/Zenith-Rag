/**
 * rag-search.ts — MCP tool for RAG-powered codebase search.
 *
 * Minimal wrapper: user provides query, tool resolves project scope
 * and delegates to the VectorStoreSearchEngine.
 */

import { z } from "zod";
import type { ToolContext, ToolServer } from "./types.js";
import { resolveProjectRoot, getProjectId } from "../utils/project-scope.js";
import { createSearchEngine, type VectorStoreSearchEngine, type SearchResponse } from "../lib/vector-store.js";
import { createVectorDB } from "../lib/db-adapter.js";
import { loadConfig, toVectorStoreConfig } from "../lib/config.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Engine Singleton
// ═══════════════════════════════════════════════════════════════════════════════

let _engine: VectorStoreSearchEngine | null = null;
let _db: ReturnType<typeof createVectorDB> | null = null;

function getEngine(): VectorStoreSearchEngine {
  if (_engine) return _engine;
  const cfg = loadConfig();
  _db = createVectorDB({ databaseUrl: cfg.databaseUrl });
  _engine = createSearchEngine(_db, toVectorStoreConfig(cfg));
  ensureExitHandler();
  return _engine;
}

/**
 * Release the search engine's DB pool. Hosts that manage the rag_search
 * tool lifecycle should call this on shutdown. Idempotent — safe to call
 * even when no engine has been built yet, or after an earlier close.
 */
export async function closeEngine(): Promise<void> {
  const db = _db;
  _engine = null;
  _db = null;
  if (db) {
    try {
      await db.close();
    } catch (err: unknown) {
      // Closing twice on pg pool throws; we already null'd, so ignore.
      console.warn(`[rag_search] closeEngine: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

// Best-effort cleanup for stand-alone runs. Hosts that own the lifecycle
// should still call closeEngine() explicitly — beforeExit only fires when
// the event loop is empty, not on SIGKILL or hard crashes.
let _exitHandlerRegistered = false;
function ensureExitHandler(): void {
  if (_exitHandlerRegistered) return;
  _exitHandlerRegistered = true;
  process.once("beforeExit", () => {
    void closeEngine();
  });
  // SIGTERM/SIGINT — synchronous best-effort close before process death.
  // We can't await here so we kick the close and let it race; the host
  // is the durable shutdown owner.
  for (const sig of ["SIGTERM", "SIGINT"] as const) {
    process.once(sig, () => {
      void closeEngine();
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Formatting
// ═══════════════════════════════════════════════════════════════════════════════

function formatResults(response: SearchResponse): string {
  if (response.results.length === 0) {
    return [
      `🔍 No results for "${response.query}"${response.projectId ? ` in project: ${response.projectId}` : ''}`,
      '',
      'Possible reasons:',
      '• Project hasn\'t been indexed yet — run rag-upload first',
      '• Query doesn\'t match any indexed content',
      '• Try broader or different search terms',
    ].join('\n');
  }

  const lines: string[] = [
    `Found ${response.resultCount} result${response.resultCount !== 1 ? "s" : ""} for "${response.query}" (${response.retrievalMode}):`,
    "",
  ];

  for (const item of response.results) {
    const meta = item.metadata;
    const loc = meta?.start_line && meta?.end_line
      ? `:${meta.start_line}-${meta.end_line}`
      : meta?.start_line ? `:${meta.start_line}` : "";
    const symbol = meta?.symbol_name ? ` (${meta.symbol_name})` : "";
    const score = item.score.toFixed(4);

    lines.push(`── ${item.source}${loc}${symbol}  [${score}]`);
    lines.push(item.content);
    lines.push("");
  }

  return lines.join("\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Registration
// ═══════════════════════════════════════════════════════════════════════════════

// content_type is an enum on the MCP wire — values map to the
// underlying metadata content_type strings written by the indexer:
//   "code" → metadata.content_type = "code"
//   "docs" → metadata.content_type IN ("markdown", "text") (logical OR)
//   "all"  → no filter
//
// "docs" maps to two underlying values, so we can't push it through
// the single-value metadata filter — the search engine expands it
// internally below by selecting which filter to apply.
const ContentTypeEnum = z.enum(["code", "docs", "all"]);
type ContentType = z.infer<typeof ContentTypeEnum>;

export function register(server: ToolServer, ctx: ToolContext): void {
  server.registerTool("rag_search", {
    title: "RAG Search",
    description:
      "Search the codebase using hybrid semantic + lexical retrieval. " +
      "Finds relevant code, documentation, and configuration across the project. " +
      "Results include file paths, line numbers, and relevance scores.",
    inputSchema: {
      query: z.string().describe(
        "Natural language search query. Can be a concept ('rate limiting'), " +
        "a symbol name ('getUserById'), or a question ('how does auth work')."
      ),
      limit: z.number().optional().default(10).describe(
        "Maximum number of results to return (1-50)."
      ),
      content_type: ContentTypeEnum.optional().describe(
        "Filter by content type: 'code' (source files), 'docs' (markdown/text), or 'all'."
      ),
    },
    annotations: { readOnlyHint: true, idempotentHint: true, destructiveHint: false },
  }, async (args: { query: string; limit?: number; content_type?: ContentType }) => {
    // Resolve project scope from the MCP host's allowed directories.
    // Zenith-MCP's ToolContext exposes getAllowedDirectories() — we
    // do NOT read ctx.cwd or ctx.allowedDirectories directly because
    // those fields aren't part of the canonical contract. When no
    // allowed directories are configured (e.g. running outside an
    // MCP host), fall back to process.cwd().
    const allowedDirs = ctx.getAllowedDirectories();
    const scopeStart = allowedDirs.length > 0 ? allowedDirs[0] : process.cwd();
    const projectRoot = resolveProjectRoot(scopeStart, {
      allowedDirectories: allowedDirs,
    });
    const projectId = projectRoot ? (getProjectId(projectRoot) ?? projectRoot) : undefined;

    // Build filters. content_type "all" means no filter; "code"
    // narrows to source code; "docs" filters for "markdown" (the
    // dominant documentation format). Text/RST files also qualify as
    // docs but the single-value metadata filter can't express OR,
    // so we use "markdown" as the best single-value approximation.
    const filters: Record<string, string> = {};
    if (projectId) filters.project_id = projectId;
    if (projectRoot) filters.repo_root = projectRoot;
    if (args.content_type === "code") {
      filters.content_type = "code";
    } else if (args.content_type === "docs") {
      // "docs" primarily targets markdown content. Text/RST files also qualify
      // but the single-value metadata filter can't express an OR condition.
      // Filtering for markdown covers the dominant documentation format.
      filters.content_type = "markdown";
    }
    // "all" intentionally leaves content_type unfiltered.

    // Execute search
    const engine = getEngine();
    const response = await engine.search({
      query: args.query,
      filters,
      limit: Math.min(Math.max(args.limit || 10, 1), 50),
      returnMetadata: true,
    });

    return {
      content: [{ type: "text", text: formatResults(response) }],
    };
  });
}

