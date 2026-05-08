/**
 * vector-store.ts — Hybrid retrieval engine for PGVector-backed codebase search.
 *
 * Implements the full search pipeline:
 *   1. Query preprocessing (identifier token expansion)
 *   2. Parallel semantic (pgvector cosine) + lexical (BMX+) retrieval
 *   3. Convex combination score fusion
 *   4. Per-file diversity capping
 *   5. Cross-encoder reranking
 *   6. Post-rerank deduplication
 *   7. Code-aware result decoration
 *
 * Designed for Zenith-MCP native integration. Uses Zenith's project scoping
 * and BMX+ engine directly.
 */

import { BMXPlusIndex } from "./bmx-plus.js";
import { ALLOWED_FILTER_KEYS } from "./schema.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface VectorStoreConfig {
  /** PGVector API base URL */
  apiBase: string;
  /** API key for auth */
  apiKey: string;
  /** Default vector store UUID */
  storeId: string;
  /** Embedding provider base URL (LiteLLM/Voyage) */
  embeddingBase: string;
  /** Embedding provider API key */
  embeddingKey: string;
  /** Embedding model identifier */
  embeddingModel: string;
  /** Reranker model (empty string = disabled) */
  rerankModel: string;
  /** Reranker timeout in ms */
  rerankTimeout: number;
  /** Max rerank retry attempts */
  rerankMaxAttempts: number;
  /** Retry delays in seconds (comma-separated) */
  rerankRetryDelays: string;
  /** Number of candidates to send to reranker */
  rerankCandidates: number;
  /** Top N results from reranker */
  rerankTopN: number;
  /** Semantic weight in convex combination [0,1] */
  semanticWeight: number;
  /** BMX+ weight in convex combination [0,1] */
  bm25Weight: number;
  /** Enable hybrid (semantic + BMX+) search */
  hybridEnabled: boolean;
  /** Default result limit */
  defaultLimit: number;
  /** Max result limit */
  maxLimit: number;
  /** Minimum similarity score threshold */
  minScore: number;
  /** BMX+ candidate pool size */
  bm25Candidates: number;
  /** Page size for loading BM25 corpus */
  bm25PageSize: number;
  /** Distance operator for pgvector (<=> cosine, <#> inner product, <-> L2) */
  distanceOperator: "<=>" | "<#>" | "<->";
  /** Prepend code-safety warning to code results */
  codeAwareEnabled: boolean;
  /** Warning text prepended to code results */
  codeResultWarning: string;
  /** Database connection string */
  databaseUrl: string;
  /** Max per-file results before reranking */
  maxPerFilePreRerank: number;
  /** Enable symbol-first fast path before semantic search (default true) */
  symbolFastPathEnabled: boolean;
  /** Maximum results to return from exact symbol match tier */
  symbolExactLimit: number;
  /** Maximum results to return from fuzzy symbol match tier */
  symbolFuzzyLimit: number;
}

export interface SearchFilters {
  project_id?: string;
  repo_root?: string;
  content_type?: string;
  [key: string]: string | undefined;
}

export interface SearchRequest {
  query: string;
  filters?: SearchFilters;
  limit?: number;
  lexicalOnly?: boolean;
  returnMetadata?: boolean;
  returnRawContent?: boolean;
}

export interface ChunkMetadata {
  doc_path?: string;
  source?: string;
  filename?: string;
  content_type?: string;
  language?: string;
  symbol_name?: string;
  chunk_type?: string;
  start_line?: number;
  end_line?: number;
  chunk_chars?: number;
  project_id?: string;
  repo_root?: string;
  content_hash?: string;
  [key: string]: any;
}

export interface SearchResultItem {
  id: string;
  score: number;
  source: string;
  content: string;
  rawContent?: string | null;
  metadata: ChunkMetadata | null;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  resultCount: number;
  retrievalMode: "hybrid" | "lexical" | "semantic" | "symbol" | "symbol+hybrid" | "symbol+lexical";
  projectId?: string;
}

/** Internal candidate during scoring pipeline */
interface SearchCandidate {
  chunkId: string;
  content: string;
  semanticContent: string;
  lexicalContent: string;
  metadata: ChunkMetadata;
  score: number;
  semanticScore?: number;
  bmxScore?: number;
  rerankScore?: number;
}

/** Raw row from PGVector API or direct DB query */
interface RawEmbeddingRow {
  id: string;
  content: string;
  semantic_content?: string;
  lexical_content?: string;
  metadata: ChunkMetadata;
  distance?: number;
}

/** BMX+ index cache entry */
interface BM25CacheEntry {
  index: BMXPlusIndex;
  rowsById: Map<string, RawEmbeddingRow>;
  builtAt: number;
  projectId: string | null;
  repoRoot: string | null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

const CAMEL_SPLIT_RE = /([a-z])([A-Z])/g;
const ALLCAPS_SPLIT_RE = /([A-Z]+)([A-Z][a-z])/g;
const HAS_CAMEL_RE = /[a-z][A-Z]/;
const HAS_ALLCAPS_TRANSITION_RE = /[A-Z]{2,}[a-z]/;

/**
 * Regex for the symbol fast path — matches bare identifiers and dotted
 * identifiers (e.g. "foo", "ns.foo", "Cls$inner").
 *
 * @internal Exported for unit testing. Used by VectorStoreSearchEngine.isSymbolQuery.
 */
export const SYMBOL_QUERY_RE = /^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$/;

/** Cache TTL for BMX+ indexes — 5 minutes */
const BM25_CACHE_TTL_MS = 5 * 60 * 1000;

// ═══════════════════════════════════════════════════════════════════════════════
// Default Configuration
// ═══════════════════════════════════════════════════════════════════════════════

export const DEFAULT_CONFIG: VectorStoreConfig = {
  apiBase: "http://localhost:8200",
  apiKey: "",
  storeId: "49e09fac-3634-4df4-9837-f90a237cb7a8",
  embeddingBase: "http://localhost:4000",
  embeddingKey: "",
  embeddingModel: "voyage-4-large",
  rerankModel: "rerank-2",
  rerankTimeout: 30_000,
  rerankMaxAttempts: 3,
  rerankRetryDelays: "1,2,4",
  rerankCandidates: 40,
  rerankTopN: 15,
  semanticWeight: 0.6,
  bm25Weight: 0.4,
  hybridEnabled: true,
  defaultLimit: 10,
  maxLimit: 50,
  minScore: 0.05,
  bm25Candidates: 40,
  bm25PageSize: 5000,
  distanceOperator: "<=>",
  codeAwareEnabled: true,
  codeResultWarning:
    "⚠️ This is a retrieved code snippet. Verify against the source before using.",
  databaseUrl: "",
  maxPerFilePreRerank: 3,
  symbolFastPathEnabled: true,
  // Per-tier caps for the symbol fast path — symbolExactLimit applies
  // to tiers 1+2 (exact and case-insensitive matches on symbol_name),
  // symbolFuzzyLimit applies to tier 3 (ILIKE substring match). The
  // overall search-result limit still bounds the combined output.
  // Defaults match lib/config.ts SYMBOL_EXACT_LIMIT / SYMBOL_FUZZY_LIMIT
  // so the fast path behaves the same whether config comes from env
  // or from this DEFAULT_CONFIG fallback.
  symbolExactLimit: 20,
  symbolFuzzyLimit: 10,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Query Preprocessing
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Expand identifier tokens for improved BMX+ recall.
 *
 * Splits camelCase, PascalCase, snake_case, and ALLCAPS identifiers
 * into their component words so lexical search can match partial names.
 *
 * Examples:
 *   "getUserById"  → "getUserById get user by id"
 *   "rate_limiter" → "rate_limiter rate limiter"
 *   "XMLParser"    → "XMLParser xml parser"
 */
export function expandIdentifierTokens(query: string): string {
  const tokens = query.split(/\s+/).filter(Boolean);
  const expanded: string[] = [];

  for (const token of tokens) {
    expanded.push(token);

    if (token.includes("_")) {
      const parts = token.split("_").filter(Boolean);
      if (parts.length > 1) {
        for (const p of parts) expanded.push(p.toLowerCase());
      }
    } else if (HAS_CAMEL_RE.test(token)) {
      const split = token.replace(CAMEL_SPLIT_RE, "$1 $2").split(/\s+/);
      if (split.length > 1) {
        for (const p of split) expanded.push(p.toLowerCase());
      }
    } else if (HAS_ALLCAPS_TRANSITION_RE.test(token)) {
      const split = token.replace(ALLCAPS_SPLIT_RE, "$1 $2").split(/\s+/);
      if (split.length > 1) {
        for (const p of split) expanded.push(p.toLowerCase());
      }
    }
  }

  return expanded.join(" ");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Score Normalization
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Min-max normalize a set of scores to [0, 1].
 * Handles degenerate cases (single value, zero spread).
 *
 * @internal Exported for unit testing. Not part of the public API.
 */
export function normalizeScores(scores: Map<string, number>): Map<string, number> {
  if (scores.size === 0) return new Map();
  if (scores.size === 1) return new Map([...scores.entries()].map(([k]) => [k, 1.0]));

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of scores.values()) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }

  const spread = hi - lo;
  if (spread < 1e-9) {
    return new Map([...scores.keys()].map((k) => [k, 1.0]));
  }

  const invSpread = 1.0 / spread;
  return new Map([...scores.entries()].map(([k, v]) => [k, (v - lo) * invSpread]));
}

// ═══════════════════════════════════════════════════════════════════════════════
// BMX+ Index Cache
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * In-memory cache for per-project BMX+ indexes.
 *
 * Keyed by (storeId, projectId, repoRoot). Entries auto-expire after TTL.
 * Invalidation triggered on writes (upload/delete).
 */
class BM25Cache {
  private entries = new Map<string, BM25CacheEntry>();

  private cacheKey(storeId: string, projectId: string | null, repoRoot: string | null): string {
    return `${storeId}::${projectId ?? ""}::${repoRoot ?? ""}`;
  }

  get(
    storeId: string,
    projectId: string | null,
    repoRoot: string | null
  ): BM25CacheEntry | null {
    const key = this.cacheKey(storeId, projectId, repoRoot);
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (Date.now() - entry.builtAt > BM25_CACHE_TTL_MS) {
      this.entries.delete(key);
      return null;
    }
    return entry;
  }

  set(
    storeId: string,
    projectId: string | null,
    repoRoot: string | null,
    index: BMXPlusIndex,
    rowsById: Map<string, RawEmbeddingRow>
  ): void {
    const key = this.cacheKey(storeId, projectId, repoRoot);
    this.entries.set(key, {
      index,
      rowsById,
      builtAt: Date.now(),
      projectId,
      repoRoot,
    });
  }

  /**
   * Invalidate cache entries.
   *
   *   - invalidate(storeId) — wipes every entry for that store
   *   - invalidate(storeId, projectId) — wipes entries for that
   *     store AND project (any repoRoot)
   *   - invalidate(storeId, projectId, repoRoot) — wipes the single
   *     (storeId, projectId, repoRoot) entry
   *
   * Targeted invalidation is required so write paths that touch one
   * project don't blow away unrelated caches in a multi-project
   * deployment.
   */
  invalidate(storeId: string, projectId?: string | null, repoRoot?: string | null): void {
    if (projectId === undefined && repoRoot === undefined) {
      // Wipe whole store
      const prefix = `${storeId}::`;
      for (const key of this.entries.keys()) {
        if (key.startsWith(prefix)) {
          this.entries.delete(key);
        }
      }
      return;
    }

    if (projectId !== undefined && repoRoot === undefined) {
      // Wipe (storeId, projectId, *) entries
      const prefix = `${storeId}::${projectId ?? ""}::`;
      for (const key of this.entries.keys()) {
        if (key.startsWith(prefix)) {
          this.entries.delete(key);
        }
      }
      return;
    }

    // Targeted: storeId + projectId + repoRoot
    const exact = this.cacheKey(storeId, projectId ?? null, repoRoot ?? null);
    this.entries.delete(exact);
  }

  clear(): void {
    this.entries.clear();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Database Interface
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Abstract database interface.
 *
 * The search engine only uses query(); the wider VectorStoreDB type
 * lives in db-adapter.ts (where it also exposes transaction() and
 * close()). Re-exporting the canonical type here keeps a single source
 * of truth — passing a PgVectorDB to VectorStoreSearchEngine and to
 * the schema helpers is structurally identical.
 */
export type { VectorStoreDB } from "./db-adapter.js";
import type { VectorStoreDB } from "./db-adapter.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Embedding Service Interface
// ═══════════════════════════════════════════════════════════════════════════════

export interface EmbeddingService {
  /** Generate a single embedding vector for a query string */
  generateQueryEmbedding(text: string): Promise<number[]>;

  /** Generate embeddings for a batch of texts */
  generateBatch(texts: string[]): Promise<number[][]>;
}

/** Optional configuration for the LiteLLM embedding service. */
export interface EmbeddingServiceConfig {
  /** Request timeout in ms (default 120 000). */
  timeout?: number;
  /** Number of retry attempts (default 3). */
  retryAttempts?: number;
  /** Per-attempt backoff delays in ms (default [1000, 3000, 8000]). */
  retryDelays?: number[];
  /** Expected embedding dimension — vectors that don't match are rejected. */
  expectedDim?: number;
}

/**
 * Default embedding service implementation using LiteLLM-compatible API.
 *
 * Matches the Python EmbeddingService retry / validation behaviour:
 *   - Retries on network errors, HTTP 429, and HTTP 503
 *   - Validates embedding dimensions when expectedDim is set
 */
export class LiteLLMEmbeddingService implements EmbeddingService {
  private timeout: number;
  private retryAttempts: number;
  private retryDelays: number[];
  private expectedDim: number | undefined;

  constructor(
    private baseUrl: string,
    private apiKey: string,
    private model: string,
    config?: EmbeddingServiceConfig
  ) {
    this.timeout = config?.timeout ?? 120_000;
    this.retryAttempts = config?.retryAttempts ?? 3;
    this.retryDelays = config?.retryDelays ?? [1000, 3000, 8000];
    this.expectedDim = config?.expectedDim;
  }

  async generateQueryEmbedding(text: string): Promise<number[]> {
    const [embedding] = await this.generateBatch([text]);
    return embedding;
  }

  async generateBatch(texts: string[]): Promise<number[][]> {
    let lastErr: unknown;

    for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
      try {
        const resp = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/embeddings`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({ model: this.model, input: texts }),
          signal: AbortSignal.timeout(this.timeout),
        });

        if (!resp.ok) {
          const status = resp.status;
          // Retry on 429 (rate-limit) and 503 (service unavailable),
          // matching the Python baseline behaviour.
          if ((status === 429 || status === 503) && attempt < this.retryAttempts - 1) {
            const delay = this.retryDelays[Math.min(attempt, this.retryDelays.length - 1)] ?? 1000;
            await new Promise((r) => setTimeout(r, delay));
            continue;
          }
          const detail = await resp.text().catch(() => "");
          throw new Error(`Embedding API error ${status}: ${detail.slice(0, 300)}`);
        }

        const data = await resp.json();
        const sorted = (data.data as { index: number; embedding: number[] }[]).sort(
          (a, b) => a.index - b.index
        );
        const embeddings = sorted.map((item) => item.embedding);

        // Dimension validation — mirrors Python's per-vector check.
        if (this.expectedDim !== undefined) {
          for (let i = 0; i < embeddings.length; i++) {
            if (embeddings[i].length !== this.expectedDim) {
              throw new Error(
                `Expected embedding dimension ${this.expectedDim} for text ${i}, got ${embeddings[i].length}`
              );
            }
          }
        }

        return embeddings;
      } catch (err) {
        lastErr = err;
        // Retry on network-level errors (TypeError from fetch, AbortError
        // from timeout, and any generic network failures). Non-retryable
        // HTTP errors (4xx other than 429) already threw above.
        const isNetworkError =
          err instanceof TypeError ||
          (err instanceof DOMException && err.name === "AbortError") ||
          (err instanceof Error && err.name === "TimeoutError");

        if (isNetworkError && attempt < this.retryAttempts - 1) {
          const delay = this.retryDelays[Math.min(attempt, this.retryDelays.length - 1)] ?? 1000;
          await new Promise((r) => setTimeout(r, delay));
          continue;
        }

        // Non-retryable or final attempt — propagate.
        throw err;
      }
    }

    // Should not be reachable, but satisfies the type system.
    throw lastErr ?? new Error(`Embedding failed after ${this.retryAttempts} attempts`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Reranker
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Cross-encoder reranking via Cohere/Voyage/Jina-compatible API.
 *
 * Returns reranked candidates with relevance scores, or null on failure
 * (callers fall back to pre-rerank ordering).
 */
async function rerankCandidates(
  query: string,
  candidates: SearchCandidate[],
  config: VectorStoreConfig,
  topN: number
): Promise<SearchCandidate[] | null> {
  if (!config.rerankModel || candidates.length === 0) return null;

  const documents = candidates.map((c) => {
    const lc = c.lexicalContent?.trim();
    return lc || c.content;
  });

  const retryDelays = config.rerankRetryDelays
    .split(",")
    .map((s) => Number(s.trim()) * 1000);

  for (let attempt = 0; attempt < config.rerankMaxAttempts; attempt++) {
    try {
      const resp = await fetch(`${config.embeddingBase.replace(/\/$/, "")}/rerank`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.embeddingKey}`,
        },
        body: JSON.stringify({
          model: config.rerankModel,
          query,
          documents,
          top_n: topN,
        }),
        signal: AbortSignal.timeout(config.rerankTimeout),
      });

      if (!resp.ok) {
        throw new Error(`Rerank HTTP ${resp.status}: ${await resp.text().catch(() => "")}`);
      }

      const data = await resp.json();
      const results: { index: number; relevance_score: number }[] = data.results || [];

      return results.map((item) => ({
        ...candidates[item.index],
        rerankScore: item.relevance_score,
        score: item.relevance_score,
      }));
    } catch (err) {
      if (attempt < config.rerankMaxAttempts - 1) {
        const delay = retryDelays[Math.min(attempt, retryDelays.length - 1)] || 1000;
        await new Promise((r) => setTimeout(r, delay));
      } else {
        console.warn(
          `[vector-store] Rerank failed after ${config.rerankMaxAttempts} attempts:`,
          err instanceof Error ? err.message : err
        );
        return null;
      }
    }
  }

  return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Diversity Controls
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Cap results per source file to ensure diversity in the reranker input pool.
 * Prevents a single large file from dominating the candidate set.
 */
function capPerFile(candidates: SearchCandidate[], maxPerFile: number): SearchCandidate[] {
  const fileCounts = new Map<string, number>();
  const capped: SearchCandidate[] = [];

  for (const c of candidates) {
    const src = c.metadata.doc_path || c.metadata.source || c.metadata.filename || "__unknown__";
    const count = fileCounts.get(src) || 0;
    if (count < maxPerFile) {
      capped.push(c);
      fileCounts.set(src, count + 1);
    }
  }

  return capped;
}

/**
 * Post-rerank deduplication: keep only the highest-scored result per source file.
 * Applied after reranking when we want one authoritative chunk per file.
 */
function deduplicateBySource(candidates: SearchCandidate[]): SearchCandidate[] {
  const seen = new Set<string>();
  const deduped: SearchCandidate[] = [];

  for (const c of candidates) {
    const src = c.metadata.doc_path || c.metadata.source || c.metadata.filename || "__unknown__";
    if (!seen.has(src)) {
      seen.add(src);
      deduped.push(c);
    }
  }

  return deduped;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convex Combination Fusion
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fuse semantic and BMX+ results via normalized convex combination.
 *
 * Both score sets are independently min-max normalized to [0,1], then
 * combined as: final = αS·norm_semantic + αB·norm_bmx
 *
 * Documents appearing in only one source receive 0 for the missing signal,
 * which naturally deprioritizes single-signal matches.
 */
function fuseConvexCombination(
  semanticCandidates: SearchCandidate[],
  bmxResults: [string, number][],
  bmxRowsById: Map<string, RawEmbeddingRow>,
  semanticWeight: number,
  bmxWeight: number
): SearchCandidate[] {
  // Build lookup maps
  const semById = new Map<string, SearchCandidate>();
  const rawSemScores = new Map<string, number>();
  for (const c of semanticCandidates) {
    semById.set(c.chunkId, c);
    rawSemScores.set(c.chunkId, c.score);
  }

  const rawBmxScores = new Map<string, number>();
  for (const [cid, score] of bmxResults) {
    rawBmxScores.set(cid, score);
  }

  // Normalize independently
  const normSem = normalizeScores(rawSemScores);
  const normBmx = normalizeScores(rawBmxScores);

  // Union all candidate IDs
  const allIds = new Set<string>([...rawSemScores.keys(), ...rawBmxScores.keys()]);
  const fused: SearchCandidate[] = [];

  for (const cid of allIds) {
    const ns = normSem.get(cid) ?? 0.0;
    const nb = normBmx.get(cid) ?? 0.0;
    const combined = semanticWeight * ns + bmxWeight * nb;

    // Resolve the candidate data
    const existing = semById.get(cid);
    if (existing) {
      fused.push({
        ...existing,
        score: combined,
        semanticScore: rawSemScores.get(cid),
        bmxScore: rawBmxScores.get(cid),
      });
    } else {
      // BMX-only hit — needs row data
      const row = bmxRowsById.get(cid);
      if (!row) continue;
      fused.push({
        chunkId: cid,
        content: row.content,
        semanticContent: row.semantic_content || row.content,
        lexicalContent: row.lexical_content || row.content,
        metadata: row.metadata || {},
        score: combined,
        semanticScore: undefined,
        bmxScore: rawBmxScores.get(cid),
      });
    }
  }

  // Sort descending
  fused.sort((a, b) => b.score - a.score);
  return fused;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metadata Filtering
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Check if a chunk's metadata satisfies all filter conditions.
 * Used for post-hoc filtering of BMX+ results (which are indexed in-memory
 * and may span the full corpus if project scope isn't baked into the index key).
 */
function metadataMatchesFilters(
  metadata: ChunkMetadata | null | undefined,
  filters: SearchFilters | null | undefined
): boolean {
  if (!filters) return true;
  const meta = metadata || {};

  for (const [key, value] of Object.entries(filters)) {
    if (value === undefined) continue;
    if (String(meta[key] ?? "") !== String(value)) return false;
  }

  return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Code-Aware Decoration
// ═══════════════════════════════════════════════════════════════════════════════

function isCodeContent(metadata: ChunkMetadata | null): boolean {
  return metadata?.content_type === "code";
}

function prependCodeWarning(text: string, metadata: ChunkMetadata | null, cfg: VectorStoreConfig): string {
  if (!cfg.codeAwareEnabled || !isCodeContent(metadata)) return text;
  const warning = cfg.codeResultWarning.trim();
  if (!warning) return text;
  if (text.startsWith(warning)) return text;
  return `${warning}\n${text}`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VectorStore Search Engine
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * The hybrid retrieval engine.
 *
 * Stateful: maintains a BMX+ index cache per (store, project) pair.
 * Thread-safe for concurrent searches (cache is read-heavy, writes are atomic).
 *
 * Usage:
 *   const engine = new VectorStoreSearchEngine(config, db, embeddingService);
 *   const results = await engine.search({ query: "rate limiter", filters: { project_id: "my-proj" } });
 */
export class VectorStoreSearchEngine {
  private config: VectorStoreConfig;
  private db: VectorStoreDB;
  private embeddings: EmbeddingService;
  private bm25Cache: BM25Cache;
  /** In-flight BMX+ index builds — prevents thundering-herd when multiple
   *  concurrent searches hit the same (projectId, repoRoot) pair. */
  private bmxBuildInFlight = new Map<string, Promise<{ index: BMXPlusIndex; rowsById: Map<string, RawEmbeddingRow> }>>();

  constructor(config: VectorStoreConfig, db: VectorStoreDB, embeddings: EmbeddingService) {
    this.config = config;
    this.db = db;
    this.embeddings = embeddings;
    this.bm25Cache = new BM25Cache();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Execute a hybrid search query.
   *
   * Full pipeline: semantic → BMX+ → fuse → cap → rerank → dedup → format.
   */
  async search(request: SearchRequest): Promise<SearchResponse> {
    const {
      query,
      filters,
      limit: requestLimit,
      lexicalOnly = false,
      returnMetadata = true,
      returnRawContent = false,
    } = request;

    const cfg = this.config;
    const finalLimit = Math.min(requestLimit || cfg.defaultLimit, cfg.maxLimit);

    const projectId = filters?.project_id ?? null;
    const repoRoot = filters?.repo_root ?? null;

    // ── Symbol fast path — skip embedding if exact symbol hits satisfy limit ──
    // When the symbol search produces *partial* results (>0 but < finalLimit)
    // we keep them in `preMergeSymbolCandidates` and merge into the final
    // result list AFTER the hybrid (or lexical-only) pipeline runs. They
    // are pre-pended at the top because they are exact-match symbol hits
    // (score >= 0.8) and should outrank fuzzy semantic/lexical results.
    let preMergeSymbolCandidates: SearchCandidate[] = [];
    if (this.config.symbolFastPathEnabled && this.isSymbolQuery(query)) {
      const symbolCandidates = await this.symbolSearch(query, filters ?? null, finalLimit);
      if (symbolCandidates.length >= finalLimit) {
        const sliced = symbolCandidates.slice(0, finalLimit);
        const results: SearchResultItem[] = sliced.map((c) => {
          let text = c.content;
          text = prependCodeWarning(text, c.metadata, cfg);
          return {
            id: c.chunkId,
            score: Math.round(c.score * 10000) / 10000,
            source: c.metadata.doc_path || c.metadata.source || "unknown",
            content: text,
            rawContent: returnRawContent ? c.content : null,
            metadata: returnMetadata ? c.metadata : null,
          };
        });
        return {
          query,
          results,
          resultCount: results.length,
          retrievalMode: "symbol",
          projectId: projectId ?? undefined,
        };
      }
      // Partial — retain them for merging post-hybrid (or post-lexical).
      preMergeSymbolCandidates = symbolCandidates;
    }

    // ── Lexical-only shortcut ──────────────────────────────────────────────
    if (lexicalOnly) {
      const lexResponse = await this.lexicalSearch(
        query,
        filters ?? null,
        finalLimit,
        returnMetadata,
        returnRawContent
      );
      if (preMergeSymbolCandidates.length === 0) {
        return lexResponse;
      }
      // Merge: symbol hits (already-rendered SearchResultItems) first, then
      // lexical results, deduped by chunk id, truncated to finalLimit.
      const symbolItems: SearchResultItem[] = preMergeSymbolCandidates.map((c) => {
        let text = c.content;
        text = prependCodeWarning(text, c.metadata, cfg);
        return {
          id: c.chunkId,
          score: Math.round(c.score * 10000) / 10000,
          source: c.metadata.doc_path || c.metadata.source || "unknown",
          content: text,
          rawContent: returnRawContent ? c.content : null,
          metadata: returnMetadata ? c.metadata : null,
        };
      });
      const seen = new Set<string>(symbolItems.map((it) => it.id));
      const merged: SearchResultItem[] = [...symbolItems];
      for (const it of lexResponse.results) {
        if (!seen.has(it.id)) {
          merged.push(it);
          seen.add(it.id);
        }
      }
      const truncated = merged.slice(0, finalLimit);
      return {
        query,
        results: truncated,
        resultCount: truncated.length,
        retrievalMode: "symbol+lexical",
        projectId: projectId ?? undefined,
      };
    }

    // ── Determine fetch pool size ──────────────────────────────────────────
    const rerankEnabled = Boolean(cfg.rerankModel);
    const hybridEnabled = cfg.hybridEnabled;
    let fetchLimit = finalLimit;
    if (rerankEnabled) fetchLimit = Math.max(fetchLimit, cfg.rerankCandidates);
    if (hybridEnabled) fetchLimit = Math.max(fetchLimit, cfg.bm25Candidates);

    // ── Parallel retrieval: embedding + BMX+ ───────────────────────────────
    const [semanticCandidates, bmxResult] = await Promise.all([
      this.semanticSearch(query, filters ?? null, fetchLimit),
      hybridEnabled
        ? this.getBmxResults(query, projectId, repoRoot, fetchLimit).catch((e) => {
            console.warn(`[vector-store] BMX+ retrieval failed: ${e}`);
            return null;
          })
        : Promise.resolve(null),
    ]);

    // ── Fuse results ───────────────────────────────────────────────────────
    let candidates: SearchCandidate[];

    if (bmxResult && bmxResult.hits.length > 0) {
      // Filter BMX results by metadata if needed
      const filteredBmx = filters && Object.keys(filters).length > 0
        ? bmxResult.hits.filter(([cid]) => {
            const row = bmxResult.rowsById.get(cid);
            return metadataMatchesFilters(row?.metadata, filters);
          })
        : bmxResult.hits;

      if (filteredBmx.length > 0) {
        candidates = fuseConvexCombination(
          semanticCandidates,
          filteredBmx,
          bmxResult.rowsById,
          cfg.semanticWeight,
          cfg.bm25Weight
        );
      } else {
        candidates = semanticCandidates;
      }
    } else {
      candidates = semanticCandidates;
    }

    // ── Merge partial symbol fast-path hits into candidate pool ──────────
    // When the symbol fast path produced partial (>0 but < finalLimit)
    // results, inject them into the candidate pool BEFORE the per-file
    // cap and reranking steps so they participate in reranking rather
    // than being blindly prepended afterward. This matches the Python
    // baseline's behaviour. Dedup by chunkId so a chunk that also
    // surfaces in hybrid doesn't appear twice.
    let mergedRetrievalMode: SearchResponse["retrievalMode"] =
      hybridEnabled && bmxResult ? "hybrid" : "semantic";
    if (preMergeSymbolCandidates.length > 0) {
      const seen = new Set<string>(preMergeSymbolCandidates.map((c) => c.chunkId));
      const merged: SearchCandidate[] = [...preMergeSymbolCandidates];
      for (const c of candidates) {
        if (!seen.has(c.chunkId)) {
          merged.push(c);
          seen.add(c.chunkId);
        }
      }
      candidates = merged;
      mergedRetrievalMode = "symbol+hybrid";
    }

    // ── Per-file diversity cap ─────────────────────────────────────────────
    candidates = capPerFile(candidates, cfg.maxPerFilePreRerank);

    // ── Reranking ──────────────────────────────────────────────────────────
    if (rerankEnabled && candidates.length > 0) {
      const pool = candidates.slice(0, cfg.rerankCandidates);
      const topN = Math.min(cfg.rerankTopN, pool.length);
      const reranked = await rerankCandidates(query, pool, cfg, topN);
      candidates = reranked ?? candidates.slice(0, finalLimit);
    } else {
      candidates = candidates.slice(0, finalLimit);
    }

    // ── Post-rerank deduplication ──────────────────────────────────────────
    candidates = deduplicateBySource(candidates);
    candidates = candidates.slice(0, finalLimit);

    // ── Format response ────────────────────────────────────────────────────
    const results: SearchResultItem[] = candidates.map((c) => {
      let text = c.content;
      text = prependCodeWarning(text, c.metadata, cfg);

      return {
        id: c.chunkId,
        score: Math.round((c.rerankScore ?? c.score) * 10000) / 10000,
        source: c.metadata.doc_path || c.metadata.source || "unknown",
        content: text,
        rawContent: returnRawContent ? c.content : null,
        metadata: returnMetadata ? c.metadata : null,
      };
    });

    return {
      query,
      results,
      resultCount: results.length,
      retrievalMode: mergedRetrievalMode,
      projectId: projectId ?? undefined,
    };
  }

  /**
   * Invalidate the BMX+ index cache for a store (call after writes).
   */
  invalidateCache(storeId?: string): void {
    if (storeId) {
      this.bm25Cache.invalidate(storeId);
    } else {
      this.bm25Cache.clear();
    }
  }

  /**
   * Invalidate BMX+ cache entries.
   *
   * Granularity is controlled by which arguments you pass:
   *   - storeId only            → wipes every entry for the store
   *   - storeId + projectId     → wipes every entry for that project (any repoRoot)
   *   - storeId + projectId + repoRoot → wipes the single matching entry
   *
   * Always prefer the most-specific call you can make so writes to
   * one project don't blow away cached indexes for unrelated
   * projects in the same store.
   */
  public invalidateBmxCache(storeId: string, projectId?: string, repoRoot?: string): void {
    if (projectId === undefined && repoRoot === undefined) {
      this.bm25Cache.invalidate(storeId);
      return;
    }
    if (projectId !== undefined && repoRoot === undefined) {
      this.bm25Cache.invalidate(storeId, projectId);
      return;
    }
    this.bm25Cache.invalidate(storeId, projectId ?? null, repoRoot ?? null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: Symbol Fast Path
  // ─────────────────────────────────────────────────────────────────────────

  private isSymbolQuery(query: string): boolean {
    return SYMBOL_QUERY_RE.test(query.trim());
  }

  private rowToCandidate(row: RawEmbeddingRow, baseScore: number): SearchCandidate {
    return {
      chunkId: row.id,
      content: row.content,
      semanticContent: row.semantic_content || row.content,
      lexicalContent: row.lexical_content || row.content,
      metadata: row.metadata || {},
      score: baseScore,
    };
  }

  private async symbolSearch(
    query: string,
    filters: SearchFilters | null,
    limit: number
  ): Promise<SearchCandidate[]> {
    const projectId = filters?.project_id;
    if (!projectId) return [];

    const candidates: SearchCandidate[] = [];

    // Per-tier caps come from config:
    //   * symbolExactLimit gates tiers 1 and 2 (the exact-match family)
    //   * symbolFuzzyLimit gates tier 3 (ILIKE substring match)
    // The outer `limit` (caller-supplied finalLimit) is always also
    // applied so the combined result never exceeds the user's request.
    // We use Math.max(0, ...) to never pass a negative LIMIT to Postgres
    // when prior tiers have already saturated the outer limit.
    const exactCap = Math.max(0, this.config.symbolExactLimit);
    const fuzzyCap = Math.max(0, this.config.symbolFuzzyLimit);

    // Tier 1: Exact symbol match.
    // SQL LIMIT = min(remaining outer slots, exact-tier cap).
    const tier1Limit = Math.max(0, Math.min(limit - candidates.length, exactCap));
    if (tier1Limit > 0) {
      const exactRows = await this.db.query<RawEmbeddingRow>(
        `SELECT id, content, semantic_content, lexical_content, metadata
         FROM embeddings
         WHERE metadata->>'project_id' = $1 AND metadata->>'symbol_name' = $2
         LIMIT $3`,
        [projectId, query, tier1Limit]
      );

      for (const row of exactRows) {
        candidates.push(this.rowToCandidate(row, 1.0));
      }
    }

    if (candidates.length >= limit) {
      return candidates.slice(0, limit);
    }

    // Tier 2: Case-insensitive exact (excluding rows already in tier 1).
    // The `symbol_name != $2` clause filters out exact-case rows, but a
    // row with a *different* symbol_name on the same chunk (e.g. an
    // unrelated alias on the same embedding row) could still hit tier 1
    // and tier 2 with different metadata. Dedupe defensively by
    // chunkId so a single embedding never appears twice.
    //
    // SQL LIMIT = min(remaining outer slots, remaining exact-tier cap).
    // The exact-tier cap is the COMBINED cap for tiers 1+2, so subtract
    // however many candidates we've already accepted from tiers 1 (all
    // of which were exact-tier hits).
    const tier2Limit = Math.max(
      0,
      Math.min(limit - candidates.length, exactCap - candidates.length)
    );
    if (tier2Limit > 0) {
      const ciRows = await this.db.query<RawEmbeddingRow>(
        `SELECT id, content, semantic_content, lexical_content, metadata
         FROM embeddings
         WHERE metadata->>'project_id' = $1
           AND LOWER(metadata->>'symbol_name') = LOWER($2)
           AND metadata->>'symbol_name' != $2
         LIMIT $3`,
        [projectId, query, tier2Limit]
      );

      for (const row of ciRows) {
        if (!candidates.some(c => c.chunkId === row.id)) {
          candidates.push(this.rowToCandidate(row, 0.95));
        }
      }
    }

    if (candidates.length >= limit) {
      return candidates.slice(0, limit);
    }

    // Tier 3: Fuzzy ILIKE match on symbol_name.
    // SQL LIMIT = min(remaining outer slots, fuzzy-tier cap).
    // fuzzyCap is independent from exactCap — fuzzy results have their
    // own budget regardless of how many exact-tier results we found.
    const tier3Limit = Math.max(0, Math.min(limit - candidates.length, fuzzyCap));
    if (tier3Limit > 0) {
      const fuzzyRows = await this.db.query<RawEmbeddingRow>(
        `SELECT id, content, semantic_content, lexical_content, metadata
         FROM embeddings
         WHERE metadata->>'project_id' = $1
           AND metadata->>'symbol_name' ILIKE $2
         LIMIT $3`,
        [projectId, `%${query}%`, tier3Limit]
      );

      for (const row of fuzzyRows) {
        if (!candidates.some(c => c.chunkId === row.id)) {
          candidates.push(this.rowToCandidate(row, 0.8));
        }
      }
    }

    return candidates;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: Semantic (Vector) Search
  // ─────────────────────────────────────────────────────────────────────────

  private async semanticSearch(
    query: string,
    filters: SearchFilters | null,
    limit: number
  ): Promise<SearchCandidate[]> {
    const cfg = this.config;
    const queryEmbedding = await this.embeddings.generateQueryEmbedding(query);
    const vecStr = `[${queryEmbedding.join(",")}]`;

    // Build parameterized query
    const params: any[] = [cfg.storeId, vecStr];
    let paramIdx = 3;

    let filterSql = "";
    if (filters) {
      const filterClauses: string[] = [];
      for (const [key, value] of Object.entries(filters)) {
        if (value === undefined) continue;
        // Defense-in-depth: validate every filter key against the canonical
        // whitelist before interpolating it into raw SQL. Reject the whole
        // query on any unknown key — silently dropping would (a) let an
        // attacker enumerate the whitelist by observing which keys cause
        // errors vs. quiet drops, and (b) mask legitimate caller bugs from
        // mistyped keys.
        if (!ALLOWED_FILTER_KEYS.has(key)) {
          throw new Error(
            `Invalid filter key '${key}'. Allowed: ${Array.from(ALLOWED_FILTER_KEYS).join(", ")}`
          );
        }
        filterClauses.push(`metadata->>'${key}' = $${paramIdx}`);
        params.push(value);
        paramIdx++;
      }
      if (filterClauses.length > 0) {
        filterSql = "AND " + filterClauses.join(" AND ");
      }
    }

    // Parameterize LIMIT for best-practice defense-in-depth even though
    // `limit` here is bounded by config (cfg.maxLimit), not raw user input.
    const limitParamIdx = paramIdx;
    params.push(limit);
    paramIdx++;

    const sql = `
      SELECT id, content, semantic_content, lexical_content, metadata,
             (embedding ${cfg.distanceOperator} $2::vector) AS distance
      FROM embeddings
      WHERE vector_store_id = $1 ${filterSql}
      ORDER BY distance ASC
      LIMIT $${limitParamIdx}
    `;

    const rows = await this.db.query<RawEmbeddingRow & { distance: number }>(sql, params);

    const candidates: SearchCandidate[] = [];
    for (const row of rows) {
      // Convert distance to similarity score (cosine: score = 1 - distance/2)
      const score = Math.max(0, 1 - row.distance / 2);
      if (score < cfg.minScore) continue;

      candidates.push({
        chunkId: row.id,
        content: row.content,
        semanticContent: row.semantic_content || row.content,
        lexicalContent: row.lexical_content || row.content,
        metadata: row.metadata || {},
        score,
        semanticScore: score,
      });
    }

    return candidates;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: Lexical-Only Search
  // ─────────────────────────────────────────────────────────────────────────

  private async lexicalSearch(
    query: string,
    filters: SearchFilters | null,
    limit: number,
    returnMetadata: boolean,
    returnRawContent: boolean
  ): Promise<SearchResponse> {
    const cfg = this.config;
    const projectId = filters?.project_id ?? null;
    const repoRoot = filters?.repo_root ?? null;

    const { index, rowsById } = await this.getOrBuildBmxIndex(projectId, repoRoot);

    const bmxQuery = expandIdentifierTokens(query);
    const hits = index.search(bmxQuery, limit);

    if (hits.length === 0) {
      return { query, results: [], resultCount: 0, retrievalMode: "lexical", projectId: projectId ?? undefined };
    }

    // Fetch full rows for hits not in cache
    const missingIds = hits.filter(([cid]) => !rowsById.has(cid)).map(([cid]) => cid);
    if (missingIds.length > 0) {
      const fetched = await this.fetchRowsByIds(missingIds);
      for (const row of fetched) rowsById.set(row.id, row);
    }

    const results: SearchResultItem[] = [];
    for (const [cid, score] of hits) {
      const row = rowsById.get(cid);
      if (!row) continue;

      // Post-hoc metadata filter
      if (!metadataMatchesFilters(row.metadata, filters)) continue;

      let text = row.content;
      text = prependCodeWarning(text, row.metadata, cfg);

      results.push({
        id: cid,
        score: Math.round(score * 10000) / 10000,
        source: row.metadata?.doc_path || row.metadata?.source || "unknown",
        content: text,
        rawContent: returnRawContent ? row.content : null,
        metadata: returnMetadata ? (row.metadata || {}) : null,
      });
    }

    return {
      query,
      results: results.slice(0, limit),
      resultCount: Math.min(results.length, limit),
      retrievalMode: "lexical",
      projectId: projectId ?? undefined,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: BMX+ Index Management
  // ─────────────────────────────────────────────────────────────────────────

  private async getBmxResults(
    query: string,
    projectId: string | null,
    repoRoot: string | null,
    limit: number
  ): Promise<{ hits: [string, number][]; rowsById: Map<string, RawEmbeddingRow> }> {
    const { index, rowsById } = await this.getOrBuildBmxIndex(projectId, repoRoot);

    const bmxQuery = expandIdentifierTokens(query);
    const hits = index.search(bmxQuery, limit);

    return { hits, rowsById };
  }

  private async getOrBuildBmxIndex(
    projectId: string | null,
    repoRoot: string | null
  ): Promise<{ index: BMXPlusIndex; rowsById: Map<string, RawEmbeddingRow> }> {
    const cfg = this.config;

    // Check cache
    const cached = this.bm25Cache.get(cfg.storeId, projectId, repoRoot);
    if (cached) {
      return { index: cached.index, rowsById: cached.rowsById };
    }

    // Thundering-herd prevention: if another caller is already building
    // the index for the same (projectId, repoRoot) pair, await that
    // promise instead of kicking off a parallel build.
    const inflightKey = `${cfg.storeId}::${projectId ?? ""}::${repoRoot ?? ""}`;
    const inflight = this.bmxBuildInFlight.get(inflightKey);
    if (inflight) {
      return inflight;
    }

    // No build in progress — start one and register the promise so
    // concurrent callers coalesce onto it.
    const buildPromise = this.buildBmxIndex(projectId, repoRoot);
    this.bmxBuildInFlight.set(inflightKey, buildPromise);

    try {
      const result = await buildPromise;
      return result;
    } finally {
      this.bmxBuildInFlight.delete(inflightKey);
    }
  }

  /** Actually build the BMX+ index — called only once per cache miss
   *  thanks to the in-flight coalescing in getOrBuildBmxIndex. */
  private async buildBmxIndex(
    projectId: string | null,
    repoRoot: string | null
  ): Promise<{ index: BMXPlusIndex; rowsById: Map<string, RawEmbeddingRow> }> {
    const cfg = this.config;

    // Build fresh index by paginating through the corpus
    const rowsById = new Map<string, RawEmbeddingRow>();
    const chunks: { chunk_id: string; text: string }[] = [];

    let offset = 0;
    while (true) {
      const params: any[] = [cfg.storeId];
      let paramIdx = 2;
      let filterSql = "";

      if (repoRoot) {
        filterSql += ` AND metadata->>'repo_root' = $${paramIdx}`;
        params.push(repoRoot);
        paramIdx++;
      }
      if (projectId) {
        filterSql += ` AND metadata->>'project_id' = $${paramIdx}`;
        params.push(projectId);
        paramIdx++;
      }

      // Parameterize LIMIT/OFFSET. Both values come from numeric internal
      // config (cfg.bm25PageSize) and a loop-controlled offset, so this
      // is best-practice defense-in-depth, not a fix for an exploitable
      // vector. The repo_root and project_id keys above are hardcoded
      // literals (not user-controlled), so they don't need whitelist
      // validation.
      const limitParamIdx = paramIdx;
      params.push(cfg.bm25PageSize);
      paramIdx++;
      const offsetParamIdx = paramIdx;
      params.push(offset);
      paramIdx++;

      const sql = `
        SELECT id, content, semantic_content, lexical_content, metadata
        FROM embeddings
        WHERE vector_store_id = $1 ${filterSql}
        ORDER BY id
        LIMIT $${limitParamIdx} OFFSET $${offsetParamIdx}
      `;

      const rows = await this.db.query<RawEmbeddingRow>(sql, params);
      if (rows.length === 0) break;

      for (const row of rows) {
        rowsById.set(row.id, row);
        // Prefer lexical_content for BMX+ (includes context headers)
        const text = row.lexical_content?.trim() ? row.lexical_content : row.content;
        if (!text?.trim()) continue;
        chunks.push({ chunk_id: row.id, text });
      }

      if (rows.length < cfg.bm25PageSize) break;
      offset += cfg.bm25PageSize;
    }

    // Build the BMX+ index
    const index = new BMXPlusIndex(null, null, true);
    index.buildIndex(chunks);

    // Cache it
    this.bm25Cache.set(cfg.storeId, projectId, repoRoot, index, rowsById);

    return { index, rowsById };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: Row Fetching
  // ─────────────────────────────────────────────────────────────────────────

  private async fetchRowsByIds(ids: string[]): Promise<RawEmbeddingRow[]> {
    if (ids.length === 0) return [];

    // Build parameterized IN clause
    const placeholders = ids.map((_, i) => `$${i + 1}`).join(", ");
    const sql = `
      SELECT id, content, semantic_content, lexical_content, metadata
      FROM embeddings
      WHERE id IN (${placeholders})
    `;

    return this.db.query<RawEmbeddingRow>(sql, ids);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════════════
//
// NOTE: Configuration loading lives in lib/config.ts (loadConfig() →
// RagConfig, then toVectorStoreConfig(cfg) → VectorStoreConfig). The
// engine intentionally does NOT load its own env vars — there is one
// canonical config loader for the project so that env precedence,
// validation, and defaults stay coherent. Callers must build a
// VectorStoreConfig (typically via toVectorStoreConfig) and pass it
// in explicitly.

/**
 * Create a fully-configured search engine instance from an explicit
 * VectorStoreConfig.
 *
 * Usage:
 *   import { loadConfig, toVectorStoreConfig } from "./config.js";
 *   import { createSearchEngine } from "./vector-store.js";
 *   const cfg = toVectorStoreConfig(loadConfig());
 *   const engine = createSearchEngine(db, cfg);
 *   const results = await engine.search({ query, filters: { project_id } });
 */
export function createSearchEngine(
  db: VectorStoreDB,
  config: VectorStoreConfig,
  configOverrides?: Partial<VectorStoreConfig>,
  embeddingConfig?: EmbeddingServiceConfig
): VectorStoreSearchEngine {
  const cfg: VectorStoreConfig = { ...config, ...(configOverrides || {}) };
  const embeddings = new LiteLLMEmbeddingService(
    cfg.embeddingBase,
    cfg.embeddingKey,
    cfg.embeddingModel,
    embeddingConfig
  );
  return new VectorStoreSearchEngine(cfg, db, embeddings);
}

