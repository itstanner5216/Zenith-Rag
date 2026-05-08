// ---------------------------------------------------------------------------
// lib/index.ts — Library barrel for Zenith-Rag
//
// Public surface for consumers embedding Zenith-Rag as a library
// (e.g. the standalone HTTP server, a custom MCP host, or test code).
// ---------------------------------------------------------------------------

// DB
export { createVectorDB, PgVectorDB, type VectorStoreDB, type CreateVectorDBOptions } from "./db-adapter.js";

// Schema (DDL + helpers)
export {
  ensureSchema,
  insertEmbeddingBatch,
  deleteEmbeddingsForDocs,
  getExistingSourceHashes,
  buildMetadataFilter,
  ALLOWED_FILTER_KEYS,
  type EmbeddingInsertItem,
  type SourceHashEntry,
  type MetadataFilterResult,
} from "./schema.js";

// Config
export { loadConfig, toVectorStoreConfig, getConfig, type RagConfig } from "./config.js";

// Chunking
export { chunkFile, type ChunkResult, type ChunkMetadata, type ChunkOptions } from "./chunker.js";

// Vector store / search engine
// NOTE: VectorStoreDB is not re-exported here from vector-store.js because it
// is already exported above from db-adapter.js; a duplicate export would cause
// a compile-time conflict.
export {
  VectorStoreSearchEngine,
  LiteLLMEmbeddingService,
  createSearchEngine,
  expandIdentifierTokens,
  normalizeScores,
  SYMBOL_QUERY_RE,
  DEFAULT_CONFIG,
  type VectorStoreConfig,
  type SearchFilters,
  type SearchRequest,
  type SearchResponse,
  type SearchResultItem,
  type EmbeddingService,
} from "./vector-store.js";

// Server-side auth helpers (used by server/http.ts and any embedder)
export { getApiKey, requireApiKey, authenticateBearer, authMiddleware } from "./server-auth.js";

// Tree-sitter integration (re-export the public API; init is lazy/internal)
export {
  getDefinitions,
  getLangForFile,
  isSupported,
  getSupportedExtensions,
  getSymbols,
  getFileSymbols,
  treeSitterAvailable,
  type SymbolInfo,
  type SymbolFilterOptions,
} from "./tree-sitter.js";

// Repo / project resolution
export { findRepoRoot } from "./repo-root.js";
export { normalizePath, expandHome, convertToWindowsPath } from "./path-utils.js";
export { ProjectRegistry, type ProjectManifest } from "./project-registry.js";
