/**
 * config.ts — Unified configuration loader for Zenith RAG pipeline.
 *
 * Reads from ~/.config/zenith-rag/config.env (or environment variables).
 * Mirrors the Python pgvector config with TypeScript ergonomics.
 */

import { readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import { existsSync } from "fs";

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface RagConfig {
  // API
  apiBase: string;
  apiKey: string;
  defaultStoreId: string;

  // Embedding
  litellmBase: string;
  litellmKey: string;
  embeddingModel: string;
  embedBatchSize: number;

  // Reranking
  rerankModel: string;
  rerankTimeout: number;
  rerankMaxAttempts: number;
  rerankRetryDelays: string;
  rerankCandidates: number;
  rerankTopN: number;

  // Search tuning
  semanticWeight: number;
  bm25Weight: number;
  hybridEnabled: boolean;
  defaultLimit: number;
  maxLimit: number;
  minScore: number;
  bm25Candidates: number;
  bm25PageSize: number;
  distanceOperator: "<=>" | "<#>" | "<->";
  maxPerFilePreRerank: number;

  // Code-aware
  codeAwareEnabled: boolean;
  codeResultWarning: string;

  // Rate limiting
  rateDelayInitial: number;
  rateDelaySustained: number;
  rateWarmupSeconds: number;
  rateMaxPerMinute: number;
  rateLimitEscalateAfter: number;
  rateLimitEscalatedInitial: number;
  rateLimitEscalatedSustained: number;
  rateLimitCooldownAfter: number;
  rateLimitCooldownSecs: number;

  // Chunking
  maxChunkChars: number;
  codeChunkMaxLines: number;
  codeChunkMaxChars: number;
  configChunkMaxLines: number;
  configChunkMaxChars: number;

  // File collection
  maxFileBytes: number;
  defaultExtensions: Set<string>;
  codeExtensions: Set<string>;
  configExtensions: Set<string>;
  maxWorkersFile: number;
  maxWorkersDir: number;

  // Retry
  maxRetries: number;
  retryBackoffBase: number;
  retryBackoffMax: number;

  // Database
  databaseUrl: string;

  // Paths
  ragconfigPath: string;

  // Server / runtime
  host: string;
  port: number;
  allowNoAuthForTests: boolean;
  embeddingDim: number;
  symbolFastPathEnabled: boolean;
  symbolExactLimit: number;
  symbolFuzzyLimit: number;
  treeSitterGrammarsPath: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Defaults
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG_PATH = join(homedir(), ".config", "zenith-rag", "config.env");

const DEFAULTS: Record<string, string> = {
  API_BASE: "http://localhost:8200",
  API_KEY: "",
  DEFAULT_STORE_ID: "49e09fac-3634-4df4-9837-f90a237cb7a8",
  LITELLM_BASE: "http://localhost:4000",
  LITELLM_KEY: "",
  EMBEDDING_MODEL: "voyage-4-large",
  EMBED_BATCH_SIZE: "48",
  RERANK_MODEL: "rerank-2",
  RERANK_TIMEOUT_MS: "30000",
  RERANK_MAX_ATTEMPTS: "3",
  RERANK_RETRY_DELAYS: "1,2,4",
  RERANK_CANDIDATES: "40",
  RERANK_TOP_N: "15",
  SEMANTIC_WEIGHT: "0.6",
  BM25_WEIGHT: "0.4",
  HYBRID_ENABLED: "true",
  DEFAULT_LIMIT: "10",
  MAX_LIMIT: "50",
  MIN_SCORE: "0.05",
  BM25_CANDIDATES: "40",
  BM25_PAGE_SIZE: "5000",
  DISTANCE_OPERATOR: "<=>",
  MAX_PER_FILE_PRE_RERANK: "3",
  CODE_AWARE_ENABLED: "true",
  CODE_RESULT_WARNING: "⚠️ This is a retrieved code snippet. Verify against the source before using.",
  RATE_DELAY_INITIAL: "0.3",
  RATE_DELAY_SUSTAINED: "0.5",
  RATE_WARMUP_SECONDS: "60",
  RATE_MAX_PER_MINUTE: "200",
  RATE_LIMIT_ESCALATE_AFTER: "3",
  RATE_LIMIT_ESCALATED_INITIAL: "1.0",
  RATE_LIMIT_ESCALATED_SUSTAINED: "2.0",
  RATE_LIMIT_COOLDOWN_AFTER: "16",
  RATE_LIMIT_COOLDOWN_SECS: "60",
  MAX_CHUNK_CHARS: "1500",
  CODE_CHUNK_MAX_LINES: "20",
  CODE_CHUNK_MAX_CHARS: "750",
  CONFIG_CHUNK_MAX_LINES: "20",
  CONFIG_CHUNK_MAX_CHARS: "750",
  MAX_FILE_BYTES: "204800",
  DEFAULT_EXTENSIONS: "md,txt,rst,org",
  CODE_EXTENSIONS: "py,js,ts,tsx,jsx,java,go,rs,c,cc,cpp,h,hpp,cs,rb,php,swift,kt,kts,scala,sh,bash,zsh,sql",
  CONFIG_EXTENSIONS: "yml,yaml,json,toml,ini,cfg,conf,env,properties,xml",
  MAX_WORKERS_FILE: "8",
  MAX_WORKERS_DIR: "120",
  MAX_RETRIES: "5",
  RETRY_BACKOFF_BASE: "2",
  RETRY_BACKOFF_MAX: "10",
  DATABASE_URL: "",
  RAGCONFIG_PATH: join(homedir(), ".ragconfig"),
  HOST: "0.0.0.0",
  PORT: "8200",
  ALLOW_NO_AUTH_FOR_TESTS: "false",
  EMBEDDING_DIM: "1024",
  SYMBOL_FAST_PATH_ENABLED: "true",
  SYMBOL_EXACT_LIMIT: "20",
  SYMBOL_FUZZY_LIMIT: "10",
  TREE_SITTER_GRAMMARS_PATH: "",
};

// ═══════════════════════════════════════════════════════════════════════════════
// Loader
// ═══════════════════════════════════════════════════════════════════════════════

function parseEnvFile(filePath: string): Record<string, string> {
  const vars: Record<string, string> = {};
  if (!existsSync(filePath)) return vars;

  const content = readFileSync(filePath, "utf-8");
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;
    const key = line.slice(0, eqIdx).trim();
    let val = line.slice(eqIdx + 1).trim();
    // Strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    vars[key] = val;
  }
  return vars;
}

export function loadConfig(overrides?: Record<string, string>): RagConfig {
  const fileVars = parseEnvFile(CONFIG_PATH);
  const envRaw: Record<string, string | undefined> = { ...DEFAULTS, ...fileVars, ...process.env, ...(overrides || {}) };
  const env: Record<string, string> = Object.fromEntries(
    Object.entries(envRaw).filter((entry): entry is [string, string] => entry[1] !== undefined)
  );

  const get = (key: string): string => env[key] ?? DEFAULTS[key] ?? "";
  const getNum = (key: string): number => Number(get(key));
  const getNumValidated = (key: string, min?: number, max?: number): number => {
    const val = getNum(key);
    if (Number.isNaN(val)) {
      throw new Error(`Invalid numeric value for ${key}: "${get(key)}"`);
    }
    if (min !== undefined && val < min) {
      throw new Error(`${key} must be >= ${min}, got ${val}`);
    }
    if (max !== undefined && val > max) {
      throw new Error(`${key} must be <= ${max}, got ${val}`);
    }
    return val;
  };
  const getBool = (key: string): boolean => {
    const v = get(key).toLowerCase();
    return v === "true" || v === "1" || v === "yes";
  };
  const getSet = (key: string): Set<string> =>
    new Set(get(key).split(",").map((s) => s.trim().toLowerCase()).filter(Boolean));

  // Python-parity validation: semantic_weight + bm25_weight must sum to ~1.0
  const semanticWeight = getNum("SEMANTIC_WEIGHT");
  const bm25Weight = getNum("BM25_WEIGHT");
  if (Math.abs((semanticWeight + bm25Weight) - 1.0) > 0.001) {
    throw new Error(
      `BM25 weight (${bm25Weight}) + semantic weight (${semanticWeight}) must sum to 1.0`
    );
  }

  return {
    apiBase: get("API_BASE"),
    apiKey: get("API_KEY"),
    defaultStoreId: get("DEFAULT_STORE_ID"),
    litellmBase: get("LITELLM_BASE"),
    litellmKey: get("LITELLM_KEY"),
    embeddingModel: get("EMBEDDING_MODEL"),
    embedBatchSize: getNum("EMBED_BATCH_SIZE"),
    rerankModel: get("RERANK_MODEL"),
    rerankTimeout: getNum("RERANK_TIMEOUT_MS"),
    rerankMaxAttempts: getNum("RERANK_MAX_ATTEMPTS"),
    rerankRetryDelays: get("RERANK_RETRY_DELAYS"),
    rerankCandidates: getNum("RERANK_CANDIDATES"),
    rerankTopN: getNum("RERANK_TOP_N"),
    semanticWeight: getNum("SEMANTIC_WEIGHT"),
    bm25Weight: getNum("BM25_WEIGHT"),
    hybridEnabled: getBool("HYBRID_ENABLED"),
    defaultLimit: getNum("DEFAULT_LIMIT"),
    maxLimit: getNum("MAX_LIMIT"),
    minScore: getNum("MIN_SCORE"),
    bm25Candidates: getNum("BM25_CANDIDATES"),
    bm25PageSize: getNum("BM25_PAGE_SIZE"),
    distanceOperator: get("DISTANCE_OPERATOR") as RagConfig["distanceOperator"],
    maxPerFilePreRerank: getNum("MAX_PER_FILE_PRE_RERANK"),
    codeAwareEnabled: getBool("CODE_AWARE_ENABLED"),
    codeResultWarning: get("CODE_RESULT_WARNING"),
    rateDelayInitial: getNum("RATE_DELAY_INITIAL"),
    rateDelaySustained: getNum("RATE_DELAY_SUSTAINED"),
    rateWarmupSeconds: getNum("RATE_WARMUP_SECONDS"),
    rateMaxPerMinute: getNum("RATE_MAX_PER_MINUTE"),
    rateLimitEscalateAfter: getNum("RATE_LIMIT_ESCALATE_AFTER"),
    rateLimitEscalatedInitial: getNum("RATE_LIMIT_ESCALATED_INITIAL"),
    rateLimitEscalatedSustained: getNum("RATE_LIMIT_ESCALATED_SUSTAINED"),
    rateLimitCooldownAfter: getNum("RATE_LIMIT_COOLDOWN_AFTER"),
    rateLimitCooldownSecs: getNum("RATE_LIMIT_COOLDOWN_SECS"),
    maxChunkChars: getNum("MAX_CHUNK_CHARS"),
    codeChunkMaxLines: getNum("CODE_CHUNK_MAX_LINES"),
    codeChunkMaxChars: getNum("CODE_CHUNK_MAX_CHARS"),
    configChunkMaxLines: getNum("CONFIG_CHUNK_MAX_LINES"),
    configChunkMaxChars: getNum("CONFIG_CHUNK_MAX_CHARS"),
    maxFileBytes: getNum("MAX_FILE_BYTES"),
    defaultExtensions: getSet("DEFAULT_EXTENSIONS"),
    codeExtensions: getSet("CODE_EXTENSIONS"),
    configExtensions: getSet("CONFIG_EXTENSIONS"),
    maxWorkersFile: getNum("MAX_WORKERS_FILE"),
    maxWorkersDir: getNum("MAX_WORKERS_DIR"),
    maxRetries: getNum("MAX_RETRIES"),
    retryBackoffBase: getNum("RETRY_BACKOFF_BASE"),
    retryBackoffMax: getNum("RETRY_BACKOFF_MAX"),
    databaseUrl: get("DATABASE_URL"),
    ragconfigPath: get("RAGCONFIG_PATH"),
    host: get("HOST"),
    port: getNumValidated("PORT", 1, 65535),
    allowNoAuthForTests: getBool("ALLOW_NO_AUTH_FOR_TESTS"),
    embeddingDim: getNumValidated("EMBEDDING_DIM", 1),
    symbolFastPathEnabled: getBool("SYMBOL_FAST_PATH_ENABLED"),
    symbolExactLimit: getNumValidated("SYMBOL_EXACT_LIMIT", 1),
    symbolFuzzyLimit: getNumValidated("SYMBOL_FUZZY_LIMIT", 1),
    treeSitterGrammarsPath: get("TREE_SITTER_GRAMMARS_PATH"),
  };
}

/** Singleton config instance */
let _config: RagConfig | null = null;

export function getConfig(): RagConfig {
  if (!_config) _config = loadConfig();
  return _config;
}

export function resetConfig(): void {
  _config = null;
}

import type { VectorStoreConfig } from "./vector-store.js";

export function toVectorStoreConfig(cfg: RagConfig): VectorStoreConfig {
  return {
    apiBase: cfg.apiBase,
    apiKey: cfg.apiKey,
    storeId: cfg.defaultStoreId,
    embeddingBase: cfg.litellmBase,
    embeddingKey: cfg.litellmKey,
    embeddingModel: cfg.embeddingModel,
    rerankModel: cfg.rerankModel,
    rerankTimeout: cfg.rerankTimeout,
    rerankMaxAttempts: cfg.rerankMaxAttempts,
    rerankRetryDelays: cfg.rerankRetryDelays,
    rerankCandidates: cfg.rerankCandidates,
    rerankTopN: cfg.rerankTopN,
    semanticWeight: cfg.semanticWeight,
    bm25Weight: cfg.bm25Weight,
    hybridEnabled: cfg.hybridEnabled,
    defaultLimit: cfg.defaultLimit,
    maxLimit: cfg.maxLimit,
    minScore: cfg.minScore,
    bm25Candidates: cfg.bm25Candidates,
    bm25PageSize: cfg.bm25PageSize,
    distanceOperator: cfg.distanceOperator,
    codeAwareEnabled: cfg.codeAwareEnabled,
    codeResultWarning: cfg.codeResultWarning,
    databaseUrl: cfg.databaseUrl,
    maxPerFilePreRerank: cfg.maxPerFilePreRerank,
    symbolFastPathEnabled: cfg.symbolFastPathEnabled,
    symbolExactLimit: cfg.symbolExactLimit,
    symbolFuzzyLimit: cfg.symbolFuzzyLimit,
  };
}

