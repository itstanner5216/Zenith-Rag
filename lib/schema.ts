// ---------------------------------------------------------------------------
// schema.ts — Postgres + pgvector schema bootstrap and helpers
//
// Mirrors the Python pgvector deployment's init.sql so a freshly provisioned
// database can be brought up by calling ensureSchema() — no manual psql, no
// hand-managed migrations. Also exposes the per-document hash map used by
// rag-upload --sync, the bulk delete used by rag_index reindex, the chunked
// batch insert used by the upload pipeline, and a metadata-filter builder
// that whitelists every key against ALLOWED_FILTER_KEYS to prevent SQL
// injection through the JSON metadata column.
// ---------------------------------------------------------------------------

import type { VectorStoreDB } from "./db-adapter.js";

// Allowed metadata filter keys — whitelist to prevent SQL injection.
// Any caller that passes a key outside this set must be rejected at the
// API boundary; never silently drop unknown keys.
export const ALLOWED_FILTER_KEYS = new Set<string>([
    "project_id",
    "repo_root",
    "content_type",
    "doc_path",
    "source",
    "language",
    "symbol_name",
    "chunk_type",
    "content_hash",
]);

export async function ensureSchema(db: VectorStoreDB, vectorDim = 1024): Promise<void> {
    await db.query(`CREATE EXTENSION IF NOT EXISTS vector`);

    // vector_stores table
    await db.query(`
        CREATE TABLE IF NOT EXISTS vector_stores (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        )
    `);

    // embeddings table with three-text model
    await db.query(`
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            vector_store_id UUID NOT NULL REFERENCES vector_stores(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            semantic_content TEXT,
            lexical_content TEXT,
            embedding vector(${vectorDim}),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    `);

    // chunk_symbols table (matches Python init.sql)
    await db.query(`
        CREATE TABLE IF NOT EXISTS chunk_symbols (
            id SERIAL PRIMARY KEY,
            project_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL UNIQUE,
            symbol_name TEXT,
            symbol_type TEXT,
            file_path TEXT,
            language TEXT,
            parent_symbol TEXT,
            start_line INT,
            end_line INT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    `);

    // Indexes
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_embeddings_store ON embeddings(vector_store_id)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_embeddings_project ON embeddings((metadata->>'project_id'))`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_embeddings_doc_path ON embeddings((metadata->>'doc_path'))`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash ON embeddings((metadata->>'content_hash'))`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_embeddings_symbol_name ON embeddings((metadata->>'symbol_name'))`
    );
    // chunk_symbols indexes — full set matching Python init.sql so
    // queries by parent_symbol (call-graph traversal), symbol_type
    // (filter by class/function/etc.), and chunk_id (UNIQUE join key)
    // all hit an index. Missing any of these caused full-table scans
    // in the Python deployment under realistic project sizes.
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_project ON chunk_symbols(project_id)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_name ON chunk_symbols(symbol_name)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_parent ON chunk_symbols(parent_symbol)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_file ON chunk_symbols(file_path)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_type ON chunk_symbols(symbol_type)`
    );
    await db.query(
        `CREATE INDEX IF NOT EXISTS idx_chunk_symbols_chunk_id ON chunk_symbols(chunk_id)`
    );
}

export interface SourceHashEntry {
    content_hash: string;
    chunk_count: number;
}

export async function getExistingSourceHashes(
    db: VectorStoreDB,
    storeId: string,
    projectId: string
): Promise<Map<string, SourceHashEntry>> {
    const rows = await db.query<{
        doc_path: string;
        content_hash: string;
        chunk_count: string;
    }>(
        `SELECT
             metadata->>'doc_path'     AS doc_path,
             metadata->>'content_hash' AS content_hash,
             COUNT(*)::text            AS chunk_count
         FROM embeddings
         WHERE vector_store_id = $1 AND metadata->>'project_id' = $2
         GROUP BY metadata->>'doc_path', metadata->>'content_hash'`,
        [storeId, projectId]
    );

    const map = new Map<string, SourceHashEntry>();
    for (const row of rows) {
        if (row.doc_path) {
            map.set(row.doc_path, {
                content_hash: row.content_hash,
                chunk_count: parseInt(row.chunk_count, 10),
            });
        }
    }
    return map;
}

export async function deleteEmbeddingsForDocs(
    db: VectorStoreDB,
    storeId: string,
    projectId: string,
    docPaths: string[]
): Promise<number> {
    if (docPaths.length === 0) return 0;

    let deleted = 0;
    const batchSize = 100;

    for (let i = 0; i < docPaths.length; i += batchSize) {
        const batch = docPaths.slice(i, i + batchSize);
        const placeholders = batch.map((_, idx) => `$${idx + 3}`).join(", ");

        const result = await db.query<{ count: string }>(
            `WITH deleted AS (
                 DELETE FROM embeddings
                 WHERE vector_store_id = $1
                   AND metadata->>'project_id' = $2
                   AND metadata->>'doc_path' IN (${placeholders})
                 RETURNING id
             )
             SELECT COUNT(*)::text AS count FROM deleted`,
            [storeId, projectId, ...batch]
        );
        deleted += parseInt(result[0]?.count ?? "0", 10);
    }

    return deleted;
}

export interface EmbeddingInsertItem {
    content: string;
    semantic_content: string;
    lexical_content: string;
    embedding: number[];
    metadata: Record<string, unknown>;
}

export async function insertEmbeddingBatch(
    db: VectorStoreDB,
    storeId: string,
    items: EmbeddingInsertItem[]
): Promise<void> {
    if (items.length === 0) return;

    // Chunk in groups of 100 rows. With 5 per-row params + 1 shared
    // store-id param this stays well under Postgres' 65535-param limit
    // while still amortizing the per-statement overhead. 1000 rows
    // becomes 10 round-trips instead of 1000.
    const BATCH_SIZE = 100;

    for (let offset = 0; offset < items.length; offset += BATCH_SIZE) {
        const slice = items.slice(offset, offset + BATCH_SIZE);
        const params: unknown[] = [storeId];
        const valuesSql: string[] = [];

        for (let i = 0; i < slice.length; i++) {
            const item = slice[i];
            // Per-row params start at $2 because $1 is store_id.
            // Each row uses 5 params: content, semantic, lexical, embedding, metadata.
            const base = i * 5 + 2;
            valuesSql.push(
                `($1, $${base}, $${base + 1}, $${base + 2}, $${base + 3}::vector, $${base + 4}::jsonb)`
            );
            params.push(
                item.content,
                item.semantic_content,
                item.lexical_content,
                `[${item.embedding.join(",")}]`,
                JSON.stringify(item.metadata)
            );
        }

        const sql =
            `INSERT INTO embeddings (vector_store_id, content, semantic_content, lexical_content, embedding, metadata) VALUES ` +
            valuesSql.join(", ");

        await db.query(sql, params);
    }
}

export interface MetadataFilterResult {
    sql: string;
    params: unknown[];
}

export function buildMetadataFilter(
    filters: Record<string, string>,
    startParamIndex = 1
): MetadataFilterResult {
    const conditions: string[] = [];
    const params: unknown[] = [];
    let paramIdx = startParamIndex;

    for (const [key, value] of Object.entries(filters)) {
        if (!ALLOWED_FILTER_KEYS.has(key)) {
            throw new Error(
                `Invalid filter key: '${key}'. Allowed keys are: ${Array.from(ALLOWED_FILTER_KEYS).join(", ")}`
            );
        }
        // Key is whitelisted — safe to interpolate; value is parameterised.
        conditions.push(`metadata->>'${key}' = $${paramIdx}`);
        params.push(value);
        paramIdx++;
    }

    return {
        sql: conditions.length > 0 ? conditions.join(" AND ") : "TRUE",
        params,
    };
}
