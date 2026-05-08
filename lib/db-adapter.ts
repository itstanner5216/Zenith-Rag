// ---------------------------------------------------------------------------
// db-adapter.ts — Direct Postgres + pgvector connection adapter
//
// Replaces the prior Prisma-based adapter. Zenith-MCP has no ctx.prisma field
// (its ToolContext is filesystem-scoped only), so the previous implementation
// was unusable inside MCP tool handlers. This module exposes a thin
// VectorStoreDB interface backed by node-postgres' connection pool, so the
// rest of the RAG pipeline can issue parameterised SQL (including pgvector
// operators that Prisma's typed builder cannot express) without a Prisma
// dependency.
// ---------------------------------------------------------------------------

import { Pool, PoolClient } from "pg";

export interface VectorStoreDB {
    query<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T[]>;
    transaction<T>(fn: (client: PoolClient) => Promise<T>): Promise<T>;
    close(): Promise<void>;
}

export class PgVectorDB implements VectorStoreDB {
    private pool: Pool;

    constructor(connectionString: string) {
        this.pool = new Pool({ connectionString });
    }

    async query<T = Record<string, unknown>>(
        sql: string,
        params: unknown[] = []
    ): Promise<T[]> {
        const result = await this.pool.query(sql, params);
        return result.rows as T[];
    }

    async transaction<T>(fn: (client: PoolClient) => Promise<T>): Promise<T> {
        const client = await this.pool.connect();
        try {
            await client.query("BEGIN");
            const result = await fn(client);
            await client.query("COMMIT");
            return result;
        } catch (e) {
            await client.query("ROLLBACK");
            throw e;
        } finally {
            client.release();
        }
    }

    async close(): Promise<void> {
        await this.pool.end();
    }
}

export interface CreateVectorDBOptions {
    databaseUrl?: string;
    pool?: Pool;
}

export function createVectorDB(options?: CreateVectorDBOptions): VectorStoreDB {
    const url = options?.databaseUrl || process.env.DATABASE_URL;
    if (!url) {
        throw new Error(
            "DATABASE_URL required. Set the DATABASE_URL environment variable or pass options.databaseUrl explicitly."
        );
    }
    return new PgVectorDB(url);
}
