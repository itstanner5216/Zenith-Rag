import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { randomUUID } from "crypto";
import { createVectorDB, type VectorStoreDB } from "../../lib/db-adapter.js";
import { ensureSchema, insertEmbeddingBatch } from "../../lib/schema.js";
import { VectorStoreSearchEngine, createSearchEngine } from "../../lib/vector-store.js";
import { chunkFile } from "../../lib/chunker.js";
import { loadConfig, toVectorStoreConfig } from "../../lib/config.js";

// Integration tests require a live Postgres+pgvector instance.
// Set DATABASE_URL in the environment before running.
//   DATABASE_URL=postgres://user:pass@localhost/testdb npm run test
//
// When DATABASE_URL is unset the suite skips cleanly via
// describe.skipIf — it does NOT error.
const skipIntegration = !process.env.DATABASE_URL;

describe.skipIf(skipIntegration)("RAG Flow Integration", () => {
  let db: VectorStoreDB;
  let searchEngine: VectorStoreSearchEngine;
  const testStoreId = randomUUID();
  const testProjectId = "/test/project-" + Date.now();

  beforeAll(async () => {
    const cfg = loadConfig();
    db = createVectorDB({ databaseUrl: cfg.databaseUrl });
    await ensureSchema(db, cfg.embeddingDim);

    await db.query(
      `INSERT INTO vector_stores (id, name) VALUES ($1::uuid, $2)
       ON CONFLICT DO NOTHING`,
      [testStoreId, "rag-flow-test"]
    );

    const baseConfig = toVectorStoreConfig(cfg);
    searchEngine = createSearchEngine(db, baseConfig, { storeId: testStoreId });
  });

  afterAll(async () => {
    // Clean up test data — delete by metadata so we don't depend
    // on the FK cascade ordering.
    await db.query(
      "DELETE FROM embeddings WHERE metadata->>'project_id' = $1",
      [testProjectId]
    );
    await db.query("DELETE FROM vector_stores WHERE id = $1::uuid", [testStoreId]).catch(() => {});
    await db.close();
  });

  it("chunks TypeScript file with symbol extraction", async () => {
    const testCode = `
export function calculateSum(a: number, b: number): number {
  return a + b;
}

export class Calculator {
  add(x: number, y: number): number { return x + y; }
}
`;
    const chunks = await chunkFile("/fake/test.ts", {
      content: testCode,
      repoRoot: testProjectId,
      projectId: testProjectId,
    });

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.some((c) => c.metadata.symbol_name === "calculateSum")).toBe(true);
    expect(chunks.some((c) => c.metadata.symbol_name === "Calculator")).toBe(true);
  });

  it("inserts and retrieves embeddings via symbol fast path", async () => {
    // Use a 1024-dim mock vector matching cfg.embeddingDim default.
    // No external HTTP calls — embedding is constructed locally.
    const mockEmbedding = new Array(1024).fill(0.1) as number[];

    await insertEmbeddingBatch(db, testStoreId, [
      {
        content: "function testFunc() { return 42; }",
        semantic_content: "A function that returns the answer",
        lexical_content: "function testFunc return 42",
        embedding: mockEmbedding,
        metadata: {
          project_id: testProjectId,
          symbol_name: "testFunc",
          content_type: "code",
          doc_path: "test.ts",
        },
      },
    ]);

    const response = await searchEngine.search({
      query: "testFunc",
      limit: 1,
      filters: { project_id: testProjectId },
    });

    expect(response.results.length).toBeGreaterThan(0);
    expect(response.results[0].metadata?.symbol_name).toBe("testFunc");
  });
});
