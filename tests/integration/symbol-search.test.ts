import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { randomUUID } from "crypto";
import { createVectorDB, type VectorStoreDB } from "../../lib/db-adapter.js";
import { ensureSchema, insertEmbeddingBatch } from "../../lib/schema.js";
import { VectorStoreSearchEngine, createSearchEngine } from "../../lib/vector-store.js";
import { loadConfig, toVectorStoreConfig } from "../../lib/config.js";

// Integration tests require a live Postgres+pgvector instance.
// Set DATABASE_URL in the environment before running.
const skipIntegration = !process.env.DATABASE_URL;

describe.skipIf(skipIntegration)("Symbol Fast Path", () => {
  let db: VectorStoreDB;
  let searchEngine: VectorStoreSearchEngine;
  const testStoreId = randomUUID();
  const testProjectId = "/symbol/test-" + Date.now();

  beforeAll(async () => {
    const cfg = loadConfig();
    db = createVectorDB({ databaseUrl: cfg.databaseUrl });
    await ensureSchema(db, cfg.embeddingDim);

    await db.query(
      `INSERT INTO vector_stores (id, name) VALUES ($1::uuid, $2)
       ON CONFLICT DO NOTHING`,
      [testStoreId, "symbol-search-test"]
    );

    const baseConfig = toVectorStoreConfig(cfg);
    searchEngine = createSearchEngine(db, baseConfig, {
      storeId: testStoreId,
      symbolFastPathEnabled: true,
    });

    // No external HTTP calls — embeddings are constructed locally.
    const mockEmbedding = new Array(1024).fill(0.1) as number[];
    await insertEmbeddingBatch(db, testStoreId, [
      {
        content: "export function UserService() {}",
        semantic_content: "User service class",
        lexical_content: "UserService function export",
        embedding: mockEmbedding,
        metadata: {
          project_id: testProjectId,
          symbol_name: "UserService",
          content_type: "code",
          doc_path: "user-service.ts",
        },
      },
      {
        content: "export function userService() {}",
        semantic_content: "User service lowercase",
        lexical_content: "userService function export",
        embedding: mockEmbedding,
        metadata: {
          project_id: testProjectId,
          symbol_name: "userService",
          content_type: "code",
          doc_path: "user-service-lc.ts",
        },
      },
      {
        content: "export function MyUserServiceImpl() {}",
        semantic_content: "User service implementation",
        lexical_content: "MyUserServiceImpl function",
        embedding: mockEmbedding,
        metadata: {
          project_id: testProjectId,
          symbol_name: "MyUserServiceImpl",
          content_type: "code",
          doc_path: "my-user-service-impl.ts",
        },
      },
    ]);
  });

  afterAll(async () => {
    await db.query(
      "DELETE FROM embeddings WHERE metadata->>'project_id' = $1",
      [testProjectId]
    );
    await db.query("DELETE FROM vector_stores WHERE id = $1::uuid", [testStoreId]).catch(() => {});
    await db.close();
  });

  it("exact symbol match returns first", async () => {
    const response = await searchEngine.search({
      query: "UserService",
      limit: 3,
      filters: { project_id: testProjectId },
    });

    expect(response.results.length).toBeGreaterThan(0);
    expect(response.results[0].metadata?.symbol_name).toBe("UserService");
  });

  it("case-insensitive match works", async () => {
    const response = await searchEngine.search({
      query: "USERSERVICE",
      limit: 3,
      filters: { project_id: testProjectId },
    });

    expect(response.results.length).toBeGreaterThan(0);
    const symbolNames = response.results
      .map((r) => r.metadata?.symbol_name)
      .filter((s): s is string => typeof s === "string");
    expect(symbolNames.some((s) => s.toLowerCase() === "userservice")).toBe(true);
  });

  it("fuzzy match finds partial", async () => {
    const response = await searchEngine.search({
      query: "UserServiceImpl",
      limit: 3,
      filters: { project_id: testProjectId },
    });

    const symbolNames = response.results
      .map((r) => r.metadata?.symbol_name)
      .filter((s): s is string => typeof s === "string");
    expect(symbolNames.some((s) => s.includes("UserService"))).toBe(true);
  });
});
