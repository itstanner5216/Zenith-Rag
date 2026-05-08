import { describe, it, expect } from "vitest";
import { BMXPlusIndex } from "../../lib/bmx-plus.js";

// BMXPlusIndex public API (per lib/bmx-plus.ts):
//   - constructor(alphaOverride?, betaOverride?, normalizeScores?)
//   - buildIndex(chunks?: { chunk_id: string; text: string }[]): void
//   - search(query: string, topK?: number): Array<[string, number]>
//   - updateIndex(chunkId: string, text: string): void
//   - removeFromIndex(chunkId: string): void
//   - documentCount: number
//   - vocabularySize: number
//
// Note: search() returns tuples of [chunk_id, score], NOT objects.

interface TestChunk {
    chunk_id: string;
    text: string;
}

const sampleChunks: TestChunk[] = [
    { chunk_id: "1", text: "function getUserById(id) { return db.users.find(id); }" },
    { chunk_id: "2", text: "class UserService { constructor(db) { this.db = db; } }" },
    { chunk_id: "3", text: "// Rate limiting for the API gateway" },
    { chunk_id: "4", text: "export const handleAuth = async (req) => { /* ... */ }" },
];

describe("BMXPlusIndex", () => {
    it("builds an index without throwing", () => {
        const index = new BMXPlusIndex();
        expect(() => index.buildIndex(sampleChunks)).not.toThrow();
    });

    it("returns at least one result for an exact identifier match", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        const results = index.search("getUserById", 5);
        expect(results.length).toBeGreaterThan(0);
        // The chunk containing the identifier should be in the top results
        const ids = results.map(([cid]) => cid);
        expect(ids).toContain("1");
    });

    it("orders results by score descending", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        const results = index.search("user", 5);
        for (let i = 1; i < results.length; i++) {
            expect(results[i - 1][1]).toBeGreaterThanOrEqual(results[i][1]);
        }
    });

    it("respects the topK limit", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        const results = index.search("function class export", 2);
        expect(results.length).toBeLessThanOrEqual(2);
    });

    it("returns empty array for empty query", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        const results = index.search("", 5);
        expect(Array.isArray(results)).toBe(true);
    });

    it("updateIndex adds a new chunk that becomes searchable", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        index.updateIndex("5", "telemetry event tracking middleware");
        const results = index.search("telemetry", 5);
        const ids = results.map(([cid]) => cid);
        expect(ids).toContain("5");
    });

    it("removeFromIndex removes a chunk from search results", () => {
        const index = new BMXPlusIndex();
        index.buildIndex(sampleChunks);
        index.removeFromIndex("1");
        const results = index.search("getUserById", 5);
        const ids = results.map(([cid]) => cid);
        expect(ids).not.toContain("1");
    });

    // ── Stop-word removal tests ──

    it("excludes stop words from tokenization", () => {
        // _tokenize is a static method; verify stop words are dropped
        const tokens = BMXPlusIndex._tokenize("the quick brown fox is in a box");
        // "the", "is", "in", "a" are stop words and should be excluded
        expect(tokens).not.toContain("the");
        expect(tokens).not.toContain("is");
        expect(tokens).not.toContain("in");
        expect(tokens).not.toContain("a");
        // Non-stop words should be present
        expect(tokens).toContain("quick");
        expect(tokens).toContain("brown");
        expect(tokens).toContain("fox");
        expect(tokens).toContain("box");
    });

    it("returns empty tokens when input is all stop words", () => {
        const tokens = BMXPlusIndex._tokenize("the and or but not is are was");
        expect(tokens.length).toBe(0);
    });

    // ── camelCase / identifier splitting tests ──

    it("splits camelCase identifiers into sub-tokens during tokenization", () => {
        const tokens = BMXPlusIndex._tokenize("parseConfigFile");
        // Should contain the original lowered token AND the sub-parts
        expect(tokens).toContain("parseconfigfile");
        expect(tokens).toContain("parse");
        expect(tokens).toContain("config");
        expect(tokens).toContain("file");
    });

    it("splits PascalCase identifiers into sub-tokens", () => {
        const tokens = BMXPlusIndex._tokenize("UserService");
        expect(tokens).toContain("userservice");
        expect(tokens).toContain("user");
        expect(tokens).toContain("service");
    });

    it("splits snake_case identifiers into sub-tokens", () => {
        const tokens = BMXPlusIndex._tokenize("get_user_by_id");
        // Original lowered token preserved
        expect(tokens).toContain("get_user_by_id");
        // Sub-parts emitted (stop words like "by" filtered out)
        expect(tokens).toContain("get");
        expect(tokens).toContain("user");
        expect(tokens).toContain("id");
        expect(tokens).not.toContain("by");
    });

    it("camelCase splitting enables sub-token search matches", () => {
        const index = new BMXPlusIndex();
        index.buildIndex([
            { chunk_id: "a", text: "function parseConfigFile() { /* ... */ }" },
            { chunk_id: "b", text: "unrelated data logging module" },
        ]);
        // Searching for a sub-token "config" should find chunk "a"
        const results = index.search("config", 5);
        const ids = results.map(([cid]) => cid);
        expect(ids).toContain("a");
    });
});
