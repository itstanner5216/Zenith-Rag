import { describe, it, expect } from "vitest";
import { ALLOWED_FILTER_KEYS, buildMetadataFilter } from "../../lib/schema.js";

describe("ALLOWED_FILTER_KEYS", () => {
    it("is a non-empty Set", () => {
        expect(ALLOWED_FILTER_KEYS).toBeInstanceOf(Set);
        expect(ALLOWED_FILTER_KEYS.size).toBeGreaterThan(0);
    });

    it("contains the canonical keys used by the indexer and search engine", () => {
        // These are the keys both rag-index and rag-search rely on.
        // If the whitelist drops any of them, the production code breaks.
        expect(ALLOWED_FILTER_KEYS.has("project_id")).toBe(true);
        expect(ALLOWED_FILTER_KEYS.has("repo_root")).toBe(true);
        expect(ALLOWED_FILTER_KEYS.has("content_type")).toBe(true);
        expect(ALLOWED_FILTER_KEYS.has("doc_path")).toBe(true);
        expect(ALLOWED_FILTER_KEYS.has("symbol_name")).toBe(true);
        expect(ALLOWED_FILTER_KEYS.has("content_hash")).toBe(true);
    });

    it("rejects malicious SQL-injection-style keys", () => {
        expect(ALLOWED_FILTER_KEYS.has("'; DROP TABLE embeddings; --")).toBe(false);
        expect(ALLOWED_FILTER_KEYS.has("metadata->>'bogus")).toBe(false);
        expect(ALLOWED_FILTER_KEYS.has("any random key")).toBe(false);
    });
});

describe("buildMetadataFilter", () => {
    it("returns 'TRUE' SQL for an empty filter object", () => {
        const result = buildMetadataFilter({});
        expect(result.sql).toBe("TRUE");
        expect(result.params).toEqual([]);
    });

    it("emits parameterized clauses for whitelisted keys", () => {
        const result = buildMetadataFilter({ project_id: "/repo", content_type: "code" });
        // Parameterized — values come back via params, keys interpolated literally
        // since they're whitelisted.
        expect(result.params).toEqual(["/repo", "code"]);
        expect(result.sql).toContain("metadata->>'project_id'");
        expect(result.sql).toContain("metadata->>'content_type'");
        expect(result.sql).toContain("$1");
        expect(result.sql).toContain("$2");
    });

    it("THROWS on a non-whitelisted key — does not silently drop", () => {
        expect(() =>
            buildMetadataFilter({ "'; DROP TABLE embeddings; --": "x" })
        ).toThrow(/Invalid filter key/);
    });

    it("THROWS on any unknown key, even benign-looking ones", () => {
        expect(() =>
            buildMetadataFilter({ author: "alice" })
        ).toThrow(/Invalid filter key/);
    });
});
