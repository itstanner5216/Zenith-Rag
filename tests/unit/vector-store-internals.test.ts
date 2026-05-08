/**
 * vector-store-internals.test.ts — Unit tests for pure internal functions
 * in lib/vector-store.ts that don't require a database connection.
 *
 * Covers:
 *   - normalizeScores: min-max normalization with degenerate cases
 *   - SYMBOL_QUERY_RE: regex used by the symbol fast path
 *   - expandIdentifierTokens: token expansion (additional cases beyond
 *     the dedicated identifier-tokens.test.ts)
 *
 * These tests run entirely in-memory — no DATABASE_URL needed.
 */

import { describe, it, expect } from "vitest";
import {
  normalizeScores,
  expandIdentifierTokens,
  SYMBOL_QUERY_RE,
} from "../../lib/vector-store.js";

// ═══════════════════════════════════════════════════════════════════════════════
// normalizeScores
// ═══════════════════════════════════════════════════════════════════════════════

describe("normalizeScores", () => {
  it("returns empty map for empty input", () => {
    const result = normalizeScores(new Map());
    expect(result.size).toBe(0);
  });

  it("maps a single entry to 1.0", () => {
    const result = normalizeScores(new Map([["a", 42]]));
    expect(result.get("a")).toBe(1.0);
  });

  it("normalizes two entries to [0, 1]", () => {
    const result = normalizeScores(
      new Map([
        ["low", 10],
        ["high", 20],
      ])
    );
    expect(result.get("low")).toBeCloseTo(0.0, 9);
    expect(result.get("high")).toBeCloseTo(1.0, 9);
  });

  it("normalizes three entries linearly", () => {
    const result = normalizeScores(
      new Map([
        ["a", 0],
        ["b", 5],
        ["c", 10],
      ])
    );
    expect(result.get("a")).toBeCloseTo(0.0, 9);
    expect(result.get("b")).toBeCloseTo(0.5, 9);
    expect(result.get("c")).toBeCloseTo(1.0, 9);
  });

  it("handles all-identical scores (zero spread) by mapping to 1.0", () => {
    const result = normalizeScores(
      new Map([
        ["a", 7],
        ["b", 7],
        ["c", 7],
      ])
    );
    for (const v of result.values()) {
      expect(v).toBe(1.0);
    }
  });

  it("handles negative scores correctly", () => {
    const result = normalizeScores(
      new Map([
        ["neg", -10],
        ["zero", 0],
        ["pos", 10],
      ])
    );
    expect(result.get("neg")).toBeCloseTo(0.0, 9);
    expect(result.get("zero")).toBeCloseTo(0.5, 9);
    expect(result.get("pos")).toBeCloseTo(1.0, 9);
  });

  it("handles very small spread (near-epsilon)", () => {
    // Spread is 1e-10 which is below the 1e-9 threshold
    const result = normalizeScores(
      new Map([
        ["a", 1.0],
        ["b", 1.0 + 1e-10],
      ])
    );
    // Both should map to 1.0 since spread < 1e-9
    expect(result.get("a")).toBe(1.0);
    expect(result.get("b")).toBe(1.0);
  });

  it("preserves all keys from input", () => {
    const input = new Map([
      ["x", 1],
      ["y", 2],
      ["z", 3],
    ]);
    const result = normalizeScores(input);
    expect(result.size).toBe(3);
    expect(result.has("x")).toBe(true);
    expect(result.has("y")).toBe(true);
    expect(result.has("z")).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SYMBOL_QUERY_RE (symbol fast-path regex)
// ═══════════════════════════════════════════════════════════════════════════════

describe("SYMBOL_QUERY_RE (symbol fast-path)", () => {
  // Positive cases — should match (these trigger the fast path)
  it.each([
    ["simple identifier", "getUserById"],
    ["single word", "foo"],
    ["PascalCase", "UserService"],
    ["underscore prefix", "_private"],
    ["dollar prefix", "$scope"],
    ["mixed dollars", "Cls$inner"],
    ["dotted namespace", "ns.foo"],
    ["deep dotted", "a.b.c.d"],
    ["dotted with mixed case", "React.Component"],
    ["ALL_CAPS", "MAX_RETRIES"],
    ["single char", "x"],
    ["underscore only start", "_"],
    ["dollar only start", "$"],
  ])("matches %s: '%s'", (_label, input) => {
    expect(SYMBOL_QUERY_RE.test(input)).toBe(true);
  });

  // Negative cases — should NOT match (these skip the fast path)
  it.each([
    ["natural language", "how does authentication work"],
    ["space in query", "get user"],
    ["starts with number", "123abc"],
    ["starts with dot", ".foo"],
    ["trailing dot", "foo."],
    ["double dot", "foo..bar"],
    ["contains operator", "a+b"],
    ["contains parens", "foo()"],
    ["contains brackets", "arr[0]"],
    ["empty string", ""],
    ["just spaces", "   "],
    ["contains slash", "path/to/file"],
    ["SQL injection attempt", "'; DROP TABLE embeddings; --"],
    ["hyphenated", "my-function"],
    ["contains colon", "foo:bar"],
    ["contains equals", "a=b"],
  ])("rejects %s: '%s'", (_label, input) => {
    expect(SYMBOL_QUERY_RE.test(input.trim())).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// expandIdentifierTokens — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════════

describe("expandIdentifierTokens — edge cases", () => {
  it("expands XMLParser (ALLCAPS transition)", () => {
    const result = expandIdentifierTokens("XMLParser");
    const lower = result.toLowerCase();
    expect(lower).toContain("xml");
    expect(lower).toContain("parser");
    // Should preserve the original token
    expect(result).toContain("XMLParser");
  });

  it("handles multiple tokens with different patterns", () => {
    const result = expandIdentifierTokens("getUserById rate_limiter");
    const lower = result.toLowerCase();
    // camelCase expansion
    expect(lower).toContain("user");
    expect(lower).toContain("by");
    // snake_case expansion
    expect(lower).toContain("rate");
    expect(lower).toContain("limiter");
  });

  it("does not expand single-word tokens without patterns", () => {
    const result = expandIdentifierTokens("authentication");
    // No camelCase, no underscore, no ALLCAPS transition → should return as-is
    expect(result).toBe("authentication");
  });

  it("handles ALLCAPS_ONLY (no transition → no expansion)", () => {
    const result = expandIdentifierTokens("HTTP");
    // No lowercase transition → HAS_ALLCAPS_TRANSITION_RE won't match
    // and no camelCase → returns unchanged
    expect(result).toBe("HTTP");
  });

  it("expands HTTPSServer", () => {
    const result = expandIdentifierTokens("HTTPSServer");
    const lower = result.toLowerCase();
    // Should split the ALLCAPS→lowercase transition
    expect(lower).toContain("server");
  });
});
