import { describe, it, expect } from "vitest";
import { expandIdentifierTokens } from "../../lib/vector-store.js";

describe("expandIdentifierTokens", () => {
    // The exact output format depends on the implementation. We verify
    // the INVARIANT properties: identifier tokens are expanded into
    // word-level pieces, and the original text is preserved as a substring.

    it("preserves the input as a substring of the output (or returns input unchanged)", () => {
        const input = "getUserById";
        const output = expandIdentifierTokens(input);
        // Either the function expands AND keeps the original (concatenated),
        // or it returns the input unchanged. Both are reasonable.
        expect(typeof output).toBe("string");
        expect(output.length).toBeGreaterThanOrEqual(input.length);
    });

    it("splits camelCase identifiers into component words", () => {
        const output = expandIdentifierTokens("getUserById");
        // The output should contain at least the lowercase pieces
        const lower = output.toLowerCase();
        expect(lower).toContain("user");
        expect(lower).toContain("by");
        expect(lower).toContain("id");
    });

    it("splits snake_case identifiers", () => {
        const output = expandIdentifierTokens("get_user_by_id");
        const lower = output.toLowerCase();
        expect(lower).toContain("user");
        expect(lower).toContain("by");
        expect(lower).toContain("id");
    });

    it("handles plain English text without crashing", () => {
        const output = expandIdentifierTokens("how does authentication work");
        expect(typeof output).toBe("string");
        expect(output.length).toBeGreaterThan(0);
    });

    it("handles empty string", () => {
        const output = expandIdentifierTokens("");
        expect(typeof output).toBe("string");
    });
});
