import { describe, it, expect } from "vitest";
import { chunkFile, type ChunkResult } from "../../lib/chunker.js";
import { createHash } from "crypto";

describe("chunker — three-text model", () => {
    const tsSource = `export function calculateSum(a: number, b: number): number {
    return a + b;
}

export class Calculator {
    add(x: number, y: number): number { return x + y; }
}
`;

    it("produces non-empty chunks for a TypeScript file", async () => {
        const chunks = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        expect(chunks.length).toBeGreaterThan(0);
    });

    it("captures top-level functions as chunks", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        const symbols = chunks.map((c) => c.metadata.symbol_name);
        expect(symbols).toContain("calculateSum");
    });

    it("captures top-level classes as chunks", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        const symbols = chunks.map((c) => c.metadata.symbol_name);
        expect(symbols).toContain("Calculator");
    });

    it("semanticContent equals raw content (Python parity invariant)", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        for (const c of chunks) {
            expect(c.semanticContent).toBe(c.content);
        }
    });

    it("lexicalContent has Python-parity header format", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        for (const c of chunks) {
            // Header line + body
            const lines = c.lexicalContent.split("\n");
            expect(lines[0]).toMatch(/^# File: .+/);
            // The header should mention File and (when symbol_name is set) Symbol
            if (c.metadata.symbol_name) {
                expect(lines[0]).toContain(`Symbol: ${c.metadata.symbol_name}`);
            }
            expect(lines[0]).toContain("Language: typescript");
            // Pieces are joined with " | "
            expect(lines[0]).toContain(" | ");
            // The body following the header equals the raw content
            const bodyAfterHeader = lines.slice(1).join("\n");
            expect(bodyAfterHeader).toBe(c.content);
        }
    });

    it("content_hash is a full 64-char SHA-256 hex digest of the source", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/fake",
            projectId: "/fake",
        });
        const expected = createHash("sha256").update(tsSource).digest("hex");
        expect(expected).toHaveLength(64);
        for (const c of chunks) {
            expect(c.metadata.content_hash).toBe(expected);
            expect(c.metadata.content_hash).toHaveLength(64);
        }
    });

    it("metadata contains the configured projectId and repoRoot", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/sample.ts", {
            content: tsSource,
            repoRoot: "/my-repo",
            projectId: "/my-repo",
        });
        for (const c of chunks) {
            expect(c.metadata.project_id).toBe("/my-repo");
            expect(c.metadata.repo_root).toBe("/my-repo");
        }
    });

    it("falls back gracefully for unsupported languages", async () => {
        const chunks: ChunkResult[] = await chunkFile("/fake/data.unknownext", {
            content: "some text without grammar",
            repoRoot: "/fake",
            projectId: "/fake",
        });
        // Either zero chunks or one fallback chunk; either is acceptable.
        expect(Array.isArray(chunks)).toBe(true);
    });
});
