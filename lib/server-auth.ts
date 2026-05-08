// ---------------------------------------------------------------------------
// server-auth.ts — Bearer token authentication for the Zenith-Rag HTTP server
//
// Mirrors the timing-safe pattern used by Zenith-MCP's HTTP server. Three
// env vars are accepted in priority order so existing operators of either
// the Python pgvector server (PGVECTOR_API_KEY) or a generic API gateway
// (API_KEY) can drop in without re-configuring. ZENITH_RAG_API_KEY is the
// canonical name and wins if multiple are set.
// ---------------------------------------------------------------------------

import { timingSafeEqual } from "crypto";
import type { Request, Response, NextFunction, RequestHandler } from "express";

export function getApiKey(): string | null {
    return (
        process.env.ZENITH_RAG_API_KEY ||
        process.env.PGVECTOR_API_KEY ||
        process.env.API_KEY ||
        null
    );
}

export function requireApiKey(allowNoAuth = false): string {
    const key = getApiKey();
    if (!key && !allowNoAuth) {
        throw new Error(
            "API key required. Set ZENITH_RAG_API_KEY, PGVECTOR_API_KEY, or API_KEY environment variable."
        );
    }
    return key ?? "";
}

export function authenticateBearer(
    authHeader: string | undefined,
    expectedKey: string
): boolean {
    if (!authHeader?.startsWith("Bearer ")) return false;
    const token = authHeader.slice(7);
    if (token.length !== expectedKey.length) return false;

    const tokenBuf = Buffer.from(token);
    const expectedBuf = Buffer.from(expectedKey);
    return timingSafeEqual(tokenBuf, expectedBuf);
}

export function authMiddleware(expectedKey: string): RequestHandler {
    return (req: Request, res: Response, next: NextFunction) => {
        if (!authenticateBearer(req.headers.authorization, expectedKey)) {
            res.status(401).json({ error: "Unauthorized" });
            return;
        }
        next();
    };
}
