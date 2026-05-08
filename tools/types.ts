// ---------------------------------------------------------------------------
// tools/types.ts — Tool registration contract for Zenith-Rag MCP tools
//
// Mirrors _external/Zenith-MCP/src/tools/types.ts so the rag tools can be
// dropped directly into a Zenith-MCP server (or any compatible MCP host).
//
// IMPORTANT: ToolContext does NOT include a `prisma` field. Zenith-MCP's
// FilesystemContext has no DB binding — DB access is established by tool
// handlers via createVectorDB() + DATABASE_URL. Adding a prisma field here
// would re-introduce the broken assumption documented in the blocker list.
// ---------------------------------------------------------------------------

export type ToolTextContent = { type: "text"; text: string };
export type ToolImageContent = { type: "image"; data: string; mimeType: string };
export type ToolAudioContent = { type: "audio"; data: string; mimeType: string };
export type ToolBlobContent = { type: "blob"; data: string; mimeType: string };

export type ToolContent =
    | ToolTextContent
    | ToolImageContent
    | ToolAudioContent
    | ToolBlobContent;

export type ToolResult = { content: ToolContent[] };

export type ToolHandler<TArgs> = (args: TArgs) => Promise<ToolResult> | ToolResult;

export type ToolRegistration = {
    title?: string;
    description?: string;
    inputSchema?: unknown;
    annotations?: {
        readOnlyHint?: boolean;
        idempotentHint?: boolean;
        destructiveHint?: boolean;
    };
};

export type ToolServer = {
    registerTool<TArgs>(
        name: string,
        registration: ToolRegistration,
        handler: ToolHandler<TArgs>
    ): void;
};

// Mirrors Zenith-MCP's FilesystemContext exactly. No prisma field
// (DB access is established by tool handlers, not injected via
// context), no `cwd` or `allowedDirectories` direct fields — those
// belonged to an earlier draft of this contract that diverged from
// Zenith-MCP. The canonical Zenith-MCP shape exposes:
//
//   * sessionId       — opaque per-session identifier (optional)
//   * validatePath()  — resolves a user-supplied path against allowed
//                       directories and resolves symlinks; throws on
//                       paths outside the sandbox
//   * getAllowedDirectories() — returns the directories the host has
//                       permitted this tool invocation to touch
//   * setAllowedDirectories() — host-side mutator, rarely called by
//                       tool code itself
//
// Tools that need a working directory should call
// getAllowedDirectories()[0] (or fall back to process.cwd() ONLY when
// no allowed directories are configured). Tools that take an explicit
// path argument MUST call validatePath() on it before touching the
// filesystem so traversal attacks are caught at the boundary.
export type ToolContext = {
    sessionId?: string;
    validatePath(inputPath: string): Promise<string>;
    getAllowedDirectories(): string[];
    setAllowedDirectories(directories: string[]): void;
};

export function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
