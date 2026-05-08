// ---------------------------------------------------------------------------
// tools/index.ts — MCP tool registration barrel
//
// Re-exports the per-tool registrar functions plus a one-call helper that
// mounts every Zenith-Rag tool onto a ToolServer-compatible host (typically
// a Zenith-MCP server, but the structural ToolServer contract makes it work
// with any compatible host).
//
// NOTE: rag-index.ts and rag-search.ts both export a bare `register` function
// (not `registerRagIndexTool` / `registerRagSearchTool`). We re-export them
// under the conventional names so callers have a stable public API.
// ---------------------------------------------------------------------------

export type { ToolServer, ToolContext, ToolResult, ToolHandler, ToolRegistration, ToolContent } from "./types.js";
export { errorMessage } from "./types.js";

import type { ToolServer, ToolContext } from "./types.js";
import { register as registerRagIndexTool } from "./rag-index.js";
import { register as registerRagSearchTool } from "./rag-search.js";

export { registerRagIndexTool, registerRagSearchTool };

/**
 * Register every Zenith-Rag MCP tool onto a host server.
 *
 * `ctx` mirrors Zenith-MCP's FilesystemContext: methods, no direct
 * fields. When the host is a real Zenith-MCP server, pass its
 * filesystem context directly. When embedding stand-alone (tests,
 * scripts), construct a minimal ctx scoped to a single allowed
 * directory — see `defaultStandaloneContext()` below.
 *
 * Hosts that already register tools selectively can call the
 * individual register* functions instead.
 */
export function registerAllRagTools(server: ToolServer, ctx: ToolContext): void {
  registerRagIndexTool(server, ctx);
  registerRagSearchTool(server, ctx);
}

/**
 * Build a minimal ToolContext suitable for stand-alone use (tests,
 * smoke scripts, embedded builds where no MCP host is wrapping these
 * tools).
 *
 * `validatePath` resolves the input against the allowed directory
 * using path.resolve and rejects anything that escapes the sandbox
 * via `..` segments. This is intentionally simpler than Zenith-MCP's
 * full validator (which also resolves symlinks); use the real
 * Zenith-MCP context whenever you have one.
 */
export function defaultStandaloneContext(allowedDir: string = process.cwd()): ToolContext {
  let allowed = [allowedDir];
  return {
    getAllowedDirectories: () => [...allowed],
    setAllowedDirectories: (dirs: string[]) => {
      allowed = [...dirs];
    },
    validatePath: async (inputPath: string): Promise<string> => {
      const path = await import("path");
      const resolved = path.resolve(inputPath);
      const okay = allowed.some(
        (root) => resolved === path.resolve(root) || resolved.startsWith(path.resolve(root) + path.sep)
      );
      if (!okay) {
        throw new Error(
          `Path "${inputPath}" resolves to "${resolved}", which is outside the allowed directories: ${allowed.join(", ")}`
        );
      }
      return resolved;
    },
  };
}
