// ---------------------------------------------------------------------------
// index.ts — Zenith-Rag package entry
//
// Top-level barrel that re-exports the lib layer, the tool registrars,
// the project-scope utilities, and the standalone HTTP server. Consumers
// should typically import from here for the public API:
//
//   import { createVectorDB, registerAllRagTools, startServer } from "zenith-rag";
//
// ---------------------------------------------------------------------------

export * from "./lib/index.js";
export * from "./tools/index.js";
export {
  resolveProjectRoot,
  getProjectId,
  isWithinProject,
  clearProjectScopeCache,
  type ResolveOptions,
} from "./utils/project-scope.js";
export { createServer, startServer, type ServerHandle } from "./server/http.js";
