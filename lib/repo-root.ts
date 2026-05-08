// ---------------------------------------------------------------------------
// repo-root.ts — Git repository root detection
//
// Extracted verbatim from Zenith-MCP's src/core/symbol-index.ts (lines 55-69)
// so Zenith-Rag does NOT have to pull in the full symbol-index module
// (which depends on better-sqlite3 and the rest of the on-disk symbol DB
// machinery). The function itself has no Zenith-MCP-specific dependencies —
// it just shells out to `git rev-parse --show-toplevel`.
//
// This is the same shared utility used by project-scope.ts; keeping the
// implementation byte-identical preserves parity with Zenith-MCP's repo
// detection behaviour.
// ---------------------------------------------------------------------------

import { statSync } from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

export function findRepoRoot(filePath: string): string | null {
    try {
        const stat = statSync(filePath);
        const cwd = stat.isDirectory() ? filePath : path.dirname(filePath);
        const result = execFileSync('git', ['rev-parse', '--show-toplevel'], {
            cwd,
            encoding: 'utf-8',
            timeout: 5000,
            stdio: ['ignore', 'pipe', 'ignore'],
        });
        return result.trim();
    } catch {
        return null;
    }
}
