/**
 * chunker.ts — Symbol-aware code chunking using Zenith's tree-sitter engine.
 *
 * Replaces the entire Python chunker_ast.py by leveraging getDefinitions()
 * from the tree-sitter adapter — getDefinitions() is async and takes
 * (source, langName), so chunkFile() is async and reads source eagerly
 * (or accepts pre-read content via ChunkOptions.content).
 *
 * Output: a list of ChunkResult objects implementing the three-text
 * model: `content` (raw symbol body), `semanticContent` (text fed to
 * the embedding model), and `lexicalContent` (text fed to BMX+).
 */

import { getDefinitions, getLangForFile, isSupported, type SymbolInfo } from "./tree-sitter.js";
import { resolveProjectRoot, getProjectId } from "../utils/project-scope.js";
import { createHash } from "crypto";
import { basename, dirname, relative } from "path";
import { readFileSync } from "fs";

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface ChunkResult {
  content: string;
  semanticContent: string;
  lexicalContent: string;
  metadata: ChunkMetadata;
}

export interface ChunkMetadata {
  doc_path: string;
  filename: string;
  language: string;
  content_type: "code" | "markdown" | "text";
  symbol_name: string;
  chunk_type: string;
  start_line: number;
  end_line: number;
  chunk_chars: number;
  content_hash: string;
  project_id: string;
  repo_root: string;
}

export interface ChunkOptions {
  /** Max characters per chunk before splitting */
  maxChunkChars?: number;
  /** Overlap characters when splitting large symbols */
  overlapChars?: number;
  /** Include module-level (non-symbol) code as chunks */
  includeModuleLevel?: boolean;
  /** Pre-read file content */
  content?: string;
  /** Override project root */
  repoRoot?: string;
  /** Override project ID */
  projectId?: string;
}

const DEFAULT_OPTIONS: Pick<Required<ChunkOptions>, "maxChunkChars" | "overlapChars" | "includeModuleLevel"> = {
  maxChunkChars: 750,
  overlapChars: 200,
  includeModuleLevel: true,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Core Chunking
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Chunk a source file into symbol-boundary-aware pieces.
 *
 * Uses Zenith's tree-sitter to get real AST definitions, then produces
 * chunks aligned to function/class/method boundaries with context headers.
 */
export async function chunkFile(filePath: string, options?: ChunkOptions): Promise<ChunkResult[]> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const source = opts.content ?? readFileSync(filePath, "utf-8");
  const lines = source.split("\n");
  const filename = basename(filePath);
  // Full SHA-256 hex digest (64 chars) — matches the Python baseline's
  // file_sha256() in cli/pgvector-upload.py:305-307. Truncating loses
  // collision resistance AND breaks hash equality against externally-
  // hashed content. The DB column is TEXT, no length constraint.
  const contentHash = createHash("sha256").update(source).digest("hex");

  // Project scoping via Zenith
  const repoRoot = opts.repoRoot ?? resolveProjectRoot(filePath) ?? dirname(filePath);
  const projectId = opts.projectId ?? getProjectId(repoRoot) ?? repoRoot;

  // Get real symbol definitions from tree-sitter
  const langName = getLangForFile(filePath);
  let definitions: SymbolInfo[] | null = null;
  if (langName && isSupported(filePath)) {
    definitions = await getDefinitions(source, langName);
  }
  const chunks: ChunkResult[] = [];

  if (!definitions || definitions.length === 0) {
    // Fallback: treat entire file as one chunk (or split by size)
    const fallbackChunks = splitBySize(source, opts.maxChunkChars, opts.overlapChars);
    for (let i = 0; i < fallbackChunks.length; i++) {
      chunks.push(buildChunk({
        content: fallbackChunks[i].text,
        symbolName: filename,
        chunkType: "module",
        startLine: fallbackChunks[i].startLine,
        endLine: fallbackChunks[i].endLine,
        filePath,
        filename,
        language: detectLanguage(filePath),
        contentHash,
        projectId,
        repoRoot,
      }));
    }
    return chunks;
  }

  // Track covered line ranges to find module-level gaps
  const coveredLines = new Set<number>();

  for (const def of definitions) {
    const startLine = Math.max(0, def.line - 1);
    const endLineIdx = Math.max(startLine, def.endLine - 1);
    const symbolLines = lines.slice(startLine, endLineIdx + 1);
    const symbolText = symbolLines.join("\n");

    for (let l = startLine; l <= endLineIdx; l++) coveredLines.add(l);

    const symbolName = def.name || "anonymous";
    const chunkType = def.type || "symbol"; // function, class, method, etc.

    // Split oversized symbols
    if (symbolText.length <= opts.maxChunkChars) {
      chunks.push(buildChunk({
        content: symbolText,
        symbolName,
        chunkType,
        startLine,
        endLine: endLineIdx,
        filePath,
        filename,
        language: detectLanguage(filePath),
        contentHash,
        projectId,
        repoRoot,
      }));
    } else {
      const parts = splitBySize(symbolText, opts.maxChunkChars, opts.overlapChars);
      for (let i = 0; i < parts.length; i++) {
        chunks.push(buildChunk({
          content: parts[i].text,
          symbolName: `${symbolName}[${i + 1}/${parts.length}]`,
          chunkType,
          startLine: startLine + parts[i].startLine,
          endLine: startLine + parts[i].endLine,
          filePath,
          filename,
          language: detectLanguage(filePath),
          contentHash,
          projectId,
          repoRoot,
        }));
      }
    }
  }

  // Module-level code (imports, constants, top-level statements between symbols)
  if (opts.includeModuleLevel) {
    const moduleLines: string[] = [];
    let moduleStart = 0;
    let inGap = false;

    for (let i = 0; i < lines.length; i++) {
      if (!coveredLines.has(i)) {
        if (!inGap) { moduleStart = i; inGap = true; }
        moduleLines.push(lines[i]);
      } else if (inGap) {
        const text = moduleLines.join("\n").trim();
        if (text.length > 20) { // Skip trivial whitespace gaps
          chunks.push(buildChunk({
            content: text,
            symbolName: "module_level",
            chunkType: "module",
            startLine: moduleStart,
            endLine: i - 1,
            filePath,
            filename,
            language: detectLanguage(filePath),
            contentHash,
            projectId,
            repoRoot,
          }));
        }
        moduleLines.length = 0;
        inGap = false;
      }
    }

    // Trailing module-level code
    if (moduleLines.length > 0) {
      const text = moduleLines.join("\n").trim();
      if (text.length > 20) {
        chunks.push(buildChunk({
          content: text,
          symbolName: "module_level",
          chunkType: "module",
          startLine: moduleStart,
          endLine: lines.length - 1,
          filePath,
          filename,
          language: detectLanguage(filePath),
          contentHash,
          projectId,
          repoRoot,
        }));
      }
    }
  }

  return chunks;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════════

interface BuildChunkParams {
  content: string;
  symbolName: string;
  chunkType: string;
  startLine: number;
  endLine: number;
  filePath: string;
  filename: string;
  language: string;
  contentHash: string;
  projectId: string;
  repoRoot: string;
}

function buildChunk(params: BuildChunkParams): ChunkResult {
  const { content, symbolName, chunkType, startLine, endLine, filePath, filename, language, contentHash, projectId, repoRoot } = params;

  // Semantic content: identical to raw content per Python parity.
  // See cli/chunker_ast.py:923 — `semantic = chunk_text`. The semantic
  // text is fed to the embedding model unchanged from the raw body.
  const semanticContent = content;

  // Lexical content: Python-parity header for BMX/BM25 indexing.
  // See cli/chunker_ast.py:880-933 (build_chunk_representations).
  // Format: "# File: <relPath> | Symbol: <symbolName> | Language: <language>\n<content>"
  // Pieces are dropped from the join when absent. If NO pieces are present
  // at all, the lexical text is just the raw content (no header line).
  // This format MUST match the Python baseline exactly so that embeddings
  // and BMX+ tokenization stay compatible across the two implementations.
  const relPath = relative(repoRoot, filePath);
  const headerParts: string[] = [];
  if (relPath) headerParts.push(`File: ${relPath}`);
  if (symbolName) headerParts.push(`Symbol: ${symbolName}`);
  if (language) headerParts.push(`Language: ${language}`);
  const lexicalContent = headerParts.length > 0
    ? `# ${headerParts.join(" | ")}\n${content}`
    : content;

  return {
    content,
    semanticContent,
    lexicalContent,
    metadata: {
      doc_path: relPath,
      filename,
      language,
      content_type: contentTypeForLanguage(language),
      symbol_name: symbolName,
      chunk_type: chunkType,
      start_line: startLine + 1,
      end_line: endLine + 1,
      chunk_chars: content.length,
      content_hash: contentHash,
      project_id: projectId,
      repo_root: repoRoot,
    },
  };
}

interface SplitPart {
  text: string;
  startLine: number;
  endLine: number;
}

/**
 * Split text into size-limited parts with line-boundary-aware overlap.
 */
function splitBySize(text: string, maxChars: number, overlapChars: number): SplitPart[] {
  if (text.length <= maxChars) {
    const lineCount = text.split("\n").length;
    return [{ text, startLine: 0, endLine: lineCount - 1 }];
  }

  const lines = text.split("\n");
  const parts: SplitPart[] = [];
  let currentLines: string[] = [];
  let currentChars = 0;
  let partStartLine = 0;

  for (let i = 0; i < lines.length; i++) {
    const lineLen = lines[i].length + 1; // +1 for newline

    if (currentChars + lineLen > maxChars && currentLines.length > 0) {
      parts.push({
        text: currentLines.join("\n"),
        startLine: partStartLine,
        endLine: partStartLine + currentLines.length - 1,
      });

      // Overlap: walk back to include overlapChars worth of context
      let overlapLines = 0;
      let overlapSize = 0;
      for (let j = currentLines.length - 1; j >= 0 && overlapSize < overlapChars; j--) {
        overlapSize += currentLines[j].length + 1;
        overlapLines++;
      }

      const kept = currentLines.slice(currentLines.length - overlapLines);
      partStartLine = i - overlapLines;
      currentLines = [...kept];
      currentChars = kept.reduce((sum, l) => sum + l.length + 1, 0);
    }

    currentLines.push(lines[i]);
    currentChars += lineLen;
  }

  if (currentLines.length > 0) {
    parts.push({
      text: currentLines.join("\n"),
      startLine: partStartLine,
      endLine: partStartLine + currentLines.length - 1,
    });
  }

  return parts;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Language Detection (simple, tree-sitter handles the real parsing)
// ═══════════════════════════════════════════════════════════════════════════════

const EXTENSION_MAP: Record<string, string> = {
  ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "jsx",
  ".py": "python", ".rs": "rust", ".go": "go", ".java": "java",
  ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
  ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
  ".cs": "csharp", ".scala": "scala", ".lua": "lua", ".zig": "zig",
  ".sh": "bash", ".bash": "bash", ".zsh": "bash",
  ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
  ".toml": "toml", ".html": "html", ".css": "css", ".sql": "sql",
};

/**
 * Languages that represent documentation rather than executable code.
 * Used by contentTypeForLanguage() to tag chunks with the correct
 * content_type so the "docs" filter in rag-search works.
 */
const MARKDOWN_LANGUAGES = new Set(["markdown"]);
const TEXT_LANGUAGES = new Set(["text", "rst", "asciidoc"]);

/**
 * Map a detected language to the appropriate content_type metadata value.
 *   - markdown        → "markdown"
 *   - text/rst/etc.   → "text"
 *   - everything else → "code"
 */
function contentTypeForLanguage(language: string): "code" | "markdown" | "text" {
  if (MARKDOWN_LANGUAGES.has(language)) return "markdown";
  if (TEXT_LANGUAGES.has(language)) return "text";
  return "code";
}

function detectLanguage(filePath: string): string {
  const ext = filePath.slice(filePath.lastIndexOf(".")).toLowerCase();
  return EXTENSION_MAP[ext] || "unknown";
}

