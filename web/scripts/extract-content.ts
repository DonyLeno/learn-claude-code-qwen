import * as fs from "fs";
import * as path from "path";
import type {
  AgentVersion,
  VersionDiff,
  DocContent,
  VersionIndex,
} from "../src/types/agent-data";
import { VERSION_META, VERSION_ORDER, LEARNING_PATH } from "../src/lib/constants";

// Resolve paths relative to this script's location (web/scripts/)
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const AGENTS_DIR = path.join(REPO_ROOT, "agents");
const DOCS_DIR = path.join(REPO_ROOT, "docs");
const OUT_DIR = path.join(WEB_DIR, "src", "data", "generated");

// Map python filenames to version IDs
// s01_agent_loop.py -> s01
// s02_tools.py -> s02
// s_full.py -> s_full (reference agent, typically skipped)
function filenameToVersionId(filename: string): string | null {
  const base = path.basename(filename, ".py");
  if (base === "s_full") return "s_full";
  if (base === "__init__") return null;

  const match = base.match(/^(s\d+[a-c]?)_/);
  if (!match) return null;
  return match[1];
}

// Extract classes from Python source
function extractClasses(
  lines: string[]
): { name: string; startLine: number; endLine: number }[] {
  const classes: { name: string; startLine: number; endLine: number }[] = [];
  const classPattern = /^class\s+(\w+)/;

  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(classPattern);
    if (m) {
      const name = m[1];
      const startLine = i + 1;
      // Find end of class: next class/function at indent 0, or EOF
      let endLine = lines.length;
      for (let j = i + 1; j < lines.length; j++) {
        if (
          lines[j].match(/^class\s/) ||
          lines[j].match(/^def\s/) ||
          (lines[j].match(/^\S/) && lines[j].trim() !== "" && !lines[j].startsWith("#") && !lines[j].startsWith("@"))
        ) {
          endLine = j;
          break;
        }
      }
      classes.push({ name, startLine, endLine });
    }
  }
  return classes;
}

// Extract top-level functions from Python source
function extractFunctions(
  lines: string[]
): { name: string; signature: string; startLine: number }[] {
  const functions: { name: string; signature: string; startLine: number }[] = [];
  const funcPattern = /^def\s+(\w+)\((.*?)\)/;

  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(funcPattern);
    if (m) {
      functions.push({
        name: m[1],
        signature: `def ${m[1]}(${m[2]})`,
        startLine: i + 1,
      });
    }
  }
  return functions;
}

// Extract tool names from Python source
// Looks for "name": "tool_name" patterns in dict literals
function extractTools(source: string): string[] {
  const toolPattern = /"name"\s*:\s*"(\w+)"/g;
  const tools = new Set<string>();
  let m;
  while ((m = toolPattern.exec(source)) !== null) {
    tools.add(m[1]);
  }
  return Array.from(tools);
}

// Count non-blank, non-comment lines
function countLoc(lines: string[]): number {
  return lines.filter((line) => {
    const trimmed = line.trim();
    return trimmed !== "" && !trimmed.startsWith("#");
  }).length;
}

// Detect locale from subdirectory path
// docs/en/s01-the-agent-loop.md -> "en"
// docs/zh/s01-the-agent-loop.md -> "zh"
// docs/ja/s01-the-agent-loop.md -> "ja"
function detectLocale(relPath: string): "en" | "zh" | "ja" {
  if (relPath.startsWith("zh/") || relPath.startsWith("zh\\")) return "zh";
  if (relPath.startsWith("ja/") || relPath.startsWith("ja\\")) return "ja";
  return "en";
}

// Extract version from doc filename (e.g., "s01-the-agent-loop.md" -> "s01")
function extractDocVersion(filename: string): string | null {
  const m = filename.match(/^(s\d+[a-c]?|s_full)-/);
  return m ? m[1] : null;
}

function extractTopLevelBlock(lines: string[], startIndex: number): string {
  let end = lines.length;
  for (let i = startIndex + 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "") continue;
    if (!line.startsWith(" ") && !line.startsWith("\t")) {
      end = i;
      break;
    }
  }
  return lines.slice(startIndex, end).join("\n").trimEnd();
}

function extractAssignmentBlock(lines: string[], varName: string): string | null {
  const start = lines.findIndex((line) =>
    new RegExp(`^${varName}\\s*=`).test(line)
  );
  if (start < 0) return null;
  let balance = 0;
  let seen = false;
  for (let i = start; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === "{" || ch === "[") {
        balance += 1;
        seen = true;
      } else if (ch === "}" || ch === "]") {
        balance -= 1;
      }
    }
    if (seen && balance <= 0) {
      return lines.slice(start, i + 1).join("\n").trimEnd();
    }
  }
  return null;
}

function extractAround(
  lines: string[],
  needle: string,
  before: number,
  after: number
): string | null {
  const idx = lines.findIndex((line) => line.includes(needle));
  if (idx < 0) return null;
  const start = Math.max(0, idx - before);
  const end = Math.min(lines.length, idx + after + 1);
  return lines.slice(start, end).join("\n").trimEnd();
}

function collectTopLevelBlocks(lines: string[]): string[] {
  const starts: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^(class|def)\s+/.test(line)) starts.push(i);
    if (/^(TOOLS|TOOL_HANDLERS|CHILD_TOOLS|PARENT_TOOLS)\s*=/.test(line)) starts.push(i);
  }
  return starts.map((i) => extractTopLevelBlock(lines, i));
}

function pickBestMatchingBlock(block: string, lines: string[]): string | null {
  const candidates = collectTopLevelBlocks(lines);
  if (!candidates.length) return null;
  const tokens = Array.from(
    new Set((block.match(/[A-Za-z_]\w+/g) ?? []).filter((w) => w.length >= 4))
  );
  if (!tokens.length) return null;
  let best: string | null = null;
  let bestScore = 0;
  for (const candidate of candidates) {
    let score = 0;
    for (const t of tokens) {
      if (candidate.includes(t)) score += 1;
    }
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return bestScore > 0 ? best : null;
}

function syncPythonFence(block: string, source: string): string {
  const lines = source.split("\n");
  const classMatch = block.match(/class\s+([A-Za-z_]\w*)/);
  if (classMatch) {
    const i = lines.findIndex((line) => line.match(new RegExp(`^class\\s+${classMatch[1]}\\b`)));
    if (i >= 0) return extractTopLevelBlock(lines, i);
  }
  const funcMatch = block.match(/def\s+([A-Za-z_]\w*)\(/);
  if (funcMatch) {
    const i = lines.findIndex((line) => line.match(new RegExp(`^def\\s+${funcMatch[1]}\\(`)));
    if (i >= 0) return extractTopLevelBlock(lines, i);
  }
  if (block.includes("TOOL_HANDLERS")) {
    const snippet = extractAssignmentBlock(lines, "TOOL_HANDLERS");
    if (snippet) return snippet;
  }
  if (block.includes("TOOLS =")) {
    const snippet = extractAssignmentBlock(lines, "TOOLS");
    if (snippet) return snippet;
  }
  if (block.includes("PARENT_TOOLS")) {
    const snippet = extractAssignmentBlock(lines, "PARENT_TOOLS");
    if (snippet) return snippet;
  }
  if (block.includes("CHILD_TOOLS")) {
    const snippet = extractAssignmentBlock(lines, "CHILD_TOOLS");
    if (snippet) return snippet;
  }
  if (block.includes("rounds_since_todo")) {
    const idx = lines.findIndex((line) => line.includes("rounds_since_todo = 0 if used_todo"));
    if (idx >= 0) {
      const end = lines.findIndex(
        (line, i) => i > idx && line.includes('messages.append({"role": "user", "content": results})')
      );
      if (end >= idx) return lines.slice(idx, end + 1).join("\n").trimEnd();
    }
  }
  if (block.includes("client.messages.create")) {
    const snippet = extractAround(lines, "response = client.messages.create(", 2, 4);
    if (snippet) return snippet;
  }
  const best = pickBestMatchingBlock(block, lines);
  if (best) return best;
  return block.trimEnd();
}

function syncDocCodeBlocks(content: string, source: string | undefined): string {
  if (!source) return content;
  return content.replace(/```python\n([\s\S]*?)```/g, (_m, code) => {
    const synced = syncPythonFence(code, source);
    return `\`\`\`python\n${synced}\n\`\`\``;
  });
}

// Main extraction
function main() {
  console.log("Extracting content from agents and docs...");
  console.log(`  Repo root: ${REPO_ROOT}`);
  console.log(`  Agents dir: ${AGENTS_DIR}`);
  console.log(`  Docs dir: ${DOCS_DIR}`);

  // Skip extraction if source directories don't exist (e.g. Vercel build).
  // Pre-committed generated data will be used instead.
  if (!fs.existsSync(AGENTS_DIR)) {
    console.log("  Agents directory not found, skipping extraction.");
    console.log("  Using pre-committed generated data.");
    return;
  }

  // 1. Read all agent files
  const agentFiles = fs
    .readdirSync(AGENTS_DIR)
    .filter((f) => f.startsWith("s") && f.endsWith(".py"));

  console.log(`  Found ${agentFiles.length} agent files`);

  const versions: AgentVersion[] = [];

  for (const filename of agentFiles) {
    const versionId = filenameToVersionId(filename);
    if (!versionId) {
      console.warn(`  Skipping ${filename}: could not determine version ID`);
      continue;
    }

    const filePath = path.join(AGENTS_DIR, filename);
    const source = fs.readFileSync(filePath, "utf-8");
    const lines = source.split("\n");

    const meta = VERSION_META[versionId];
    const classes = extractClasses(lines);
    const functions = extractFunctions(lines);
    const tools = extractTools(source);
    const loc = countLoc(lines);

    versions.push({
      id: versionId,
      filename,
      title: meta?.title ?? versionId,
      subtitle: meta?.subtitle ?? "",
      loc,
      tools,
      newTools: [], // computed after all versions are loaded
      coreAddition: meta?.coreAddition ?? "",
      keyInsight: meta?.keyInsight ?? "",
      classes,
      functions,
      layer: meta?.layer ?? "tools",
      source,
    });
  }

  // Sort versions according to VERSION_ORDER
  const orderMap = new Map(VERSION_ORDER.map((v, i) => [v, i]));
  versions.sort(
    (a, b) => (orderMap.get(a.id as any) ?? 99) - (orderMap.get(b.id as any) ?? 99)
  );

  // 2. Compute newTools for each version
  for (let i = 0; i < versions.length; i++) {
    const prev = i > 0 ? new Set(versions[i - 1].tools) : new Set<string>();
    versions[i].newTools = versions[i].tools.filter((t) => !prev.has(t));
  }

  // 3. Compute diffs between adjacent versions in LEARNING_PATH
  const diffs: VersionDiff[] = [];
  const versionMap = new Map(versions.map((v) => [v.id, v]));

  for (let i = 1; i < LEARNING_PATH.length; i++) {
    const fromId = LEARNING_PATH[i - 1];
    const toId = LEARNING_PATH[i];
    const fromVer = versionMap.get(fromId);
    const toVer = versionMap.get(toId);

    if (!fromVer || !toVer) continue;

    const fromClassNames = new Set(fromVer.classes.map((c) => c.name));
    const fromFuncNames = new Set(fromVer.functions.map((f) => f.name));
    const fromToolNames = new Set(fromVer.tools);

    diffs.push({
      from: fromId,
      to: toId,
      newClasses: toVer.classes
        .map((c) => c.name)
        .filter((n) => !fromClassNames.has(n)),
      newFunctions: toVer.functions
        .map((f) => f.name)
        .filter((n) => !fromFuncNames.has(n)),
      newTools: toVer.tools.filter((t) => !fromToolNames.has(t)),
      locDelta: toVer.loc - fromVer.loc,
    });
  }

  // 4. Read doc files from locale subdirectories (en/, zh/, ja/)
  const docs: DocContent[] = [];

  if (fs.existsSync(DOCS_DIR)) {
    const localeDirs = ["en", "zh", "ja"];
    let totalDocFiles = 0;

    for (const locale of localeDirs) {
      const localeDir = path.join(DOCS_DIR, locale);
      if (!fs.existsSync(localeDir)) continue;

      const docFiles = fs
        .readdirSync(localeDir)
        .filter((f) => f.endsWith(".md"));

      totalDocFiles += docFiles.length;

      for (const filename of docFiles) {
        const version = extractDocVersion(filename);
        if (!version) {
          console.warn(`  Skipping doc ${locale}/${filename}: could not determine version`);
          continue;
        }

        const filePath = path.join(localeDir, filename);
        const rawContent = fs.readFileSync(filePath, "utf-8");
        const syncedContent = syncDocCodeBlocks(rawContent, versionMap.get(version)?.source);

        const titleMatch = syncedContent.match(/^#\s+(.+)$/m);
        const title = titleMatch ? titleMatch[1] : filename;

        docs.push({ version, locale: locale as "en" | "zh" | "ja", title, content: syncedContent });
      }
    }

    console.log(`  Found ${totalDocFiles} doc files across ${localeDirs.length} locales`);
  } else {
    console.warn(`  Docs directory not found: ${DOCS_DIR}`);
  }

  // 5. Write output
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const index: VersionIndex = { versions, diffs };
  const indexPath = path.join(OUT_DIR, "versions.json");
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  console.log(`  Wrote ${indexPath}`);

  const docsPath = path.join(OUT_DIR, "docs.json");
  fs.writeFileSync(docsPath, JSON.stringify(docs, null, 2));
  console.log(`  Wrote ${docsPath}`);

  // Summary
  console.log("\nExtraction complete:");
  console.log(`  ${versions.length} versions`);
  console.log(`  ${diffs.length} diffs`);
  console.log(`  ${docs.length} docs`);
  for (const v of versions) {
    console.log(
      `    ${v.id}: ${v.loc} LOC, ${v.tools.length} tools, ${v.classes.length} classes, ${v.functions.length} functions`
    );
  }
}

main();
