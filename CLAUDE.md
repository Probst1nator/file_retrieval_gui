# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI mode
python main.py [directory]

# Run CLI mode (reads from clipboard, processes files, copies result to clipboard)
python main.py -m [directory]

# Launch GUI with an agentic search query (opens Agentic Search tab, auto-submits)
python main.py [directory] -a "find all files that handle authentication"
```

## Architecture

This is a Tkinter-based GUI application for file management with WebSocket integration.

### Core Components

- **main.py** - Entry point, handles CLI args (`-m` CLI mode, `-a QUERY` agentic query mode) and launches either GUI or CLI mode
- **gui.py** - Main `FileCopierApp` class with the Tkinter interface, WebSocket server, and all UI logic
- **smart_paster.py** - File discovery, path resolution, clipboard content building, and file change preview/apply logic

### Key Patterns

**File Selection Flow**: Files are selected via tree view (double-click or drag) → stored in `selected_files_map` → built into formatted clipboard content via `build_clipboard_content()` → broadcasted to WebSocket clients

**WebSocket Server**: Runs on ports 8765-8769 (auto-discovers available port). Broadcasts file selection updates to connected clients. Config stored in `.file_copier_config.json`.

**Smart Paster**: Extracts file paths from text using prioritized regex patterns (file references → quoted paths → absolute → relative → filenames), resolves against project files.

**Apply Changes**: Parses markdown code blocks with file paths (e.g., `# path/to/file.ext` followed by code block), supports preview with GitHub-style diffs before applying.

### tools/getuserfiles.py

External tool that connects to the GUI's WebSocket server to retrieve current file selection. Uses multi-tier discovery: environment vars → user config → GUI config → port scanning.

### Agentic Search Tab

Chat interface with agentic capabilities for codebase exploration:
- **explorer_agent.py** - `ExplorerAgentLoop` orchestrates `AgenticLoop`, `ToolDiscovery`, and `LLMRouter`. Changes cwd to project dir during queries so tools resolve relative paths correctly. Truncates LLM output after first XML tool call to prevent hallucination cascades from weaker models. System prompt passed via `system=` kwarg (not as a message) to avoid being silently dropped by Gemini's message converter.
- **LLMRouter** (`_shared/agent/llm_router.py`) - Multi-provider LLM routing with fallback chaining (Gemini -> Ollama -> llama.cpp). Model selected from `.env` config.
- **ToolCap filtering** - Auto-discovers XML-based tools from `_shared/agent_tools/` and `agent_tools/` filtered by `ToolCap` enum capabilities (allows `READ` + `SANDBOXED` only, blocking `WRITE`/`EXECUTE`/`NETWORK`). Common tools: `grep`, `filesystem`, `readfile`, `file_search`.
- **AgentsNotebookTool** - Persistent scratchpad (`~/.cache/file_retrieval_gui/agents_notebook.txt`) that agents use to save file paths during exploration. Displayed in the UI Notebook Panel.
- **File extraction** - Parses `[FILES: path1, path2]` from final agent responses to auto-select files.
- **Context logging** - Each session writes a JSONL log to `~/.cache/file_retrieval_gui/agent_logs/session_YYYYMMDD_HHMMSS.jsonl`. Events: `query_start` (system prompt, tools), `llm_response` (model, full text), `tool_result` (tool name, output preview), `query_complete` (full message context). "Copy agent log" button in the header copies the current session log to clipboard.

**WebSocket Message Protocol** for external integration:
```json
{"type": "agentic_search_query", "query": "find all API endpoints"}
{"type": "agentic_search_response", "response": "Found these files... [FILES: api.py]"}
{"type": "get_status"}  // Returns connection status, model, enabled tools
```

## File Formats

**Clipboard output format** (used by `build_clipboard_content`):
```
# relative/path/to/file.ext
```language
file contents
```
```

**Apply changes input formats.** `preview_changes_to_files` consumes `<patch>` blocks
first, then the three markdown forms in order. A path claimed by a `<patch>` block is
never re-matched by a markdown pattern.

0. `<patch>` — surgical search/replace, or a full overwrite:
```markdown
<patch file="path/to/file.ext">
<search>
exact text to find
</search>
<replace>
text to put there
</replace>
</patch>
```
A block with no `<search>` is a full overwrite (body, or a lone `<replace>` block).

1. **File:** annotation (most explicit):
```markdown
**File:** `path/to/file.ext`

```python
new file content
```
```

2. Heading immediately followed by code block:
```markdown
# path/to/file.ext
```python
new file content
```
```

3. Code block with path inside (fallback):
```markdown
```python
# path/to/file.ext
new file content
```
```

**Notes:**
- File extension required in all markdown formats (the `<patch>` `file=` attribute is
  taken as given).
- **The opening fence must follow the path marker immediately** — only blank lines may
  intervene. The patterns are compiled with `re.DOTALL` for the content group, so before
  this was enforced a heading merely *mentioning* a filename captured the next unrelated
  code block paragraphs later.
- **Several `<patch>` blocks naming the same file chain onto each other** — block 2 is
  applied to block 1's result and they collapse into one `FileChange` with one combined
  diff. If any block's `<search>` fails to match, the whole file is marked invalid and
  nothing is written for it (all-or-nothing, per `apply_text_patch`).
- The markdown patterns dedup by path: the *first* block for a path wins, later ones are
  dropped.
- Absolute paths and paths with `..` are rejected for security.
- Changes that fail to parse become `ChangeType.INVALID_PATH`, carry the reason in
  `error_message`, and are created **unselected**. `status_summary` surfaces that reason
  (collapsed to one line for the tree; the full text is in the change's preview tab), and
  `apply_selected_changes` reports a selected-but-invalid change as an error rather than
  skipping it silently.

**Undo/redo.** Both apply paths record an `ApplyOperation` in `ApplyHistoryManager`,
which is keyed **by project root**. `apply_selected_with_history(changes, root_directory)`
therefore takes the root explicitly — it must be the same string the GUI later passes to
`can_undo()`/`can_redo()` (`self.directory`), or the operation is filed under a key
nobody queries and Undo silently stays disabled.

Tests for all of the above: `test_apply_changes.py`.
