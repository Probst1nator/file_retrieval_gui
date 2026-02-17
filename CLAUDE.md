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
```

## Architecture

This is a Tkinter-based GUI application for file management with WebSocket integration.

### Core Components

- **main.py** - Entry point, handles CLI args and launches either GUI or CLI mode
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

Chat interface with Ollama integration for AI-assisted file discovery:
- **ollama_client.py** - HTTP client for Ollama REST API (list models, streaming chat)
- **ChatMessage** dataclass - Stores chat history with role, content, timestamp, extracted files
- **Tool checkboxes** - Auto-detected from `agent/main.py:INTERRUPTING_TAGS`, dynamically enable/disable
- **File extraction** - Parses `[FILES: path1, path2]` from agent responses, auto-adds to selection

**WebSocket Message Protocol** for external integration:
```json
{"type": "agentic_search_query", "query": "find all API endpoints"}
{"type": "agentic_search_response", "response": "Found these files... [FILES: api.py]"}
{"type": "get_status"}  // Returns ollama status, model, enabled tools
```

## File Formats

**Clipboard output format** (used by `build_clipboard_content`):
```
# relative/path/to/file.ext
```language
file contents
```
```

**Apply changes input formats** (supports multiple formats, tried in priority order):

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
- File extension required in all formats
- Duplicate detection prevents applying same code block multiple times
- Absolute paths and paths with `..` are rejected for security
