# File Retrieval GUI

A powerful file management tool for flexible and scalable filesystem interactions.

![File Retrieval GUI](assets/gui.png)

## Features

- 🤖 **File Discovery** - Smart code location discovery, works with most raw llm responses
- 📋 **Multi-Selection Interface** - Drag & drop reordering with clipboard integration  
- 🔍 **Preview Changes** - GitHub-style diff viewer before applying modifications
- 🔌 **WebSocket Server** - Real-time broadcasting to connect external tools
- 🎯 **Smart Filtering** - Regex-based exclusions and search

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e core/ -e shared/

# Run GUI
python main.py .

# Run CLI mode (reads from clipboard)
python main.py -m .
```

## Usage
Launch the interface to browse, select, and manage files with:
```bash
python main.py
```
- Use **Smart Paster** to select files based on all paths included in a text
- Use **Apply Changes** to modify files with preview-before-apply functionality
- WebSocket server runs on port 8765 for external tool integration (![see tools/getuserfiles.py](tools/getuserfiles.py))

## License

![MIT](LICENSE)