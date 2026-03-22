"""Agent's notebook tool.

Provides a file-backed scratchpad for the agent to store found file paths
or important notes between steps or queries.
"""

import os
from pathlib import Path
import tempfile
from typing import List, Callable, Optional

try:
    from _shared.agent import BaseTool, ToolCap
except ImportError:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))
    from _shared.agent import BaseTool, ToolCap


class AgentsNotebookTool(BaseTool):
    """File-backed scratchpad for the agent.
    
    XML format:
        <notebook>
            <action>add</action>  <!-- list, add, remove, clear -->
            <entry>src/main.py</entry> <!-- required for add/remove -->
        </notebook>
    """
    name = "notebook"
    description = "A persistent scratchpad to save file paths or notes. Operations: list, add, remove, clear."
    example = "<notebook><action>add</action><entry>path/to/interesting_file.py</entry></notebook>"
    aliases: List[str] = ["scratchpad"]
    capabilities = {ToolCap.SANDBOXED}

    def __init__(self, on_change: Optional[Callable[[], None]] = None):
        super().__init__()
        self.on_change = on_change
        cache_dir = Path.home() / ".cache" / "file_retrieval_gui"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = cache_dir / "agents_notebook.txt"
        if not self.filepath.exists():
            self._write_entries([])

    def get_entries(self) -> List[str]:
        """Read all entries from the notebook."""
        if not self.filepath.exists():
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return []

    def _write_entries(self, entries: List[str]) -> None:
        """Atomically write entries to the notebook."""
        # Clean and deduplicate entries loosely
        cleaned = []
        seen = set()
        for e in entries:
            c = e.strip()
            if c and c not in seen:
                seen.add(c)
                cleaned.append(c)

        try:
            fd, tmp_path = tempfile.mkstemp(dir=self.filepath.parent, text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for entry in cleaned:
                    f.write(f"{entry}\n")
            os.replace(tmp_path, self.filepath)
            
            if self.on_change:
                self.on_change()
        except Exception as e:
            print(f"Error writing to notebook: {e}")

    def run(self, content: str) -> str:
        action = self.parse_param(content, "action")
        entry = self.parse_param(content, "entry")

        if not action:
            # Fallback format: Just list if empty? Or just the action as content
            action_text = content.strip().lower()
            if action_text in ["list", "clear"]:
                action = action_text
            else:
                return "Error: <action> must be 'list', 'add', 'remove', or 'clear'."

        action = action.lower()
        
        if action == "list":
            entries = self.get_entries()
            if not entries:
                return "Notebook is empty."
            return "Notebook entries:\n" + "\n".join(f"- {e}" for e in entries)
            
        elif action == "clear":
            self._write_entries([])
            return "Notebook cleared."
            
        elif action == "add":
            if not entry:
                return "Error: <entry> is required for 'add' action."
            entries = self.get_entries()
            if entry in entries:
                return f"Entry already exists: {entry}"
            entries.append(entry)
            self._write_entries(entries)
            return f"Added entry: {entry}"
            
        elif action == "remove":
            if not entry:
                return "Error: <entry> is required for 'remove' action."
            entries = self.get_entries()
            if entry not in entries:
                return f"Entry not found: {entry}"
            entries.remove(entry)
            self._write_entries(entries)
            return f"Removed entry: {entry}"
            
        else:
            return f"Error: Unknown action '{action}'. Use list, add, remove, or clear."
