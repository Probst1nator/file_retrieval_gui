"""Explorer Agent Loop.

Wires ToolDiscovery + AgenticLoop + LLMRouter together to provide
an agentic search capability for the file retrieval GUI.
"""

import os
import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Dict, Optional, List

try:
    from _shared.agent.tool_discovery import ToolDiscovery
    from _shared.agent.xml_tools import AgenticLoop, ToolCap
    from _shared.agent.llm_router import LLMRouter
except ImportError:
    import sys
    # Add parent of tools dir to path assuming tools is the repo root
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from _shared.agent.tool_discovery import ToolDiscovery
    from _shared.agent.xml_tools import AgenticLoop, ToolCap
    from _shared.agent.llm_router import LLMRouter

try:
    from agent_tools.notebook import AgentsNotebookTool
except ImportError:
    from .agent_tools.notebook import AgentsNotebookTool


LOG_DIR = Path.home() / ".cache" / "file_retrieval_gui" / "agent_logs"

logger = logging.getLogger("explorer_agent")


class ExplorerAgentLoop:
    """Agentic search loop replacing direct Ollama usage.

    Uses ToolDiscovery to find tools, AgenticLoop to orchestrate,
    and LLMRouter for reliable model generation.

    Maintains a JSONL context log at ~/.cache/file_retrieval_gui/agent_logs/.
    Each query creates a log entry with: system prompt, messages, tool results,
    and final response.
    """

    def __init__(
        self,
        project_dir: str,
        on_tool_result: Optional[Callable[[str, Any], None]] = None,
        on_notebook_change: Optional[Callable[[], None]] = None
    ):
        self.project_dir = Path(project_dir)
        self.on_tool_result = on_tool_result
        self.on_notebook_change = on_notebook_change

        # Setup logging
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        
        # Discover tools
        discovery_paths = [
            str(Path(__file__).parent.parent.parent / "_shared" / "agent_tools"),
            str(Path(__file__).parent / "agent_tools")
        ]
        
        self.discovery = ToolDiscovery(
            discovery_paths,
            allow_capabilities={ToolCap.READ, ToolCap.SANDBOXED},
        )
        
        self.available_tools_dict = self.discovery.discover()
        
        # Deduplicate tools since discovery might return aliases
        self.tools = []
        seen = set()
        for t in self.available_tools_dict.values():
            if id(t) not in seen:
                self.tools.append(t)
                seen.add(id(t))
        
        # Add notebook explicitly if not discovered
        if not any(t.name == "notebook" for t in self.tools):
            self.notebook_tool = AgentsNotebookTool(on_change=on_notebook_change)
            self.tools.append(self.notebook_tool)
        else:
            # Set callback on existing
            for t in self.tools:
                if t.name == "notebook":
                    t.on_change = on_notebook_change
                    self.notebook_tool = t
        
        # Initialize LLM Router
        self.router = LLMRouter()
        
        # Find CLAUDE.md files
        self.claude_mds = self._find_claude_mds()

    def _find_claude_mds(self) -> List[str]:
        """Find all CLAUDE.md files in the project."""
        pattern = str(self.project_dir / "**" / "CLAUDE.md")
        files = glob.glob(pattern, recursive=True)
        return [os.path.relpath(f, self.project_dir) for f in files]

    def _build_system_prompt(self, tools: List[Any]) -> str:
        """Create the system prompt including tools and CLAUDE.md info."""
        tool_docs = []
        for t in tools:
            doc = f"## Tool: {t.name}\n"
            if t.aliases:
                doc += f"Aliases: {', '.join(t.aliases)}\n"
            doc += f"Description: {t.description}\n"
            if hasattr(t, 'example') and t.example:
                doc += f"Example: {t.example}\n"
            tool_docs.append(doc)
            
        tools_str = "\n".join(tool_docs)
        
        claude_md_str = ""
        if self.claude_mds:
            claude_md_str = "\nProject Documentation (read these if relevant):\n"
            for cmd in self.claude_mds:
                claude_md_str += f"- {cmd}\n"
                
        return f"""You are an autonomous file-exploration agent with direct access to the user's local codebase via XML tools.
You MUST use your tools to answer questions — you have DIRECT filesystem access.

CRITICAL RULES:
- Output EXACTLY ONE xml tool call per message and NOTHING ELSE. No explanation, no text before or after.
- NEVER fabricate tool output. NEVER write <context> tags yourself — the system adds those automatically.
- NEVER hallucinate file contents or directory listings — always use a tool to check.
- After each tool call, wait for the real result before proceeding.
- When you are done and ready to answer the user, respond with plain text only (no XML).
- If your final answer references files, end with: [FILES: path1, path2]

The project root is: {self.project_dir}
Use absolute paths or paths relative to the project root.

{tools_str}
{claude_md_str}"""

    def _log_context(self, entry: Dict[str, Any]) -> None:
        """Append a JSON entry to the session log file."""
        try:
            entry["timestamp"] = datetime.now().isoformat()
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write agent log: %s", e)

    def run_query(self, query: str, active_tool_names: List[str], max_iterations: int = 10, print_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Run an agentic search query.

        Args:
            query: The user's query
            active_tool_names: List of tool names to enable
            max_iterations: Maximum loop iterations
            print_callback: Optional callback for streaming/status updates

        Returns:
            Dict with 'response', 'tool_results', 'iterations'
        """
        from _shared.agent.xml_tools import ToolResult

        active_tools = [t for t in self.tools if t.name in active_tool_names]
        system_prompt = self._build_system_prompt(active_tools)

        # Log query start
        self._log_context({
            "event": "query_start",
            "query": query,
            "active_tools": active_tool_names,
            "max_iterations": max_iterations,
            "system_prompt": system_prompt,
        })

        # Track messages for logging
        messages_log: List[Dict[str, str]] = []

        # Build list of tool tag names for truncation
        all_tool_names = []
        for t in active_tools:
            all_tool_names.append(t.name)
            for alias in getattr(t, 'aliases', []):
                all_tool_names.append(alias)

        def _truncate_after_first_tool_call(text: str) -> str:
            """Truncate LLM output after the first complete tool call.

            Prevents models from hallucinating fake tool results and
            multi-turn conversations in a single response.
            """
            import re as _re
            for tname in all_tool_names:
                close_tag = f"</{_re.escape(tname)}>"
                match = _re.search(close_tag, text)
                if match:
                    return text[:match.end()]
            return text

        def custom_generate(msgs: List[Dict[str, str]]) -> str:
            # Strip any system messages from the list — pass via system= kwarg instead
            chat_msgs = [m for m in msgs if m.get("role") != "system"]
            result = self.router.generate(messages=chat_msgs, stream=False, system=system_prompt)
            # LLMRouter may return str, Iterator, or None
            if result is None:
                return "Error: LLM returned no response."
            if isinstance(result, str):
                text = result
            else:
                text = "".join(result)

            # Truncate after first tool call to prevent hallucinated multi-turn
            text = _truncate_after_first_tool_call(text)

            # Log LLM turn
            self._log_context({
                "event": "llm_response",
                "model": getattr(self.router, "last_model", "unknown"),
                "message_count": len(chat_msgs),
                "response_length": len(text),
                "response": text,
            })
            messages_log.clear()
            messages_log.extend(msgs)

            if print_callback:
                print_callback(text)
            return text

        # Wrap the GUI callback to match AgenticLoop's expected signature
        tool_results_log: List[Dict[str, Any]] = []
        on_result_cb = None
        if self.on_tool_result:
            gui_callback = self.on_tool_result
            def on_result_cb(result: ToolResult) -> None:
                tool_entry = {
                    "tool": result.tool_name,
                    "success": result.success,
                    "output_length": len(result.output),
                    "output_preview": result.output[:500],
                    "error": result.error,
                }
                tool_results_log.append(tool_entry)
                self._log_context({"event": "tool_result", **tool_entry})
                status = result.output[:200] if result.success else f"ERROR: {result.error}"
                gui_callback(result.tool_name, status)
        else:
            def on_result_cb(result: ToolResult) -> None:
                tool_entry = {
                    "tool": result.tool_name,
                    "success": result.success,
                    "output_length": len(result.output),
                    "output_preview": result.output[:500],
                    "error": result.error,
                }
                tool_results_log.append(tool_entry)
                self._log_context({"event": "tool_result", **tool_entry})

        # Use a dummy llm_generate since we'll call run_sync
        loop = AgenticLoop(
            tools=active_tools,
            llm_generate=lambda msgs: iter([""]),
            max_iterations=max_iterations
        )

        # Change cwd to project dir so tools resolve relative paths correctly
        old_cwd = os.getcwd()
        try:
            os.chdir(self.project_dir)
            result = loop.run_sync(
                user_message=query,
                llm_generate_sync=custom_generate,
                on_tool_result=on_result_cb
            )
        finally:
            os.chdir(old_cwd)

        # Log final result
        self._log_context({
            "event": "query_complete",
            "response": result.get("response", ""),
            "iterations": result.get("iterations", 0),
            "tool_calls_total": len(tool_results_log),
            "full_context": messages_log,
        })

        return result
