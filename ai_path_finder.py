# ai_path_finder.py
import re
import json
import sys
import os
from typing import Optional

# Add project root directory to path to access py_classes
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# These are the dependencies from your existing infrastructure that this class will use.
from core.llm_router import LlmRouter
from core.chat import Chat, Role

class AIFixPath:
    """
    An AI-powered assistant to find the most likely filepath for a given code block.
    This class is self-contained and uses the provided LlmRouter for its operations.
    It has no side-effects (e.g., no printing or user input).
    """

    # Approach 1: Simple and direct. Fast but less reliable for complex cases.
    PROMPT_TEMPLATE_SIMPLE = """
I am searching for the location of a specific code block within my project.
Based on the provided project file structure and content, please identify the most likely filepath for the following code.

Return only the string of the most probable filepath and nothing else.

# Code Block to Find:
{code_block}

# Full Project Content (or relevant parts):
{full_project_context}
"""

    # Approach 2: Fallback. Uses a file tree, which is more token-efficient and can be more accurate.
    PROMPT_TEMPLATE_STRUCTURED = """
You are an expert code analysis assistant. Your task is to find the exact filepath for a given code block from a list of available files in the project.

Analyze the code block and the project file tree below to determine the most likely file location.

# Code Block to Find:
{code_block}

# Project File Tree:
{project_tree}

# Output:
Return your answer as a JSON object with a single key "filepath".
Example: {"filepath": "src/components/user/Profile.js"}
"""

    @staticmethod
    def _is_valid_path_candidate(path: str) -> bool:
        """A final sanity check on a potential filepath string."""
        if not path or not isinstance(path, str):
            return False
        # Reject paths containing markdown headers or list items at the start
        if path.strip().startswith(('##', '#', '*', '- ')):
            return False
        # Reject paths that are clearly part of a sentence
        if len(path.split()) > 4:
            return False
        # Ensure it contains at least one directory separator
        if '/' not in path and '\\' not in path:
            return False
        return True

    @classmethod
    def _extract_filepath_from_response(cls, text: str) -> Optional[str]:
        """Robustly extracts a filepath using a multi-layered strategy."""
        # Layer 1: Attempt to parse the entire response as JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'filepath' in data:
                path = data['filepath']
                if cls._is_valid_path_candidate(path): return path
        except (json.JSONDecodeError, TypeError): pass

        # Layer 2: Attempt to find and parse a JSON fenced code block
        json_match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and 'filepath' in data:
                    path = data['filepath']
                    if cls._is_valid_path_candidate(path): return path
            except (json.JSONDecodeError, TypeError): pass

        # Layer 3: Use an advanced regex for plaintext paths
        path_regex = re.compile(
            r"^\s*(?!`|#|\*)([\"']?([a-zA-Z0-9_./\\-]+[/\\][a-zA-Z0-9_./\\-]+)[\"']?)\s*$",
            re.MULTILINE
        )
        match = path_regex.search(text)
        if match:
            path = match.group(2) or match.group(1)
            if cls._is_valid_path_candidate(path): return path

        return None

    async def find_path(
        self,
        code_block: str,
        full_project_context: str,
        project_tree: str
    ) -> Optional[str]:
        """
        Asynchronously finds the filepath for a code block using a two-step LLM approach.

        Args:
            code_block: The orphan code block to find a path for.
            full_project_context: The full original string, used for the simple prompt.
            project_tree: A string representing the file tree, for the structured prompt.

        Returns:
            The suggested filepath as a string, or None if no path could be found.
        """
        # --- Approach 1: Simple Prompt ---
        prompt1 = self.PROMPT_TEMPLATE_SIMPLE.format(
            code_block=code_block,
            full_project_context=full_project_context
        )
        chat1 = Chat()
        chat1.add_message(Role.USER, prompt1)
        
        try:
            response_text = await LlmRouter.generate_completion(chat=chat1, temperature=0.0, force_free=True)
            filepath = self._extract_filepath_from_response(response_text)
            if filepath:
                return filepath
        except Exception:
            # Could be an API error, etc. We will proceed to the fallback.
            pass

        # --- Approach 2: Fallback to Structured Prompt ---
        prompt2 = self.PROMPT_TEMPLATE_STRUCTURED.format(
            code_block=code_block,
            project_tree=project_tree
        )
        chat2 = Chat()
        chat2.add_message(Role.USER, prompt2)

        try:
            response_text = await LlmRouter.generate_completion(chat=chat2, temperature=0.0, force_free=True)
            filepath = self._extract_filepath_from_response(response_text)
            if filepath:
                return filepath
        except Exception:
            # The second attempt also failed.
            return None

        return None