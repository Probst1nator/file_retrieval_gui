import argparse
import json
import os
from typing import Dict, Tuple

from termcolor import colored

from core.chat import Chat, Role
from core.llm_router import LlmRouter
from shared.cmd_execution import select_and_execute_commands
from core.globals import g

def evaluate_existing_script(
    script_path: str, 
    context_chat: Chat, 
    requirements: str,
    args: argparse.Namespace
) -> Dict:
    """
    Evaluate if an existing script meets current requirements.
    
    Args:
        script_path: Path to the existing script
        context_chat: Current conversation context
        requirements: New requirements for the script
        args: Command line arguments
    
    Returns:
        Dict containing evaluation results with keys:
        - reasoning: str explaining the evaluation
        - decision: 'keep', 'modify', or 'replace'
        - modifications_needed: Optional list of needed changes
    """
    with open(script_path, "r") as f:
        file_content = f.read()
    
    evaluation_chat = context_chat.deep_copy()
    evaluation_chat.add_message(
        Role.USER,
        f"""Evaluate if this existing Python script meets these requirements:

REQUIREMENTS:
{requirements}

EXISTING SCRIPT:
```python
{file_content}
```

Respond with ONLY a JSON object containing:
{{
    "reasoning": "detailed evaluation of script",
    "decision": "keep" | "modify" | "replace",
    "modifications_needed": ["list of needed changes"] | null
}}"""
    )
    
    response = LlmRouter.generate_completion(
        evaluation_chat, 
        ["claude-3-5-sonnet-latest"], 
        force_local=args.local
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "reasoning": "Failed to parse evaluation response",
            "decision": "replace",
            "modifications_needed": None
        }

def request_implementation(
    context_chat: Chat, 
    requirements: str, 
    args: argparse.Namespace
) -> str:
    """
    Request a new implementation with clear requirements.
    
    Args:
        context_chat: Current conversation context
        requirements: Requirements for the script
        args: Command line arguments
    
    Returns:
        str containing the Python implementation
    """
    implement_chat = context_chat.deep_copy()
    implement_chat.add_message(
        Role.USER,
        f"""Create a Python script that meets these requirements:
{requirements}

The implementation must:
1. Use type hints for all functions and variables
2. Include comprehensive error handling
3. Include docstrings and comments
4. Be self-contained and reusable
5. Follow PEP 8 style guidelines

Respond with ONLY the Python code, no explanations or markdown."""
    )
    
    response = LlmRouter.generate_completion(
        implement_chat,
        ["claude-3-5-sonnet-latest", "gpt-4o", "qwen2.5-coder:7b-instruct"],
        force_local=args.local
    )
    
    # Clean response to ensure we only get code
    if "```python" in response:
        response = response[response.find("```python") + 9:response.rfind("```")]
    return response.strip()

def handle_execution_error(
    error_details: str, 
    context_chat: Chat, 
    args: argparse.Namespace
) -> Dict:
    """
    Analyze and handle script execution errors.
    
    Args:
        error_details: Error output from script execution
        context_chat: Current conversation context
        args: Command line arguments
    
    Returns:
        Dict containing error analysis with keys:
        - error_type: Classification of the error
        - analysis: What went wrong
        - fix_strategy: How to fix it
        - requires_rewrite: Whether a complete rewrite is needed
    """
    error_chat = context_chat.deep_copy()
    error_chat.add_message(
        Role.USER,
        f"""Analyze this Python execution error and suggest fixes:
{error_details}

Respond with ONLY a JSON object containing:
{{
    "error_type": "classification of error",
    "analysis": "what went wrong",
    "fix_strategy": "how to fix it",
    "requires_rewrite": boolean
}}"""
    )
    
    response = LlmRouter.generate_completion(
        error_chat,
        ["claude-3-5-sonnet-latest"],
        force_local=args.local
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "error_type": "unknown",
            "analysis": "Failed to analyze error",
            "fix_strategy": "Complete rewrite recommended",
            "requires_rewrite": True
        }

def execute_script(
    script_path: str, 
    context_chat: Chat,
    args: argparse.Namespace
) -> Tuple[str, bool]:
    """
    Execute a Python script and handle the results.
    
    Args:
        script_path: Path to the script to execute
        context_chat: Current conversation context
        args: Command line arguments
    
    Returns:
        Tuple of (execution_details: str, success: bool)
    """
    execution_details, execution_summary = select_and_execute_commands(
        [f"python3 {script_path}"],
        auto_execute=True,
        detached=False
    )
    
    if "error" in execution_summary.lower() or "error" in execution_details.lower():
        print(colored("Script execution encountered an error", "yellow"))
        if args.debug:
            print(colored(f"Execution details: {execution_details}", "yellow"))
        return execution_details, False
    
    return execution_details, True

def handle_python_tool(tool: dict, context_chat: Chat, args: argparse.Namespace) -> None:
    """
    Handle Python tool execution with improved error handling and script management.
    
    Args:
        tool: Dictionary containing tool details including:
            - title: Script filename
            - reasoning: Why the script is needed
        context_chat: Current conversation context
        args: Command line arguments
    """
    try:
        script_title = tool.get('title', '')
        script_reasoning = tool.get('reasoning', '')
        script_requirements = tool.get('requirements', '')
        
        script_description = f"""Title: {script_title}
Reasoning: {script_reasoning}
Requirements: {script_requirements}"""
        
        if not script_title or not script_reasoning:
            raise ValueError("Python tool requires both title and reasoning")
        
        # Setup script path
        script_dir = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "python_tool")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, script_title)
        
        # Track if we're creating a new script or modifying existing
        is_new_script = not os.path.exists(script_path)
        
        # Handle script implementation
        if is_new_script:
            print(colored(f"Creating new script: {script_title}", "green"))
            final_script = request_implementation(context_chat, script_description, args)
        else:
            print(colored(f"Evaluating existing script: {script_path}", "yellow"))
            evaluation = evaluate_existing_script(script_path, context_chat, script_description, args)
            
            if evaluation['decision'] == 'keep':
                print(colored("Using existing script as-is", "green"))
                with open(script_path, 'r') as f:
                    final_script = f.read()
            else:
                print(colored(
                    f"{'Modifying' if evaluation['decision'] == 'modify' else 'Replacing'} existing script",
                    "yellow"
                ))
                final_script = request_implementation(context_chat, script_description, args)
        
        # Write script
        with open(script_path, "w") as f:
            f.write(final_script)
        
        # Execute script
        print(colored(f"Executing script: {script_title}", "green"))
        execution_details, success = execute_script(script_path, context_chat, args)
        
        if not success:
            error_analysis = handle_execution_error(execution_details, context_chat, args)
            if error_analysis['requires_rewrite']:
                print(colored("Attempting script rewrite due to error", "yellow"))
                if args.debug:
                    print(colored(f"Error analysis: {json.dumps(error_analysis, indent=2)}", "yellow"))
                # Recursively try again with a fresh implementation
                return handle_python_tool(tool, context_chat, args)
        
        # Update context
        context_chat.add_message(Role.USER, execution_details)
        context_chat.add_message(
            Role.ASSISTANT,
            {
                "tool": "python",
                "reasoning": f"Script execution completed {'successfully' if success else 'with errors'}",
                "title": script_title,
                "reply": f"Python script '{script_title}' has been executed {'successfully' if success else 'with errors'}."
            }
        )

    except Exception as e:
        error_msg = f"Error in Python tool execution: {str(e)}"
        context_chat.add_message(
            Role.ASSISTANT,
            {
                "tool": "python",
                "reasoning": "Error occurred during Python tool execution",
                "title": tool.get('title', 'unknown'),
                "reply": error_msg
            }
        )
        print(colored(error_msg, "red"))
        if args.debug:
            import traceback
            print(colored(traceback.format_exc(), "red"))
