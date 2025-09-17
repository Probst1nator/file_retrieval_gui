import os
import subprocess
import sys
from typing import List, Tuple, Dict
import chromadb
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
from core.globals import g
from core.providers.cls_ollama_interface import OllamaClient

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyperclip
from termcolor import colored


def format_command_result(result: Dict[str, str]) -> str:
    """
    Format the command execution result for chatbot output.
    
    Args:
        result (Dict[str, str]): The result from run_command.
        
    Returns:
        str: Formatted result string.
    """
    truncation_note = "\n\nNote: Output was truncated." if result["truncated"] else ""
    
    formatted_result = f"```\n$ {result['command']}\n{result['output']}\n```"
    formatted_result += truncation_note
    
    return formatted_result

def run_command(command: str, verbose: bool = True, detached: bool = False) -> Dict[str, str]:
    """
    Run a shell command and capture its output, optimized for chatbot interaction.
    
    Args:
        command (str): The shell command to execute.
        verbose (bool): Whether to print the command and its output.
        detached (bool): Whether to run the command in detached mode.
        
    Returns:
        Dict[str, str]: A dictionary containing the command execution results.
    """
    try:
        if verbose:
            print(colored(command, 'light_green'))
        
        if detached:
            # For detached processes, use nohup to ensure they keep running
            if not command.endswith('&'):
                command = f"nohup {command} > /dev/null 2>&1 &"
            
            # For detached processes, we only care about starting them
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
                start_new_session=True  # This ensures the process is fully detached
            )
            
            # Return immediately for detached processes
            return {
                "command": command,
                "output": "Process started in background",
                "exit_code": 0,  # We assume success since we're not waiting
                "truncated": False
            }
        else:
            # Execute the command in the user's current working directory
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            stdout, stderr = process.communicate()
            
            output = stdout if stdout else stderr
            truncated = False
            
            if verbose:
                print(output)
            
            return {
                "command": command,
                "output": output.strip(),
                "exit_code": process.returncode,
                "truncated": truncated
            }
    except Exception as e:
        return {
            "command": command,
            "output": f"Error executing command: {str(e)}",
            "exit_code": -1,
            "truncated": False
        }

def select_and_execute_commands(commands: List[str], auto_execute: bool = False, verbose: bool = True, detached: bool = False) -> Tuple[str, str]:
    """
    Allow the user to select and execute a list of commands.

    Args:
        commands (List[str]): The list of commands to choose from.
        auto_execute (bool): If True, execute all commands without user selection.
        verbose (bool): Whether to print command outputs.
        detached (bool): If True, run commands in detached mode.

    Returns:
        Tuple[str, str]: Formatted result and execution summarization.
    """
    if not auto_execute:
        checkbox_list = CheckboxList(
            values=[(cmd, cmd) for i, cmd in enumerate(commands)],
            default_values=[cmd for cmd in commands]
        )
        bindings = KeyBindings()

        @bindings.add("e")
        def _execute(event) -> None:
            app.exit(result=checkbox_list.current_values)

        @bindings.add("c")
        def _copy_and_quit(event) -> None:
            selected_commands = " && ".join(checkbox_list.current_values)
            pyperclip.copy(selected_commands)
            app.exit(result=["exit"])
            
        @bindings.add("a")
        def _abort(event) -> None:
            app.exit(result=[])

        instructions = Label(text="Press 'e' to execute commands or 'c' to copy selected commands and quit. ('a' to abort)")
        root_container = HSplit([
            Frame(title="Select commands to execute, in order", body=checkbox_list),
            instructions
        ])
        layout = Layout(root_container)
        app = Application(layout=layout, key_bindings=bindings, full_screen=False)
        selected_commands = app.run()
        
        if selected_commands == ["exit"]:
            print(colored("Selected commands copied to clipboard.", "light_green"))
            sys.exit(0)
    else:
        selected_commands = commands
    
    client = chromadb.PersistentClient(g.CLIAGENT_PERSISTENT_STORAGE_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="commands")

    results = []
    summary = []
    
    for cmd in selected_commands:
        
        result = run_command(cmd, verbose, detached)
        formatted_result = format_command_result(result)
        results.append(formatted_result)
        
        status = "succeeded" if result["exit_code"] == 0 else "failed"
        summary.append(f"Command '{cmd}' {status} (Exit code: {result['exit_code']})")
        
        if not collection.get(cmd):
            cmd_embedding = OllamaClient.generate_embedding(cmd, "bge-m3")
            if cmd_embedding:
                collection.add(
                    ids=[cmd],
                    embeddings=cmd_embedding,
                    documents=[cmd]
                )
    
    formatted_output = "\n\n".join(results)
    summary_output = "\n".join(summary)
    
    return formatted_output, summary_output

# Example usage
if __name__ == "__main__":
    commands_to_run = [
        "echo Hello, World!",
        "ls -l /nonexistent",
        "python --version"
    ]
    
    detailed_output, summary = select_and_execute_commands(commands_to_run)
    print("Summary:")
    print(summary)
    print("\nDetailed Output:")
    print(detailed_output)