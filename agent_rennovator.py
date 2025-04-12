#!/usr/bin/env python3
import os
import json
import subprocess
import requests
import sys
import getpass
import re
import readline  # For command history
from typing import List, Dict, Any, Optional, Tuple

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# This model ID will be updated with an available free model
MODEL = "anthropi/claude-instant-1.2"  # Default, will try to find a free model
MAX_TOKENS = 5,000
HISTORY_FILE = os.path.expanduser("~/.agent_one_history")

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_history():
    """Set up command history for the terminal interface."""
    try:
        readline.read_history_file(HISTORY_FILE)
        readline.set_history_length(5000)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Could not set up command history: {e}")

def save_history():
    """Save command history when exiting."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception as e:
        print(f"Warning: Could not save command history: {e}")

def get_api_key() -> str:
    """Get the OpenRouter API key from environment or user input."""
    api_key = OPENROUTER_API_KEY
    if not api_key:
        print(f"{Colors.CYAN}You need an OpenRouter API key to use this tool.{Colors.ENDC}")
        print(f"{Colors.CYAN}You can get one for free at https://openrouter.ai{Colors.ENDC}")
        api_key = getpass.getpass(f"{Colors.CYAN}Enter your OpenRouter API key: {Colors.ENDC}")
    return api_key

def list_models(api_key: str) -> List[Dict[str, Any]]:
    """Get a list of available models from OpenRouter."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        response.raise_for_status()
        models = response.json()
        return models.get("data", [])
    except Exception as e:
        print(f"{Colors.FAIL}Error fetching models: {str(e)}{Colors.ENDC}")
        return []

def select_model(api_key: str) -> str:
    """Allow the user to select a model from available options."""
    global MODEL
    
    print(f"{Colors.CYAN}Fetching available models from OpenRouter...{Colors.ENDC}")
    models = list_models(api_key)
    
    if not models:
        print(f"{Colors.WARNING}Could not fetch models. Using default: {MODEL}{Colors.ENDC}")
        return MODEL
    
    # Filter for potentially free or cheaper models
    free_models = [m for m in models if m.get("pricing", {}).get("prompt") == 0 or "free" in m.get("id", "").lower()]
    cheaper_models = [m for m in models if m not in free_models and m.get("pricing", {}).get("prompt", 100) < 0.01]
    
    if free_models:
        print(f"{Colors.GREEN}Found potentially free models:{Colors.ENDC}")
        for i, model in enumerate(free_models):
            print(f"{i+1}. {model.get('id')} - {model.get('description', 'No description')}")
        
        while True:
            try:
                choice = input(f"{Colors.CYAN}Select a model (1-{len(free_models)}) or press Enter for default: {Colors.ENDC}")
                if not choice:
                    break
                
                index = int(choice) - 1
                if 0 <= index < len(free_models):
                    MODEL = free_models[index]["id"]
                    print(f"{Colors.GREEN}Selected model: {MODEL}{Colors.ENDC}")
                    return MODEL
                else:
                    print(f"{Colors.WARNING}Invalid selection. Please try again.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.WARNING}Please enter a number.{Colors.ENDC}")
    elif cheaper_models:
        print(f"{Colors.WARNING}No free models found, but found some cheaper options:{Colors.ENDC}")
        for i, model in enumerate(cheaper_models):
            pricing = model.get("pricing", {})
            print(f"{i+1}. {model.get('id')} - Prompt: ${pricing.get('prompt', 'N/A')}/token, Completion: ${pricing.get('completion', 'N/A')}/token")
    else:
        print(f"{Colors.WARNING}No free models found. Using default: {MODEL}{Colors.ENDC}")
    
    return MODEL

def chat_with_model(messages: List[Dict[str, str]], api_key: str) -> str:
    """Send a chat request to the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/agent-one",  # To identify your app to OpenRouter
        "X-Title": "Agent-One Renovator"  # Optional title
    }
    
    data = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"{Colors.FAIL}Error communicating with OpenRouter API: {e}{Colors.ENDC}")
        if hasattr(e, 'response') and e.response:
            print(f"{Colors.FAIL}Response: {e.response.text}{Colors.ENDC}")
        return "Error: Failed to communicate with the API. Please check your internet connection and API key."

def extract_command(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract a command from text containing an EXECUTE: marker.
    Returns tuple (command, context) if found, None otherwise.
    """
    # Look for EXECUTE: pattern
    pattern = r"EXECUTE:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text)
    
    if not match:
        return None
    
    command = match.group(1).strip()
    # Get the paragraph containing the command for context
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "EXECUTE:" in line:
            start = max(0, i-2)
            end = min(len(lines), i+3)
            context = '\n'.join(lines[start:end])
            return command, context
    
    return command, match.group(0)

def execute_command(command: str) -> str:
    """Execute a shell command and return the output."""
    try:
        print(f"{Colors.CYAN}Executing: {command}{Colors.ENDC}")
        # Use shell=True to execute complex commands with pipes, etc.
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30  # Add a timeout to prevent hanging
        )
        
        output = ""
        if result.stdout:
            output += f"{Colors.GREEN}Stdout:{Colors.ENDC}\n{result.stdout}\n"
        if result.stderr:
            output += f"{Colors.WARNING}Stderr:{Colors.ENDC}\n{result.stderr}\n"
            
        return f"Exit code: {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"{Colors.FAIL}Error: Command execution timed out after 30 seconds{Colors.ENDC}"
    except Exception as e:
        return f"{Colors.FAIL}Error executing command: {str(e)}{Colors.ENDC}"

def is_dangerous_command(command: str) -> bool:
    """
    Check if a command might be dangerous.
    This is a simple check and not exhaustive.
    """
    dangerous_patterns = [
        r"\brm\s", r"\bdd\s", r"\bmv\s", r"\bchmod\s", r"\bchown\s",
        r"\bmkfs", r"\bfdisk", r">\s*/dev", r"\btruncate\s", r"\bsed\s+-i",
        r"\bformat\s", r";rm", r"\|rm", r"\bwipe\s", r"\bshred\s",
        r">\s*[^|]", r"2>\s*[^|]"  # Redirects that could overwrite files
    ]
    
    return any(re.search(pattern, command) for pattern in dangerous_patterns)

def main():
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║             Agent-One Renovation Helper               ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    setup_history()
    
    api_key = get_api_key()
    if not api_key:
        print(f"{Colors.FAIL}API key is required to continue.{Colors.ENDC}")
        sys.exit(1)
    
    # Let the user select a model
    selected_model = select_model(api_key)
    
    # Initialize chat history with system message
    messages = [
        {
            "role": "system", 
            "content": """You are Agent-One-Renovator, an AI assistant designed to help renovate and organize the Agent-One repository.
            You have the ability to suggest shell commands to explore, organize, and improve the repository.
            Think of the repository as a run-down home that needs renovation - you need to explore what's there,
            clean up messes, and organize things properly.
            
            When you want the user to execute a shell command, write it in the following format:
            EXECUTE: <command>
            
            Be careful with destructive commands (rm, mv, etc.). Always explain why you're suggesting
            each command and what it will do. For destructive operations, consider using safer alternatives
            first (like 'mv' instead of 'rm' to move files to a backup directory).
            
            Some useful initial commands might be:
            - ls -la
            - find . -type f | grep -v "__pycache__" | sort
            - git status (if it's a git repository)
            
            Start by exploring the repository structure to understand what you're working with.
            Then suggest commands to help organize and improve the code quality, structure, and documentation.
            
            The goal is to help the user renovate the Agent-One repository to make it more organized,
            maintainable, and functional. Think of yourself as a home renovation expert but for code repositories.
            
            Don't try to implement everything at once. Take small, incremental steps and verify the results
            after each operation. This ensures a safer and more controlled renovation process.
            """
        }
    ]
    
    print(f"{Colors.CYAN}Connecting to {selected_model}...{Colors.ENDC}")
    print(f"{Colors.GREEN}Connected! You can now chat with Agent-One Renovator.{Colors.ENDC}")
    print(f"{Colors.CYAN}Type 'exit' to quit, 'execute' to run the AI's last suggested command.{Colors.ENDC}")
    print(f"{Colors.CYAN}Type 'help' to see more commands.{Colors.ENDC}")
    
    # Add an initial message from the user to get things started
    messages.append({
        "role": "user",
        "content": "I need your help renovating the Agent-One repository. It's quite messy and I need help organizing and improving it. Can you start by exploring what we're working with?"
    })
    
    # Get an initial response from the model
    print(f"\n{Colors.CYAN}Agent-One-Renovator is thinking...{Colors.ENDC}")
    try:
        ai_response = chat_with_model(messages, api_key)
        messages.append({"role": "assistant", "content": ai_response})
        print(f"\n{Colors.GREEN}Agent-One-Renovator:{Colors.ENDC}")
        print(ai_response)
        
        if "EXECUTE: " in ai_response:
            print(f"\n{Colors.CYAN}The AI has suggested a command. Type 'execute' to run it, or continue the conversation.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Error getting initial response: {e}{Colors.ENDC}")
    
    while True:
        # Get user input
        try:
            user_input = input(f"\n{Colors.BOLD}You: {Colors.ENDC}")
        except EOFError:
            print(f"\n{Colors.WARNING}End of input. Exiting...{Colors.ENDC}")
            break
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'help':
            print(f"\n{Colors.CYAN}Available commands:{Colors.ENDC}")
            print(f"{Colors.CYAN}- exit: Exit the program{Colors.ENDC}")
            print(f"{Colors.CYAN}- execute: Execute the last command suggested by the AI{Colors.ENDC}")
            print(f"{Colors.CYAN}- help: Show this help message{Colors.ENDC}")
            print(f"{Colors.CYAN}- clear: Clear the screen{Colors.ENDC}")
            continue
        
        if user_input.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        
        if user_input.lower() == 'execute':
            # Find and execute the last command suggested by the AI
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                print(f"{Colors.WARNING}No AI response to execute a command from.{Colors.ENDC}")
                continue
                
            last_response = messages[-1]["content"]
            extracted = extract_command(last_response)
            
            if not extracted:
                print(f"{Colors.WARNING}No executable command found in the last AI message.{Colors.ENDC}")
                continue
                
            command, context = extracted
            print(f"\n{Colors.CYAN}Found command in context:{Colors.ENDC}")
            print(context)
            print(f"\n{Colors.CYAN}Extracted command:{Colors.ENDC} {command}")
            
            # Check if it's potentially dangerous
            if is_dangerous_command(command):
                print(f"\n{Colors.FAIL}⚠️ WARNING: This command may modify or delete files! ⚠️{Colors.ENDC}")
                confirmation = input(f"{Colors.WARNING}Are you absolutely sure you want to execute this command? (yes/no): {Colors.ENDC}")
                if confirmation.lower() != 'yes':
                    print(f"{Colors.CYAN}Command execution cancelled.{Colors.ENDC}")
                    messages.append({
                        "role": "user", 
                        "content": "I decided not to execute that command as it seems risky. Can you suggest a safer alternative or explain more about what you're trying to accomplish?"
                    })
                    continue
            else:
                confirmation = input(f"{Colors.CYAN}Execute this command? (y/n): {Colors.ENDC}")
                if confirmation.lower() != 'y':
                    print(f"{Colors.CYAN}Command execution cancelled.{Colors.ENDC}")
                    messages.append({
                        "role": "user", 
                        "content": "I decided not to execute that command. Can you explain more about what you're trying to do or suggest an alternative?"
                    })
                    continue
            
            # Execute the command
            output = execute_command(command)
            print(f"\n{Colors.CYAN}Command output:{Colors.ENDC}")
            print(output)
            
            # Add the execution and result to the chat history
            messages.append({
                "role": "user", 
                "content": f"I executed the command: {command}\nResult:\n{output}\n\nWhat should I do next based on this output?"
            })
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_input})
        
        # Get response from the model
        print(f"\n{Colors.CYAN}Agent-One-Renovator is thinking...{Colors.ENDC}")
        try:
            ai_response = chat_with_model(messages, api_key)
            
            # Add the AI's response to the chat history
            messages.append({"role": "assistant", "content": ai_response})
            
            # Display the AI's response
            print(f"\n{Colors.GREEN}Agent-One-Renovator:{Colors.ENDC}")
            print(ai_response)
            
            # If the AI suggested a command, prompt the user
            if "EXECUTE: " in ai_response:
                print(f"\n{Colors.CYAN}The AI has suggested a command. Type 'execute' to run it, or continue the conversation.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error getting response from the model: {e}{Colors.ENDC}")
    
    save_history()
    print(f"\n{Colors.GREEN}Thank you for using Agent-One Renovator. Goodbye!{Colors.ENDC}")

if __name__ == "__main__":
    main()
