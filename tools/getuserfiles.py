#!/usr/bin/env python3
"""
File Provider Tool with Dynamic WebSocket Discovery

This tool connects to the File Retrieval GUI to retrieve the current file selection
and their contents. It uses a multi-tier discovery mechanism to find the active
WebSocket server automatically.

Discovery Strategy:
1. Environment variables (highest priority)
2. User config file (~/.cli-agent/getuserfiles.json)
3. GUI config file (.file_copier_config.json)
4. Port range scanning (8765-8769)
"""

import asyncio
import websockets
from termcolor import colored
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Discovery configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
PORT_RANGE_START = 8765
PORT_RANGE_END = 8769
CONNECTION_TIMEOUT = 3
RECEIVE_TIMEOUT = 5

# Config file paths
USER_CONFIG_DIR = Path.home() / '.cli-agent'
USER_CONFIG_FILE = USER_CONFIG_DIR / 'getuserfiles.json'
GUI_CONFIG_FILE = Path('.file_copier_config.json')

class WebSocketDiscovery:
    """Handles dynamic WebSocket endpoint discovery"""
    
    def __init__(self):
        self.cached_uri: Optional[str] = None
        self.cache_timestamp = 0
        self.cache_duration = 5  # Shorter cache for testing, 5 seconds
    
    async def discover_websocket_endpoint(self) -> str:
        """Discover WebSocket endpoint using multi-tier strategy"""
        
        # Check cache first (for rapid successive calls)
        if self._is_cache_valid():
            assert self.cached_uri is not None  # _is_cache_valid() ensures this
            print(colored(f"🔄 Using cached WebSocket endpoint: {self.cached_uri}", "cyan"))
            return self.cached_uri
        
        strategies = [
            self._strategy_environment_variables,
            self._strategy_user_config_file,
            self._strategy_gui_config_file,
            self._strategy_port_scanning
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                uri = await strategy()
                if uri and await self._test_connection(uri):
                    print(colored(f"✅ Strategy {i} successful: {uri}", "green"))
                    self._cache_uri(uri)
                    return uri
                elif uri:
                    print(colored(f"⚠️  Strategy {i} found URI {uri} but connection failed", "yellow"))
            except Exception as e:
                print(colored(f"❌ Strategy {i} error: {e}", "red"))
        
        raise ConnectionError("No accessible WebSocket server found using any discovery strategy")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached URI is still valid"""
        return (self.cached_uri is not None and
                time.time() - self.cache_timestamp < self.cache_duration)
    
    def _cache_uri(self, uri: str):
        """Cache successful URI"""
        self.cached_uri = uri
        self.cache_timestamp = time.time()
    
    async def _strategy_environment_variables(self) -> Optional[str]:
        """Strategy 1: Check environment variables"""
        host = os.getenv('GETUSERFILES_HOST', os.getenv('WEBSOCKET_HOST'))
        port = os.getenv('GETUSERFILES_PORT', os.getenv('WEBSOCKET_PORT'))
        
        if port:
            host = host or DEFAULT_HOST
            uri = f"ws://{host}:{port}"
            print(colored(f"🌍 Found environment config: {uri}", "blue"))
            return uri
        
        return None
    
    async def _strategy_user_config_file(self) -> Optional[str]:
        """Strategy 2: Check user config file"""
        if not USER_CONFIG_FILE.exists():
            return None
            
        try:
            with open(USER_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            host = config.get('host', DEFAULT_HOST)
            port = config.get('port')
            
            if port:
                uri = f"ws://{host}:{port}"
                print(colored(f"📁 Found user config: {uri}", "blue"))
                return uri
                
        except (json.JSONDecodeError, IOError) as e:
            print(colored(f"⚠️  Error reading user config: {e}", "yellow"))
        
        return None
    
    async def _strategy_gui_config_file(self) -> Optional[str]:
        """Strategy 3: Check GUI config file for active server"""
        config_paths = [
            GUI_CONFIG_FILE,
            Path.cwd() / '.file_copier_config.json',
            Path.home() / '.file_copier_config.json'
        ]
        
        for config_path in config_paths:
            if not config_path.exists():
                continue
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check for websocket server info in global section
                websocket_config = config.get('global', {}).get('websocket_server')
                if websocket_config:
                    uri = websocket_config.get('uri')
                    if uri:
                        print(colored(f"🖥️  Found GUI config: {uri}", "blue"))
                        return uri
                        
            except (json.JSONDecodeError, IOError) as e:
                print(colored(f"⚠️  Error reading GUI config {config_path}: {e}", "yellow"))
        
        return None
    
    async def _strategy_port_scanning(self) -> Optional[str]:
        """Strategy 4: Port range scanning"""
        print(colored(f"🔍 Scanning ports {PORT_RANGE_START}-{PORT_RANGE_END}...", "cyan"))
        
        for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
            uri = f"ws://{DEFAULT_HOST}:{port}"
            try:
                if await self._test_connection(uri, quick_test=True):
                    print(colored(f"📡 Found active server: {uri}", "blue"))
                    return uri
            except Exception:
                continue  # Try next port
        
        return None
    
    async def _test_connection(self, uri: str, quick_test: bool = False) -> bool:
        """Test if WebSocket server responds correctly"""
        try:
            timeout = 1 if quick_test else CONNECTION_TIMEOUT
            async with websockets.connect(uri, open_timeout=timeout) as websocket:
                # Try to receive content (basic handshake test)
                content = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                
                # Basic validation - should not be an error message
                if content:
                    if isinstance(content, bytes):
                        content_str = content.decode('utf-8', errors='ignore')
                    else:
                        content_str = content
                    if not content_str.startswith("Error:"):
                        return True
                    
        except (websockets.exceptions.ConnectionClosed, 
                websockets.exceptions.WebSocketException,
                ConnectionRefusedError,
                asyncio.TimeoutError,
                OSError):
            pass
        except Exception as e:
            if not quick_test:
                print(colored(f"⚠️  Connection test error for {uri}: {e}", "yellow"))
        
        return False


class getuserfiles:
    """
    A tool to retrieve the current file selection and their contents
    from the File Copier GUI application using dynamic discovery.
    """
    
    _discovery = WebSocketDiscovery()

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'getuserfiles'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "getuserfiles",
            "description": "Gets the current selection of files and their formatted content from the user's active File Copier GUI session. Uses automatic discovery to find the GUI server. Useful for getting project context directly from the user.",
            "example": "<getuserfiles></getuserfiles>",
            "requirements": "File Copier GUI must be running with WebSocket server enabled"
        }

    @staticmethod
    async def _fetch_content_from_gui():
        """The core async logic to connect, receive, and close using discovery."""
        try:
            # Discover the WebSocket endpoint
            uri = await getuserfiles._discovery.discover_websocket_endpoint()
            
            # Connect and retrieve content
            async with websockets.connect(uri, open_timeout=CONNECTION_TIMEOUT) as websocket:
                content = await asyncio.wait_for(websocket.recv(), timeout=RECEIVE_TIMEOUT)
                return content
                
        except ConnectionError as e:
            return f"Error: {e}"
        except (ConnectionRefusedError, asyncio.TimeoutError):
            return "Error: Could not connect to the File Copier GUI. Please ensure the GUI application is running with WebSocket server enabled."
        except websockets.exceptions.ConnectionClosed as e:
            return f"Error: Connection to the File Copier GUI was closed unexpectedly. Reason: {e.code} {e.reason}"
        except Exception as e:
            return f"An unexpected error occurred while communicating with the GUI: {e}"

    @staticmethod
    def run(content: str) -> str:
        """
        Executes the tool by connecting to the GUI's WebSocket server using discovery.

        Args:
            content: This tool does not take any arguments inside its tags.

        Returns:
            The string containing the formatted file paths and contents, or an error message.
        """
        print(colored(f"🖥️  Requesting file context from GUI via WebSocket discovery...", "cyan"))
        
        # Handle both sync and async contexts
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create a new thread to run the async code
                import concurrent.futures
                import threading
                
                def sync_wrapper():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(getuserfiles._fetch_content_from_gui())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(sync_wrapper)
                    result = future.result(timeout=30)
                    
            except RuntimeError:
                # No running event loop - we can use asyncio.run normally
                result = asyncio.run(getuserfiles._fetch_content_from_gui())
                
        except Exception as e:
            result = f"Error: {e}"
        
        if isinstance(result, bytes):
            result_str = result.decode('utf-8', errors='ignore')
        else:
            result_str = result

        if result_str.startswith("Error:"):
            print(colored(f"   -> {result_str}", "red"))
        else:
            size_kb = len(result_str) / 1024
            print(colored(f"   -> Successfully received {size_kb:.1f} KB of context.", "green"))

        return result_str

    @staticmethod
    def create_user_config(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Utility method to create user configuration file
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        config = {
            "host": host,
            "port": port,
            "protocol": "ws",
            "timeout": CONNECTION_TIMEOUT,
            "created": time.time()
        }
        
        # Ensure config directory exists
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(USER_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        print(colored(f"✅ User config created: {USER_CONFIG_FILE}", "green"))
        print(colored(f"   Server: ws://{host}:{port}", "cyan"))


# CLI interface for testing and configuration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="File Provider Tool with Dynamic Discovery")
    parser.add_argument('--test', action='store_true', help='Test connection discovery')
    parser.add_argument('--config', action='store_true', help='Create user config file')
    parser.add_argument('--host', default=DEFAULT_HOST, help='WebSocket host for config')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='WebSocket port for config')
    
    args = parser.parse_args()
    
    if args.config:
        getuserfiles.create_user_config(args.host, args.port)
    elif args.test:
        # Test the discovery mechanism
        discovery = WebSocketDiscovery()
        try:
            uri = asyncio.run(discovery.discover_websocket_endpoint())
            print(colored(f"🎉 Discovery successful: {uri}", "green", attrs=["bold"]))
        except Exception as e:
            print(colored(f"💥 Discovery failed: {e}", "red", attrs=["bold"]))
    else:
        # Run the tool
        result = getuserfiles.run("")
        if not result.startswith("Error:"):
            print("\n" + "="*60)
            print("RETRIEVED CONTENT:")
            print("="*60)
            print(result)