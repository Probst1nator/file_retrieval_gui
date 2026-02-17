# ollama_client.py
"""
Ollama REST API client for the Agentic Search feature.
Uses urllib.request (standard library) to avoid new dependencies.
"""
import asyncio
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, AsyncGenerator, Any
from dataclasses import dataclass, field


@dataclass
class OllamaConfig:
    """Configuration for Ollama API connection."""
    host: str = "localhost"
    port: int = 11434
    timeout: int = 30

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class OllamaClient:
    """
    HTTP client for Ollama REST API.

    Endpoints used:
    - GET /api/tags - List available models
    - POST /api/chat - Chat completion (streaming)
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def _make_request(self, endpoint: str, method: str = "GET",
                      data: Optional[Dict] = None) -> Any:
        """Make HTTP request to Ollama API."""
        url = f"{self.config.base_url}{endpoint}"

        request = urllib.request.Request(url, method=method)
        request.add_header("Content-Type", "application/json")

        if data:
            request.data = json.dumps(data).encode('utf-8')

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Ollama API error: {e.code} {e.reason}")

    def _make_streaming_request(self, endpoint: str, data: Dict):
        """Make streaming HTTP request to Ollama API."""
        url = f"{self.config.base_url}{endpoint}"

        request = urllib.request.Request(url, method="POST")
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(data).encode('utf-8')

        try:
            response = urllib.request.urlopen(request, timeout=self.config.timeout)
            return response
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Ollama API error: {e.code} {e.reason}")

    def list_models(self) -> List[str]:
        """
        Get list of available models from Ollama.

        Returns:
            List of model names (e.g., ['llama2', 'codellama', 'mistral'])
        """
        try:
            response = self._make_request("/api/tags")
            models = response.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except Exception:
            return []

    def chat(self, model: str, messages: List[Dict[str, str]],
             stream: bool = False) -> str:
        """
        Send chat completion request (non-streaming).

        Args:
            model: Model name (e.g., 'llama2')
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response

        Returns:
            Complete response content
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        response = self._make_request("/api/chat", method="POST", data=data)
        return response.get("message", {}).get("content", "")

    def chat_stream(self, model: str, messages: List[Dict[str, str]]):
        """
        Send chat completion request with streaming.

        Args:
            model: Model name
            messages: List of message dicts

        Yields:
            Response content chunks as they arrive
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        response = self._make_streaming_request("/api/chat", data)

        try:
            for line in response:
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        # Check if this is the final message
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

    def is_available(self) -> bool:
        """
        Check if Ollama server is running and accessible.

        Returns:
            True if server responds, False otherwise
        """
        try:
            self._make_request("/api/tags")
            return True
        except Exception:
            return False

    def get_model_info(self, model: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model.

        Args:
            model: Model name

        Returns:
            Model info dict or None if not found
        """
        try:
            response = self._make_request("/api/show", method="POST",
                                          data={"name": model})
            return response
        except Exception:
            return None


# Async wrappers for use with threading
async def async_list_models(client: OllamaClient) -> List[str]:
    """Async wrapper for list_models."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, client.list_models)


async def async_chat(client: OllamaClient, model: str,
                     messages: List[Dict[str, str]]) -> str:
    """Async wrapper for chat."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, client.chat, model, messages, False)


async def async_is_available(client: OllamaClient) -> bool:
    """Async wrapper for is_available."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, client.is_available)
