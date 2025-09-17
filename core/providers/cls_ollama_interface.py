import json
import sys
from typing import Any, Dict, List, Optional, Tuple
import ollama
from termcolor import colored
from core.chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from core.globals import g
import logging
import time
import re

logger = logging.getLogger(__name__)

def parse_relative_date(date_str: str) -> Optional[datetime]:
    """
    Convert relative date strings like '14 hours ago' to datetime objects.
    
    Args:
        date_str (str): Relative date string like "14 hours ago", "3 days ago"
        
    Returns:
        Optional[datetime]: Parsed datetime or None if parsing fails
    """
    if not date_str:
        return None
        
    try:
        # Parse relative dates like "14 hours ago", "3 days ago"
        match = re.search(r'(\d+)\s*(hours?|days?|weeks?|months?|years?)\s*ago', date_str, re.IGNORECASE)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            
            from datetime import timedelta
            now = datetime.now()
            
            if unit.startswith('hour'):
                return now - timedelta(hours=amount)
            elif unit.startswith('day'):
                return now - timedelta(days=amount)
            elif unit.startswith('week'):
                return now - timedelta(weeks=amount)
            elif unit.startswith('month'):
                # Approximate: 30 days per month
                return now - timedelta(days=amount * 30)
            elif unit.startswith('year'):
                # Approximate: 365 days per year
                return now - timedelta(days=amount * 365)
        
        # Try parsing absolute dates
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return datetime.strptime(date_str, '%Y-%m-%d')
            
    except Exception as e:
        logging.debug(f"Failed to parse relative date '{date_str}': {e}")
        
    return None

@dataclass
class OllamaModel:
    """
    Represents an Ollama model with its metadata.
    
    Attributes:
        model (str): The name/identifier of the model.
        modified_at (str): The last modification timestamp of the model.
        size (int): The size of the model in bytes.
        digest (str): The digest (hash) of the model.
    """
    model: str
    modified_at: str
    size: int
    digest: str
    
    def to_dict(self) -> dict:
        """
        Converts OllamaModel to a dictionary.
        
        Returns:
            dict: A dictionary representation of the OllamaModel.
        """
        return asdict(self)

@dataclass 
class OllamaModelList:
    """
    Represents a list of Ollama models.
    
    Attributes:
        models (List[OllamaModel]): A list of OllamaModel instances.
    """
    models: List[OllamaModel] = field(default_factory=list)
    
    def to_json(self) -> str:
        """
        Converts OllamaModelList to a JSON string.
        
        Returns:
            str: A JSON string representation of the OllamaModelList.
        """
        return json.dumps({"models": [model.to_dict() for model in self.models]}, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OllamaModelList':
        """
        Creates OllamaModelList from a JSON string.
        
        Args:
            json_str (str): A JSON string to parse.
        
        Returns:
            OllamaModelList: The parsed OllamaModelList instance.
        """
        data = json.loads(json_str)
        models = [
            OllamaModel(
                model=model_data["model"],
                modified_at=model_data["modified_at"],
                size=model_data["size"],
                digest=model_data["digest"]
            )
            for model_data in data["models"]
        ]
        return cls(models=models)

class OllamaClient(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []
    current_host: Optional[str] = None
    # Host discovery cache with timestamps (host -> timestamp)
    _host_cache: Dict[str, Dict[str, float]] = {"reachable": {}, "unreachable": {}}
    _cache_ttl: float = 30.0  # Cache for 30 seconds
    
    @classmethod
    def reset_host_cache(cls):
        """Reset the host reachability cache to allow retrying all hosts."""
        cls.reached_hosts.clear()
        cls.unreachable_hosts.clear()
        cls._host_cache["reachable"].clear()
        cls._host_cache["unreachable"].clear()
        # Note: unreachable_hosts now also contains host:model combinations

    @classmethod
    def check_host_reachability(cls, host: str, chat: Optional[Chat] = None) -> bool:
        """
        Validates if a host is reachable using a socket connection with caching.
        
        Args:
            host (str): The hostname to validate.
            chat (Optional[Chat]): Chat object for debug printing with title.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        current_time = time.time()
        
        # Check cache first
        if host in cls._host_cache["reachable"]:
            if current_time - cls._host_cache["reachable"][host] < cls._cache_ttl:
                return True
            else:
                # Cache expired, remove from cache
                del cls._host_cache["reachable"][host]
        
        if host in cls._host_cache["unreachable"]:
            if current_time - cls._host_cache["unreachable"][host] < cls._cache_ttl:
                return False
            else:
                # Cache expired, remove from cache
                del cls._host_cache["unreachable"][host]
        
        try:
            hostname, port_str = host.split(':') if ':' in host else (host, '11434')
            port = int(port_str)
            
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Checking host <{host}>...", "green", force_print=True, prefix=prefix)
            else:
                logging.debug(f"Ollama-Api: Checking host <{host}>...")
                
            with socket.create_connection((hostname, port), timeout=1): # 1-second timeout
                # Cache successful result
                cls._host_cache["reachable"][host] = current_time
                return True
        except (socket.timeout, socket.error, ValueError):
            # Cache failure result
            cls._host_cache["unreachable"][host] = current_time
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Host <{host}> is unreachable", "red", is_error=True, prefix=prefix)
            else:
                logging.debug(f"Ollama-Api: Host <{host}> is unreachable")
            return False

    @staticmethod
    def get_valid_client(model_key: str, chat: Optional[Chat] = None, is_small_model: bool = False) -> Tuple[ollama.Client|None, str, str]:
        """
        Returns a valid client for the given model, automatically downloading if not found.
        
        Args:
            model_key (str): The model to find a valid client for.
            chat (Optional[Chat]): Chat object for debug printing with title.
            is_small_model (bool): Whether this is a small/fast model.
        
        Returns:
            Tuple[Optional[ollama.Client], str, str]: [A valid client or None, found model_key, host].
        """
        # Get hosts from comma-separated environment variables
        ollama_host_env = os.getenv("OLLAMA_HOST", "")
        if ollama_host_env:
            ollama_hosts = ollama_host_env.split(",")
        else:
            # Default to localhost if no OLLAMA_HOST is set
            ollama_hosts = ["localhost"]
        
        # Remove the localhost from the list if explicitly configured to force remote
        force_local_remote_host = os.getenv("FORCE_REMOTE_HOST_FOR_HOSTNAME", "")
        if socket.gethostname() in force_local_remote_host:
            try:
                ollama_hosts.remove("localhost")
                ollama_hosts.remove(socket.gethostbyname(socket.gethostname()))
            except Exception:
                pass

        # Track failed hosts for this specific attempt to reduce noise
        failed_hosts_this_attempt = []
        
        for host in ollama_hosts:
            host = host.strip()
            if not host:
                continue
            
            # Skip host+model combinations that have recently failed with connection issues
            problematic_identifier = f"{host}:{model_key}"
            if problematic_identifier in OllamaClient.unreachable_hosts:
                continue
                
            if host not in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                if OllamaClient.check_host_reachability(host, chat):
                    OllamaClient.reached_hosts.append(host)
                else:
                    OllamaClient.unreachable_hosts.append(host)
                    failed_hosts_this_attempt.append(host)
            
            if host in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                client = ollama.Client(host=f'http://{host}:11434')
                try:
                    response = client.list()
                    
                    # Handle both dict and ollama._types.ListResponse
                    if hasattr(response, 'models'):
                        models_data = response.models
                    elif isinstance(response, dict) and 'models' in response:
                        models_data = response.get('models', [])
                    else:
                        logger.warning(f"Ollama host {host} returned unexpected response format: {type(response)}")
                        continue # Skip to the next host

                    logger.debug("=== START MODEL PROCESSING ===")

                    # Convert to dict for JSON serialization, compatible with OllamaModelList
                    response_dict = {"models": []}
                    for model_data in models_data:
                        model_name = model_data.get('model') or model_data.get('name') or ''
                        modified_at = model_data.get('modified_at')
                        
                        if hasattr(modified_at, 'isoformat'):
                            modified_at_str = modified_at.isoformat()
                        else:
                            modified_at_str = modified_at or datetime.now().isoformat()

                        response_dict["models"].append({
                            "model": model_name,
                            "modified_at": modified_at_str,
                            "size": model_data.get('size') or 0,
                            "digest": model_data.get('digest') or ""
                        })
                    
                    serialized = json.dumps(response_dict)
                    model_list = OllamaModelList.from_json(serialized)
                    
                    # Sort models by modification date (newest first)
                    model_list.models.sort(key=lambda m: m.modified_at, reverse=True)
                    
                    # Look for model name in all available models
                    logger.debug(f"\nSearching for model key: {model_key}")
                    found_model_key = next((
                        model.model 
                        for model in model_list.models 
                        if model.model and model_key.lower() in model.model.lower()
                    ), None)
                    logger.debug(f"Found model key: {found_model_key}")
                    logger.debug("=== END MODEL PROCESSING ===\n")
                    
                    if found_model_key:
                        # Set current_host for logging consistency
                        OllamaClient.current_host = host
                        return client, found_model_key, host
                    else:
                        # Model not found on this host, add to debug info
                        available_models = [model.model for model in model_list.models[:5]]  # Show first 5 available models
                        logger.debug(f"Model {model_key} not found on host {host}. Available: {available_models}")
                    
                    # Always attempt to download missing models
                    if chat:
                        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                        g.debug_log(f"{host} is pulling {model_key}...", "yellow", force_print=True, prefix=prefix)
                    else:
                        logging.info(colored(f"{host} is pulling {model_key}...", "yellow"))
                    try:
                        import time
                        from datetime import datetime, timedelta
                        
                        def bytes_to_mb(bytes_value):
                            return bytes_value / (1024 * 1024)

                        # Download speed tracking
                        download_start_time = time.time()
                        total_downloaded = 0
                        last_update_time = download_start_time
                        
                        for response in client.pull(model_key, stream=True):
                            if "status" in response:
                                if response["status"] == "pulling manifest":
                                    status = colored("Pulling manifest...", "yellow")
                                elif response["status"].startswith("pulling"):
                                    digest = response.get("digest", "")
                                    total = bytes_to_mb(response.get("total", 0))
                                    completed = bytes_to_mb(response.get("completed", 0))
                                    total_downloaded = max(total_downloaded, completed)
                                    
                                    # Calculate speed and ETA after 2 seconds of downloading
                                    current_time = time.time()
                                    elapsed_time = current_time - download_start_time
                                    
                                    if elapsed_time >= 2.0 and total > 0:
                                        # Calculate average speed in MB/s
                                        speed_mbps = completed / elapsed_time if elapsed_time > 0 else 0
                                        
                                        # Calculate ETA
                                        remaining_mb = total - completed
                                        eta_seconds = remaining_mb / speed_mbps if speed_mbps > 0 else 0
                                        
                                        # Format ETA
                                        if eta_seconds > 60:
                                            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                                        else:
                                            eta_str = f"{int(eta_seconds)}s"
                                        
                                        # Calculate finish time
                                        finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                                        finish_str = finish_time.strftime("%H:%M:%S")
                                        
                                        status = colored(f"Pulling {digest}: {completed:.1f}/{total:.1f} MB ({speed_mbps:.1f} MB/s, ETA: {eta_str}, done ~{finish_str})", "yellow")
                                    else:
                                        status = colored(f"Pulling {digest}: {completed:.2f}/{total:.2f} MB", "yellow")
                                else:
                                    continue
                                
                                sys.stdout.write('\r' + status)
                                sys.stdout.flush()
                        print()
                        # Set current_host for logging consistency
                        OllamaClient.current_host = host
                        return client, model_key, host
                    except Exception as e:
                        if chat:
                            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                            g.debug_log(f"Error pulling model {model_key} on host {host}: {e}", "red", is_error=True, prefix=prefix)
                        else:
                            print(f"Error pulling model {model_key} on host {host}: {e}")
                except Exception as e:
                    if chat:
                        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                        g.debug_log(f"Error checking models on host {host}: {e}", "red", is_error=True, prefix=prefix)
                    else:
                        print(f"Error checking models on host {host}: {e}")
                    # Only mark host as unreachable if it's a connection issue, not a model issue
                    error_str = str(e).lower()
                    if any(conn_error in error_str for conn_error in ['connection', 'timeout', 'refused', 'unreachable', 'network']):
                        OllamaClient.unreachable_hosts.append(host)
                        failed_hosts_this_attempt.append(host)
                    # If it's just a model not found or other non-connection error, continue to next host
                    # but don't mark this host as completely unreachable
        
        # Only show summary error if no hosts were reachable and we actually tried some
        eligible_hosts = [h.strip() for h in ollama_hosts if h.strip()]
        if failed_hosts_this_attempt and len(failed_hosts_this_attempt) == len(eligible_hosts):
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                failed_host_list = ", ".join(failed_hosts_this_attempt[:3])
                if len(failed_hosts_this_attempt) > 3:
                    failed_host_list += f" (and {len(failed_hosts_this_attempt) - 3} others)"
                g.debug_log(f"No reachable Ollama hosts found for model {model_key}. Failed hosts: {failed_host_list}", "red", is_error=True, prefix=prefix)
        elif len(eligible_hosts) == 0:
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"No eligible Ollama hosts for model {model_key}", "yellow", prefix=prefix)
        
        # If we reach here, no host had the model. Try to download on the first reachable host
        if OllamaClient.reached_hosts:
            first_host = OllamaClient.reached_hosts[0]
            client = ollama.Client(host=f'http://{first_host}:11434')
            
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"{first_host} is pulling {model_key}...", "yellow", force_print=True, prefix=prefix)
            else:
                logging.info(colored(f"{first_host} is pulling {model_key}...", "yellow"))
                
            try:
                import time
                from datetime import datetime, timedelta
                
                def bytes_to_mb(bytes_value):
                    return bytes_value / (1024 * 1024)

                # Download speed tracking
                download_start_time = time.time()
                total_downloaded = 0
                last_update_time = download_start_time
                
                for response in client.pull(model_key, stream=True):
                    if "status" in response:
                        if response["status"] == "pulling manifest":
                            status = colored("Pulling manifest...", "yellow")
                        elif response["status"].startswith("pulling"):
                            digest = response.get("digest", "")
                            total = bytes_to_mb(response.get("total", 0))
                            completed = bytes_to_mb(response.get("completed", 0))
                            total_downloaded = max(total_downloaded, completed)
                            
                            # Calculate speed and ETA after 2 seconds of downloading
                            current_time = time.time()
                            elapsed_time = current_time - download_start_time
                            
                            if elapsed_time >= 2.0 and total > 0:
                                # Calculate average speed in MB/s
                                speed_mbps = completed / elapsed_time if elapsed_time > 0 else 0
                                
                                # Calculate ETA
                                remaining_mb = total - completed
                                eta_seconds = remaining_mb / speed_mbps if speed_mbps > 0 else 0
                                
                                # Format ETA
                                if eta_seconds > 60:
                                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                                else:
                                    eta_str = f"{int(eta_seconds)}s"
                                
                                # Calculate finish time
                                finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                                finish_str = finish_time.strftime("%H:%M:%S")
                                
                                status = colored(f"Pulling {digest}: {completed:.1f}/{total:.1f} MB ({speed_mbps:.1f} MB/s, ETA: {eta_str}, done ~{finish_str})", "yellow")
                            else:
                                status = colored(f"Pulling {digest}: {completed:.2f}/{total:.2f} MB", "yellow")
                        else:
                            continue
                        
                        sys.stdout.write('\r' + status)
                        sys.stdout.flush()
                print()
                # Set current_host for logging consistency
                OllamaClient.current_host = first_host
                return client, model_key, first_host
            except Exception as e:
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log(f"Error pulling model {model_key} on host {first_host}: {e}", "red", is_error=True, prefix=prefix)
                else:
                    logging.error(f"Error pulling model {model_key} on host {first_host}: {e}")
        
        return None, None, None

    @staticmethod
    def generate_response(
        chat: Chat | str,
        model_key: str = "phi3.5:3.8b",
        temperature: Optional[float] = None,
        silent_reason: str = "",
        thinking_budget: Optional[int] = None
    ) -> Any:
        """
        Generates a response using the Ollama API, automatically downloading models if needed.

        Args:
            chat (Chat | str): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (Optional[float]): The temperature setting for the model.
            silent_reason (str): If provided, suppresses output and shows this reason.
            thinking_budget (Optional[int]): Not used in Ollama, kept for compatibility.

        Returns:
            Any: A stream object that yields response chunks.
            
        Raises:
            Exception: If there's an error generating the response, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat_inner = chat_obj
        else:
            chat_inner = chat
            
        # Store original model_key for error messages
        original_model_key = model_key
        
        # Determine if this is a small model for routing
        is_small_model = any(size in model_key.lower() for size in ['1b', '2b', '3b', '4b', 'small'])
        
        # Get a valid client for this model
        client, found_model_key, host = OllamaClient.get_valid_client(model_key, chat_inner, is_small_model)
        if not client:
            # Provide more detailed information about why no client was found
            if len(OllamaClient.unreachable_hosts) > 0:
                unreachable_info = ", ".join(OllamaClient.unreachable_hosts[:3])
                if len(OllamaClient.unreachable_hosts) > 3:
                    unreachable_info += f" (and {len(OllamaClient.unreachable_hosts) - 3} others)"
                raise Exception(f"No valid Ollama host found for model {original_model_key}. Unreachable hosts/models: {unreachable_info}")
            else:
                raise Exception(f"No valid Ollama host found for model {original_model_key}. Check if Ollama is running and the model is available.")
        
        # Store the host information on the class for logging
        OllamaClient.current_host = host
        
        # Use the found model key (which might be different from requested if we found a partial match)
        model_key = found_model_key
        
        # Default temperature if not specified
        if temperature is None:
            temperature = 0.0
            
        # Convert chat messages to Ollama format - get the messages first
        openai_messages = chat_inner.to_openai()
        
        messages = []
        for message_dict in openai_messages:
            role = message_dict["role"]
            content = message_dict["content"]
            if role == "system":
                messages.append({"role": "system", "content": content})
            elif role == "user":
                # Handle images in user messages
                if hasattr(chat_inner, 'base64_images') and chat_inner.base64_images:
                    message = {"role": "user", "content": content, "images": chat_inner.base64_images}
                    chat_inner.base64_images = []  # Clear after use
                else:
                    message = {"role": "user", "content": content}
                messages.append(message)
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        
        try:
            
            # Make the streaming API call
            stream = client.chat(
                model=model_key,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                }
            )
            
            # Verify stream is valid before returning
            if stream is None:
                host = "unknown host"
                if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                    host = client._client.base_url.host
                raise Exception(f"Ollama API at {host} returned None stream for model {model_key}")
                
            return stream
        except ollama.ResponseError as e:
            # This handles specific API errors from Ollama, like model not found.
            # We create a more descriptive error message to be handled by the router.
            host = "unknown host"
            if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                host = client._client.base_url.host
            error_message = f"Ollama API error from host {host}: {e.error}"
            raise Exception(error_message) from e
        except Exception as e:
            # For other errors (like connection errors), let the router classify them.
            # Add host info for better debugging
            host = "unknown host"
            try:
                if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                    host = client._client.base_url.host
            except:
                pass
            
            # Re-raise with host context
            if "host" not in str(e).lower():
                raise Exception(f"Ollama error from host {host}: {str(e)}") from e
            else:
                raise e

    @staticmethod
    def get_downloaded_models(host: str = "localhost") -> List[Dict[str, Any]]:
        """
        Get list of downloaded models from specified Ollama host.
        
        Args:
            host (str): The Ollama host to query
            
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        try:
            client = ollama.Client(host=f'http://{host}:11434')
            response = client.list()
            
            # Handle new dictionary-based response from ollama-python >= 0.2.0
            if isinstance(response, dict) and 'models' in response:
                models = []
                for model_data in response['models']:
                    model_info = {
                        'name': model_data.get('name', ''),
                        'size': model_data.get('size', 0),
                        'modified_at': model_data.get('modified_at', None)
                    }
                    models.append(model_info)
                # Sort by modification date (newest first), handling None
                models.sort(key=lambda m: m.get('modified_at') or datetime.min, reverse=True)
                return models
            
            # Fallback for older object-based response
            elif hasattr(response, 'models'):
                models = []
                for model in response.models:
                    model_info = {
                        'name': getattr(model, 'name', '') or getattr(model, 'model', ''),
                        'size': getattr(model, 'size', 0),
                        'modified_at': getattr(model, 'modified_at', None)
                    }
                    models.append(model_info)
                # Sort by modification date (newest first), handling None
                models.sort(key=lambda m: m.get('modified_at') or datetime.min, reverse=True)
                return models
            
            else:
                return []
            
        except Exception:
            return []

    @staticmethod
    def get_comprehensive_model_status(hosts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive model status across multiple hosts.
        
        Args:
            hosts (List[str]): List of hosts to check
            
        Returns:
            Dict[str, Dict[str, Any]]: Model status information
        """
        model_status = {}
        
        for host in hosts:
            try:
                downloaded_models = OllamaClient.get_downloaded_models(host)
                for model_info in downloaded_models:
                    model_name = model_info['name']
                    base_name = model_name.split(':')[0]  # Remove tag
                    
                    if base_name not in model_status:
                        model_status[base_name] = {
                            'downloaded': False,
                            'hosts': []
                        }
                    
                    model_status[base_name]['downloaded'] = True
                    model_status[base_name]['hosts'].append({
                        'host': host,
                        'full_name': model_name,
                        'size': model_info.get('size', 0)
                    })
            
            except Exception:
                continue
                
        return model_status

    @staticmethod
    def get_model_context_length(model_key: str, host: str = "localhost") -> Optional[int]:
        """
        Get context length for a specific model.
        
        Args:
            model_key (str): The model key to check
            host (str): The host to query
            
        Returns:
            Optional[int]: Context length if available, None otherwise
        """
        try:
            client = ollama.Client(host=f'http://{host}:11434')
            
            # Try to get model info - this might not work for all Ollama versions
            try:
                model_info = client.show(model_key)
                # Context length might be in different places depending on model
                # This is a best-effort attempt
                if hasattr(model_info, 'parameters') and model_info.parameters:
                    params = model_info.parameters
                    if 'num_ctx' in params:
                        return int(params['num_ctx'])
            except Exception:
                pass
            
            # Fallback to reasonable defaults based on model name
            model_lower = model_key.lower()
            if any(x in model_lower for x in ['large', '70b', '72b']):
                return 8192
            elif any(x in model_lower for x in ['medium', '13b', '14b', '32b', '34b']):
                return 32768
            else:
                return 128000  # Most modern models support this
                
        except Exception:
            return None

    @staticmethod
    def get_model_variants_from_web(model_name: str, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape comprehensive model variant information from Ollama library pages.
        
        Args:
            model_name (str): The base model name (e.g., 'llama3.1', 'gpt-oss')
            timeout (int): Request timeout in seconds
            
        Returns:
            List[Dict[str, Any]]: List of model variant information with tags, sizes, and specs
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import re
            
            url = f"https://ollama.com/library/{model_name}"
            
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; OllamaClient/1.0)'
            })
            
            if response.status_code != 200:
                logging.debug(f"Failed to fetch {url}: HTTP {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            variants = []
            
            # Get full page text for global size searching
            full_page_text = soup.get_text()
            
            # Look for global file size patterns in the page content
            global_sizes = {}  # tag -> size
            
            # Split into lines for better parsing
            lines = full_page_text.split('\n')
            global_dates = {}  # tag -> modification date
            
            for i, line in enumerate(lines):
                # Look for model tag lines, then check surrounding lines for sizes and dates
                model_tag_match = re.search(re.escape(model_name) + r':(\w+)', line, re.IGNORECASE)
                if model_tag_match:
                    tag = model_tag_match.group(1)
                    
                    # Check current line and next few lines for size and date information
                    search_lines = lines[i:i+5] if i+5 < len(lines) else lines[i:]
                    for search_line in search_lines:
                        # Look for size patterns like "14GB", "65GB", etc.
                        size_match = re.search(r'(\d+(?:\.\d+)?\s*[GMT]B)', search_line, re.IGNORECASE)
                        if size_match and tag not in global_sizes:
                            size = size_match.group(1)
                            global_sizes[tag] = size
                            logging.debug(f"Found size for {model_name}:{tag}: {size}")
                        
                        # Look for modification date patterns like "14 hours ago", "3 days ago", etc.
                        date_match = re.search(r'(\d+)\s*(hours?|days?|weeks?|months?|years?)\s*ago', search_line, re.IGNORECASE)
                        if date_match and tag not in global_dates:
                            amount = int(date_match.group(1))
                            unit = date_match.group(2).lower()
                            date_str = f"{amount} {unit} ago"
                            global_dates[tag] = date_str
                            logging.debug(f"Found date for {model_name}:{tag}: {date_str}")
                        
                        # Also look for absolute dates if present
                        abs_date_match = re.search(r'(\d{4}-\d{2}-\d{2})', search_line)
                        if abs_date_match and tag not in global_dates:
                            date_str = abs_date_match.group(1)
                            global_dates[tag] = date_str
                            logging.debug(f"Found absolute date for {model_name}:{tag}: {date_str}")
                
                # Also look for standalone tags that might be associated with sizes and dates
                # (sometimes the model name is on one line, tag on another, size/date on a third)
                standalone_tag_match = re.search(r'^(\w+b|latest|instruct|code)$', line.strip(), re.IGNORECASE)
                if standalone_tag_match:
                    tag = standalone_tag_match.group(1)
                    # Look backwards and forwards for model name, size, and date
                    context_lines = lines[max(0, i-3):i+4]
                    has_model_name = any(model_name in context_line.lower() for context_line in context_lines)
                    if has_model_name:
                        for context_line in context_lines:
                            # Look for size
                            size_match = re.search(r'(\d+(?:\.\d+)?\s*[GMT]B)', context_line, re.IGNORECASE)
                            if size_match and tag not in global_sizes:
                                size = size_match.group(1)
                                global_sizes[tag] = size
                                logging.debug(f"Found size for {model_name}:{tag} via context: {size}")
                            
                            # Look for date
                            date_match = re.search(r'(\d+)\s*(hours?|days?|weeks?|months?|years?)\s*ago', context_line, re.IGNORECASE)
                            if date_match and tag not in global_dates:
                                amount = int(date_match.group(1))
                                unit = date_match.group(2).lower()
                                date_str = f"{amount} {unit} ago"
                                global_dates[tag] = date_str
                                logging.debug(f"Found date for {model_name}:{tag} via context: {date_str}")
            
            # Also try broader patterns for sizes near model names
            broader_patterns = re.findall(
                r'(?:' + re.escape(model_name) + r':(\w+).*?(\d+(?:\.\d+)?\s*[GMT]B))|(?:(\d+(?:\.\d+)?\s*[GMT]B).*?' + re.escape(model_name) + r':(\w+))', 
                full_page_text, 
                re.IGNORECASE | re.DOTALL
            )
            
            for pattern in broader_patterns:
                if pattern[0] and pattern[1]:  # model:tag ... size format
                    tag, size = pattern[0], pattern[1]
                    if tag not in global_sizes:  # Don't overwrite more specific matches
                        global_sizes[tag] = size
                elif pattern[2] and pattern[3]:  # size ... model:tag format  
                    size, tag = pattern[2], pattern[3]
                    if tag not in global_sizes:
                        global_sizes[tag] = size
            
            # Look for model variant sections - these contain the download tags
            # Pattern 1: Look for elements with model tags (like :7b, :13b, :70b)
            tag_elements = soup.find_all(text=re.compile(r':[\w\d\.]+b$|:latest$|:instruct$|:code$'))
            
            for tag_text in tag_elements:
                tag_match = re.search(r':(\w[\w\d\.]*(?:b|latest|instruct|code|chat|text))$', tag_text.strip())
                if tag_match:
                    tag = tag_match.group(1)
                    full_name = f"{model_name}:{tag}"
                    
                    # Try to get size and date from global patterns first, then local context
                    size_info = global_sizes.get(tag, "")  # Check global size patterns first
                    date_info = global_dates.get(tag, "")   # Check global date patterns first
                    context_window = None
                    
                    # If no global size found, look in local context
                    if not size_info:
                        parent = tag_text.parent if hasattr(tag_text, 'parent') else None
                        if parent:
                            parent_text = parent.get_text()
                            # First, look specifically for file sizes (GB, MB, TB) - actual download sizes
                            file_size_match = re.search(r'([\d\.]+\s*[GMT]B)', parent_text, re.IGNORECASE)
                            if file_size_match:
                                size_info = file_size_match.group(1)
                            # If no clear file size, look for broader patterns but exclude parameter counts
                            elif not size_info:
                                broader_match = re.search(r'([\d\.]+\s*[KMGT]?B)(?!\s*param)', parent_text, re.IGNORECASE)
                                if broader_match and any(unit in broader_match.group(1).upper() for unit in ['GB', 'MB', 'TB']):
                                    size_info = broader_match.group(1)
                    
                    # Extract context window info from parent context if available
                    if not context_window:
                        parent = tag_text.parent if hasattr(tag_text, 'parent') else None
                        if parent:
                            parent_text = parent.get_text()
                            # Look for context window info
                            context_match = re.search(r'(\d+)k?\s*context', parent_text, re.IGNORECASE)
                            if context_match:
                                context_num = int(context_match.group(1))
                                context_window = context_num * 1000 if 'k' in context_match.group(0).lower() else context_num
                    
                    variants.append({
                        'name': full_name,
                        'tag': tag,
                        'size_str': size_info,
                        'context_window': context_window,
                        'model_base': model_name,
                        'variant_modified_at': date_info  # Individual variant modification date
                    })
            
            # Pattern 2: Look for structured model information in tables or lists
            # Find download buttons or model cards that might contain variant info
            model_cards = soup.find_all(['div', 'section'], class_=re.compile(r'model|variant|download', re.IGNORECASE))
            
            for card in model_cards:
                card_text = card.get_text()
                # Look for patterns like "ollama run llama3.1:7b"
                run_patterns = re.findall(r'ollama\s+run\s+([\w\-\.]+:[\w\d\.]+(?:b|latest|instruct|code|chat|text))', card_text, re.IGNORECASE)
                
                for pattern in run_patterns:
                    if pattern.startswith(model_name + ':'):
                        tag = pattern.split(':')[1]
                        full_name = pattern
                        
                        # Extract additional info from the card
                        size_info = ""
                        date_info = ""
                        context_window = None
                        
                        # Look for actual file sizes (GB, MB, TB) in card text
                        file_size_match = re.search(r'([\d\.]+\s*[GMT]B)', card_text, re.IGNORECASE)
                        if file_size_match:
                            size_info = file_size_match.group(1)
                        # Fallback to broader pattern but prefer actual file sizes
                        elif not size_info:
                            broader_match = re.search(r'([\d\.]+\s*[KMGT]?B)(?!\s*param)', card_text, re.IGNORECASE)
                            if broader_match and any(unit in broader_match.group(1).upper() for unit in ['GB', 'MB', 'TB']):
                                size_info = broader_match.group(1)
                        
                        # Look for modification dates in card text
                        date_match = re.search(r'(\d+)\s*(hours?|days?|weeks?|months?|years?)\s*ago', card_text, re.IGNORECASE)
                        if date_match:
                            amount = int(date_match.group(1))
                            unit = date_match.group(2).lower()
                            date_info = f"{amount} {unit} ago"
                        
                        context_match = re.search(r'(\d+)k?\s*context', card_text, re.IGNORECASE)
                        if context_match:
                            context_num = int(context_match.group(1))
                            context_window = context_num * 1000 if 'k' in context_match.group(0).lower() else context_num
                        
                        # Check if we already have this variant
                        if not any(v['name'] == full_name for v in variants):
                            variants.append({
                                'name': full_name,
                                'tag': tag,
                                'size_str': size_info,
                                'context_window': context_window,
                                'model_base': model_name,
                                'variant_modified_at': date_info  # Individual variant modification date
                            })
            
            # Remove duplicates and sort by tag (parameter count)
            unique_variants = {}
            for variant in variants:
                if variant['name'] not in unique_variants:
                    unique_variants[variant['name']] = variant
            
            result = list(unique_variants.values())
            
            # Sort by parameter count (extract numbers from tags)
            def sort_key(variant):
                tag = variant['tag']
                # Extract numeric part for sorting (e.g., '7b' -> 7, '13b' -> 13)
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', tag)
                if numeric_match:
                    return float(numeric_match.group(1))
                elif tag == 'latest':
                    return 999  # Put latest at the end
                else:
                    return 0  # Unknown tags at the beginning
            
            result.sort(key=sort_key)
            
            logging.debug(f"Scraped {len(result)} variants for {model_name}: {[v['name'] for v in result]}")
            return result
            
        except ImportError:
            logging.debug("BeautifulSoup not available for web scraping, install with: pip install beautifulsoup4")
            return []
        except Exception as e:
            logging.debug(f"Failed to scrape model variants for {model_name}: {e}")
            return []

    @staticmethod
    def get_comprehensive_downloadable_models(hosts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive list of downloadable models using web scraping and API integration.
        
        Args:
            hosts (List[str]): List of Ollama hosts to check
            
        Returns:
            Dict[str, Dict[str, Any]]: Comprehensive model information including variants
        """
        model_status = {}
        
        try:
            # First, get downloaded models status
            for host in hosts:
                try:
                    downloaded_models = OllamaClient.get_downloaded_models(host)
                    for model_info in downloaded_models:
                        model_name = model_info['name']
                        base_name = model_name.split(':')[0]
                        
                        if base_name not in model_status:
                            model_status[base_name] = {
                                'downloaded': False,
                                'hosts': [],
                                'variants': {}
                            }
                        
                        model_status[base_name]['downloaded'] = True
                        model_status[base_name]['hosts'].append({
                            'host': host,
                            'full_name': model_name,
                            'size': model_info.get('size', 0)
                        })
                        
                        # Add this specific variant as downloaded
                        if 'variants' not in model_status[base_name]:
                            model_status[base_name]['variants'] = {}
                        
                        model_status[base_name]['variants'][model_name] = {
                            'downloaded': True,
                            'size': model_info.get('size', 0),
                            'modified_at': model_info.get('modified_at')
                        }
                        
                except Exception:
                    continue
                    
            # Now get comprehensive model list from APIs and web scraping
            try:
                import requests
                
                # Get popular models from community API with last_updated dates
                community_models = {}  # model_name -> last_updated
                try:
                    community_url = "https://ollamadb.dev/api/v1/models?sort_by=last_updated&order=desc&limit=100"
                    response = requests.get(community_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        for model_data in data.get('models', []):
                            model_name = model_data.get('model_name', '')
                            last_updated = model_data.get('last_updated', '')
                            if model_name and last_updated:
                                community_models[model_name] = last_updated
                    logging.debug(f"Fetched {len(community_models)} models with dates from Community API")
                except Exception as e:
                    logging.debug(f"Community API failed: {e}")
                    pass
                
                # Add common models if API fails
                if not community_models:
                    from datetime import datetime
                    fallback_date = datetime.now().strftime('%Y-%m-%d')  # Use today's date as fallback
                    community_models = {
                        model: fallback_date for model in [
                            'llama3.3', 'llama3.2', 'llama3.1', 'llama3', 'llama2',
                            'mistral', 'mixtral', 'qwen2.5', 'qwen2', 'qwen3', 'phi3.5', 'phi3',
                            'codellama', 'deepseek-coder', 'gemma2', 'gemma', 'gpt-oss',
                            'nomic-embed-text', 'mxbai-embed-large', 'bge-m3'
                        ]
                    }
                
                # For each model, get comprehensive variant information via web scraping
                for model_base, last_updated in community_models.items():
                    if model_base not in model_status:
                        model_status[model_base] = {
                            'downloaded': False,
                            'hosts': [],
                            'variants': {},
                            'api_modified_at': last_updated  # Store API last_updated date
                        }
                    else:
                        # Add API date even for downloaded models
                        model_status[model_base]['api_modified_at'] = last_updated
                    
                    # Get variants via web scraping
                    scraped_variants = OllamaClient.get_model_variants_from_web(model_base)
                    
                    for variant_info in scraped_variants:
                        variant_name = variant_info['name']
                        
                        # Only add if not already marked as downloaded
                        if variant_name not in model_status[model_base]['variants']:
                            model_status[model_base]['variants'][variant_name] = {
                                'downloaded': False,
                                'tag': variant_info['tag'],
                                'size_str': variant_info.get('size_str', ''),
                                'context_window': variant_info.get('context_window'),
                                'model_base': variant_info['model_base'],
                                'api_modified_at': last_updated,  # Propagate API date to variants
                                'variant_modified_at': variant_info.get('variant_modified_at', '')  # Individual variant date
                            }
                
            except Exception as e:
                logging.debug(f"Failed to get comprehensive model list: {e}")
                
        except Exception as e:
            logging.warning(f"Error getting comprehensive downloadable models: {e}")
            
        return model_status

    @staticmethod
    def generate_embedding(text: str, model: str = "bge-m3", **kwargs) -> Optional[List[float]]:
        """
        Generate embeddings for text using Ollama embedding models.
        
        Args:
            text (str): The text to generate embeddings for
            model (str): The embedding model to use
            
        Returns:
            Optional[List[float]]: The embedding vector or None if failed
        """
        try:
            # Get a valid client for this model
            client, found_model_key, _ = OllamaClient.get_valid_client(model, None, False)
            if not client:
                return None
                
            # Use the found model key
            model_key = found_model_key
            
            # Generate embedding using the newer embed method
            response = client.embed(model=model_key, input=text)
            
            # Extract embedding from response
            if hasattr(response, 'embeddings') and response.embeddings:
                # Return the first embedding (since we're only passing one input)
                return response.embeddings[0] if response.embeddings else None
            elif isinstance(response, dict) and 'embeddings' in response:
                # Return the first embedding from the list
                embeddings = response['embeddings']
                return embeddings[0] if embeddings else None
            else:
                return None
                
        except Exception as e:
            # Fail silently to allow fallback to other embedding methods
            logger.debug(f"Failed to generate embedding with Ollama model {model}: {e}")
            return None


def test_ollama_functionality():
    """
    Test function to manually verify Ollama functionality.
    Run this script directly to test the Ollama interface.
    """
    print(colored("=== Testing Ollama Interface ===", "cyan", attrs=["bold"]))
    
    # Test 1: Check host reachability
    print(colored("\n1. Testing host reachability...", "yellow"))
    from core.globals import g
    test_hosts = g.DEFAULT_OLLAMA_HOSTS
    for host in test_hosts:
        reachable = OllamaClient.check_host_reachability(host)
        status = colored("✓ Reachable", "green") if reachable else colored("✗ Unreachable", "red")
        print(f"   {host}: {status}")
    
    # Test 2: List available models
    print(colored("\n2. Testing model listing...", "yellow"))
    try:
        test_models = ["qwen3-coder:latest", "bge-m3", "phi3.5:3.8b"]
        for model in test_models:
            client, found_model, _ = OllamaClient.get_valid_client(model, None, False)
            if client and found_model:
                print(f"   {model}: {colored('✓ Available as ' + found_model, 'green')}")
            else:
                print(f"   {model}: {colored('✗ Not available', 'red')}")
    except Exception as e:
        print(f"   {colored('Error listing models: ' + str(e), 'red')}")
    
    # Test 3: Test embedding generation
    print(colored("\n3. Testing embedding generation...", "yellow"))
    test_text = "Hello world, this is a test"
    try:
        embedding = OllamaClient.generate_embedding(test_text, "bge-m3")
        if embedding:
            print(f"   bge-m3: {colored('✓ Generated embedding of length ' + str(len(embedding)), 'green')}")
            print(f"   Sample values: {embedding[:5]}...")
        else:
            print(f"   bge-m3: {colored('✗ Failed to generate embedding', 'red')}")
    except Exception as e:
        print(f"   bge-m3: {colored('✗ Error: ' + str(e), 'red')}")
    
    # Test 4: Test response generation
    print(colored("\n4. Testing response generation...", "yellow"))
    test_prompt = "What is 2+2? Please respond briefly."
    test_models = ["qwen3-coder:latest", "phi3.5:3.8b"]
    
    for model in test_models:
        try:
            print(f"   Testing {model}...")
            stream = OllamaClient.generate_response(test_prompt, model, temperature=0.0)
            if stream:
                print(f"   {model}: {colored('✓ Stream created successfully', 'green')}")
                
                # Try to read a few chunks from the stream
                response_text = ""
                chunk_count = 0
                try:
                    for chunk in stream:
                        if hasattr(chunk, 'message') and chunk.message.content:
                            response_text += chunk.message.content
                            chunk_count += 1
                            if chunk_count > 10:  # Limit to prevent too much output
                                break
                    
                    if response_text:
                        print(f"     Response preview: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                    else:
                        print(f"     {colored('⚠ Stream created but no content received', 'yellow')}")
                        
                except Exception as stream_e:
                    print(f"     {colored('⚠ Error reading stream: ' + str(stream_e), 'yellow')}")
            else:
                print(f"   {model}: {colored('✗ Failed to create stream', 'red')}")
                
        except Exception as e:
            print(f"   {model}: {colored('✗ Error: ' + str(e), 'red')}")
    
    # Test 5: Test model info
    print(colored("\n5. Testing model information...", "yellow"))
    try:
        status = OllamaClient.get_comprehensive_model_status(["localhost"])
        if status:
            print(f"   Found {len(status)} model types:")
            for model_name, info in list(status.items())[:3]:  # Show first 3
                downloaded = colored("✓ Downloaded", "green") if info['downloaded'] else colored("✗ Not downloaded", "red")
                print(f"     {model_name}: {downloaded}")
        else:
            print(f"   {colored('No model status available', 'yellow')}")
    except Exception as e:
        print(f"   {colored('Error getting model status: ' + str(e), 'red')}")
    
    print(colored("\n=== Test Complete ===", "cyan", attrs=["bold"]))


if __name__ == "__main__":
    test_ollama_functionality()