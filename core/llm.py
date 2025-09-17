import time
from typing import Dict, List, Optional, Any
from core.ai_strengths import AIStrengths
from py_classes.unified_interfaces import AIProviderInterface
import logging


class Llm:
    """
    Class representing a Language Model (LLM) with its properties and capabilities.
    """

    def __init__(
        self, 
        provider: AIProviderInterface, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[float], 
        context_window: int, 
        strengths: List[AIStrengths] = [], 
    ):
        """
        Initialize an LLM instance.

        Args:
            provider (ChatClientInterface): The chat client interface for the LLM.
            model_key (str): Unique identifier for the model.
            pricing_in_dollar_per_1M_tokens (Optional[float]): Pricing information.
            context_window (int): The context window size of the model.
            strength (AIStrengths): The strength category of the model.
        """
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
        self.context_window = context_window
        self.strengths = strengths
    
    def __str__(self) -> str:
        """
        Returns a string representation of the LLM.
        
        Returns:
            str: A formatted string with the LLM's attributes
        """
        provider_name = self.provider.__class__.__name__
        pricing = f"${self.pricing_in_dollar_per_1M_tokens}/1M tokens" if self.pricing_in_dollar_per_1M_tokens else "Free"
        strengths = ", ".join(s.name for s in self.strengths) if self.strengths else "None"
        
        return f"LLM(provider={provider_name}, model={self.model_key}, pricing={pricing}, " \
               f"context_window={self.context_window}, strengths=[{strengths}])"
    
    @property
    def has_vision(self) -> bool:
        """Returns whether the model has vision capabilities."""
        return any(s == AIStrengths.VISION for s in self.strengths)
    
    @property 
    def is_uncensored(self) -> bool:
        """Returns whether this is an uncensored model."""
        return any(s == AIStrengths.UNCENSORED for s in self.strengths)
    
    @property
    def local(self) -> bool:
        """Returns whether this is a local model."""
        return any(s == AIStrengths.LOCAL for s in self.strengths)
    
    def get_context_window(self) -> int:
        """
        Get the context window size. If None, fetch dynamically from Ollama.
        
        Returns:
            int: The context window size
        """
        if self.context_window is not None:
            return self.context_window
        
        # For Ollama models, fetch context length dynamically
        if self.provider.__class__.__name__ == 'OllamaClient':
            try:
                from core.providers.cls_ollama_interface import OllamaClient
                # Try different Ollama hosts
                from core.globals import g
                ollama_hosts = g.DEFAULT_OLLAMA_HOSTS
                for host in ollama_hosts:
                    try:
                        context_length = OllamaClient.get_model_context_length(self.model_key, host)
                        if context_length is not None:
                            # Cache the result to avoid repeated calls
                            self.context_window = context_length
                            return context_length
                    except Exception:
                        continue
                # If all hosts failed, use consistent fallback
                fallback_context = 128000
                self.context_window = fallback_context
                return fallback_context
            except Exception:
                # Final fallback - consistent value for all models
                fallback_context = 128000
                self.context_window = fallback_context
                return fallback_context
        
        # For non-Ollama models, return consistent default
        default_context = 128000
        self.context_window = default_context
        return default_context
    
    @classmethod
    def get_available_llms(cls, exclude_guards: bool = False, include_dynamic: bool = True) -> List["Llm"]:
        """
        Get the list of available LLMs, optionally including dynamically discovered Ollama models.
        
        Args:
            exclude_guards (bool): Whether to exclude guard models
            include_dynamic (bool): Whether to include dynamic discovery (slower but complete)
        
        Returns:
            List[Llm]: A list of Llm instances representing the available models.
        """
        # Import provider classes at the beginning to avoid scoping issues
        try:
            from core.providers.cls_google_interface import GoogleAPI
        except ImportError:
            GoogleAPI = None
        
        try:
            from core.providers.cls_groq_interface import GroqAPI  
        except ImportError:
            GroqAPI = None
            
        # Static cloud models and configured Ollama models
        llms = [
            # Llm(HumanAPI(), "human", None, 131072, [AIStrengths.VISION]), # For testing
        ]
        
        # Add static Google models if GoogleAPI is available
        if GoogleAPI:
            llms.extend([
                Llm(GoogleAPI(), "gemini-2.5-flash", None, 1000000, [AIStrengths.VISION]),
                Llm(GoogleAPI(), "gemini-2.5-pro", None, 1000000, [AIStrengths.VISION]),
                Llm(GoogleAPI(), "gemini-2.5-flash-preview-05-20", None, 1000000, [AIStrengths.VISION]),
                Llm(GoogleAPI(), "gemini-2.5-flash-lite-preview-06-17", None, 1000000, [AIStrengths.VISION]),
                Llm(GoogleAPI(), "gemini-2.0-flash", None, 1000000, [AIStrengths.VISION]),
            ])
        
        # Add static Groq models if GroqAPI is available
        if GroqAPI:
            llms.extend([
                Llm(GroqAPI(), "moonshotai/kimi-k2-instruct", None, 128000, []),
                Llm(GroqAPI(), "qwen/qwen3-32b", None, 128000, []),
                Llm(GroqAPI(), "llama-3.3-70b-versatile", None, 128000, []),
                Llm(GroqAPI(), "deepseek-r1-distill-llama-70b", None, 128000, []),
                Llm(GroqAPI(), "llama-3.1-8b-instant", None, 128000, []),
            ])
        
        # Add other static models
        # Llm(AnthropicAPI(), "claude-3-7-sonnet-20250219", 3, 200000, []),
        
        # Static configured Ollama models (these will be supplemented by dynamic discovery)
        # Context lengths will be dynamically fetched from Ollama
        
        # Import OllamaClient here to avoid circular import
        try:
            from core.providers.cls_ollama_interface import OllamaClient
            llms.extend([
                Llm(OllamaClient(), "gemma3n:e4b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "mistral-small3.2:latest", None, None, [AIStrengths.VISION, AIStrengths.LOCAL]),
                Llm(OllamaClient(), "magistral:latest", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "qwen3:30b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "qwen3-coder:latest", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "devstral:latest", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "gemma3n:e2b", None, None, [AIStrengths.LOCAL]),
                
                
                Llm(OllamaClient(), "qwen2.5vl:3b", None, None, [AIStrengths.VISION, AIStrengths.LOCAL]),
                Llm(OllamaClient(), "cogito:32b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "qwen3:1.7b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "qwen2.5-coder:1.5b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "qwen2.5-coder:7b", None, None, [AIStrengths.LOCAL]),
                
                
                # Guard models
                Llm(OllamaClient(), "llama-guard3:8b", None, None, [AIStrengths.LOCAL]),
                Llm(OllamaClient(), "shieldgemma:2b", None, None, [AIStrengths.LOCAL]),
            ])
        except ImportError:
            # OllamaClient not available, skip Ollama models
            pass
            
        # Non-Ollama guard models
        if GroqAPI:
            llms.extend([
                Llm(GroqAPI(), "llama-guard-4-12b", None, 128000, []),
            ])
        
        # Get existing model keys to avoid duplicates
        existing_models = {llm.model_key for llm in llms}
        
        # Add dynamically discovered Ollama models (ONLY downloaded/local models)
        # Only do this if dynamic discovery is requested (to avoid startup delays)
        if include_dynamic:
            try:
                # Try different Ollama hosts
                from core.globals import g
                ollama_hosts = g.DEFAULT_OLLAMA_HOSTS
                for host in ollama_hosts:
                    try:
                        from core.providers.cls_ollama_interface import OllamaClient
                        # IMPORTANT: Only get actually downloaded/installed models
                        # Never include downloadable-but-not-installed models in auto-selection
                        downloaded_models = OllamaClient.get_downloaded_models(host)
                        for model_info in downloaded_models:
                            model_name = model_info.get('name', '')
                            if model_name and model_name not in existing_models:
                                # Verify this model is actually downloaded by checking if it has size/modified info
                                # This ensures we don't accidentally include downloadable models
                                if not (model_info.get('size', 0) > 0 or model_info.get('modified_at')):
                                    continue  # Skip if no size/modified info (not actually downloaded)
                                
                                # Determine strengths based on model name patterns
                                strengths = [AIStrengths.LOCAL]  # All Ollama models are local
                                
                                # Add specific strengths based on model name
                                model_lower = model_name.lower()
                                if any(x in model_lower for x in ['vision', 'vl', 'visual']):
                                    strengths.append(AIStrengths.VISION)
                                if any(x in model_lower for x in ['uncensored', 'dolphin', 'wizard']):
                                    strengths.append(AIStrengths.UNCENSORED)
                                
                                # Create dynamic Llm instance (only for actually downloaded models)
                                llms.append(Llm(
                                    OllamaClient(), 
                                    model_name, 
                                    None, 
                                    None,  # Context window will be fetched dynamically
                                    strengths
                                ))
                                existing_models.add(model_name)
                    except ImportError:
                        pass  # OllamaClient not available
                    except Exception:
                        continue  # Skip failed hosts
            except Exception:
                pass  # Fall back to static list if dynamic discovery fails
        return llms

    @classmethod
    def discover_models_with_progress(cls, progress_callback=None) -> Dict[str, Any]:
        """
        Manually discover models from all providers with progress tracking.
        Uses parallel execution to update progress as each provider completes.
        
        Args:
            progress_callback: Function to call with progress updates (provider, status, models_found)
                             Status can be: 'starting', 'success', 'error', 'skipped'
        
        Returns:
            Dict containing discovered models by provider and summary information
        """
        import concurrent.futures
        import threading
        
        # Import provider classes
        try:
            from core.providers.cls_google_interface import GoogleAPI
        except ImportError:
            GoogleAPI = None
        
        try:
            from core.providers.cls_groq_interface import GroqAPI  
        except ImportError:
            GroqAPI = None
        
        results = {
            'providers': {},
            'total_discovered': 0,
            'errors': []
        }
        
        # Get current static models and previously discovered models for comparison
        static_llms = cls.get_available_llms(include_dynamic=False)
        existing_models = {llm.model_key for llm in static_llms}
        
        # Load previously discovered models from cache
        previously_discovered = cls._load_previously_discovered_models()
        existing_models.update(previously_discovered)
        
        # Thread-safe results lock
        results_lock = threading.Lock()
        
        def discover_google_models():
            """Discover Google models in a separate thread."""
            if not GoogleAPI:
                with results_lock:
                    results['providers']['Google'] = {
                        'status': 'skipped',
                        'reason': 'GoogleAPI not available or API key missing',
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                if progress_callback:
                    progress_callback('Google', 'skipped', 0)
                return
                
            if progress_callback:
                progress_callback('Google', 'starting', 0)
            
            try:
                # Check if API key is available
                import os
                if not os.getenv('GEMINI_API_KEY'):
                    with results_lock:
                        results['providers']['Google'] = {
                            'status': 'skipped',
                            'reason': 'No GEMINI_API_KEY found',
                            'models_found': 0,
                            'new_models': 0,
                            'models': []
                        }
                    if progress_callback:
                        progress_callback('Google', 'skipped', 0)
                    return
                
                discovered_google_models = GoogleAPI.get_available_models()
                new_models = []
                
                for model_info in discovered_google_models:
                    model_name = model_info.get('name', '')
                    # Clean up model name - remove "models/" prefix if present
                    if model_name.startswith('models/'):
                        model_name = model_name[7:]
                    
                    if model_name and model_name not in existing_models:
                        # Determine context window from model info
                        context_window = model_info.get('input_token_limit') or 1000000
                        
                        # Determine strengths based on model capabilities
                        strengths = []
                        methods = model_info.get('supported_generation_methods', [])
                        if 'generateContent' in methods:
                            strengths.append('VISION')
                        
                        new_models.append({
                            'name': model_name,
                            'context_window': context_window,
                            'strengths': strengths,
                            'provider': 'GoogleAPI',
                            'raw_info': model_info
                        })
                
                with results_lock:
                    results['providers']['Google'] = {
                        'status': 'success',
                        'models_found': len(discovered_google_models),
                        'new_models': len(new_models),
                        'models': new_models
                    }
                    results['total_discovered'] += len(new_models)
                
                if progress_callback:
                    progress_callback('Google', 'success', len(new_models), len(discovered_google_models))
                    
            except Exception as e:
                error_msg = f"Google discovery failed: {str(e)}"
                with results_lock:
                    results['providers']['Google'] = {
                        'status': 'error',
                        'error': error_msg,
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                    results['errors'].append(error_msg)
                
                if progress_callback:
                    progress_callback('Google', 'error', 0)
        
        def discover_groq_models():
            """Discover Groq models in a separate thread."""
            if not GroqAPI:
                with results_lock:
                    results['providers']['Groq'] = {
                        'status': 'skipped',
                        'reason': 'GroqAPI not available or API key missing',
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                if progress_callback:
                    progress_callback('Groq', 'skipped', 0)
                return
                
            if progress_callback:
                progress_callback('Groq', 'starting', 0)
            
            try:
                # Check if API key is available
                import os
                if not os.getenv('GROQ_API_KEY'):
                    with results_lock:
                        results['providers']['Groq'] = {
                            'status': 'skipped',
                            'reason': 'No GROQ_API_KEY found',
                            'models_found': 0,
                            'new_models': 0,
                            'models': []
                        }
                    if progress_callback:
                        progress_callback('Groq', 'skipped', 0)
                    return
                
                discovered_groq_models = GroqAPI.get_available_models()
                new_models = []
                
                for model_info in discovered_groq_models:
                    model_name = model_info.get('id', '')
                    
                    if model_name and model_name not in existing_models:
                        # Get context window from model info
                        context_window = model_info.get('context_window') or model_info.get('max_model_len') or 128000
                        
                        # Determine strengths based on model capabilities  
                        strengths = []
                        modalities = model_info.get('supported_modalities', [])
                        if 'vision' in modalities or 'image' in modalities:
                            strengths.append('VISION')
                        
                        new_models.append({
                            'name': model_name,
                            'context_window': context_window,
                            'strengths': strengths,
                            'provider': 'GroqAPI',
                            'raw_info': model_info
                        })
                
                with results_lock:
                    results['providers']['Groq'] = {
                        'status': 'success',
                        'models_found': len(discovered_groq_models),
                        'new_models': len(new_models),
                        'models': new_models
                    }
                    results['total_discovered'] += len(new_models)
                
                if progress_callback:
                    progress_callback('Groq', 'success', len(new_models), len(discovered_groq_models))
                    
            except Exception as e:
                error_msg = f"Groq discovery failed: {str(e)}"
                with results_lock:
                    results['providers']['Groq'] = {
                        'status': 'error',
                        'error': error_msg,
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                    results['errors'].append(error_msg)
                
                if progress_callback:
                    progress_callback('Groq', 'error', 0)
        
        def discover_ollama_models():
            """Discover Ollama models in a separate thread."""
            if progress_callback:
                progress_callback('Ollama', 'starting', 0)
            
            try:
                from core.providers.cls_ollama_interface import OllamaClient
                from core.globals import g
                
                # For model discovery progress, only get downloaded models (fast)
                # Skip the slow comprehensive web scraping for better UX
                comprehensive_status = {}
                ollama_hosts_list = [h.split(':')[0] for h in g.DEFAULT_OLLAMA_HOSTS]
                
                for host in ollama_hosts_list:
                    try:
                        if not OllamaClient.check_host_reachability(host):
                            continue
                        
                        downloaded_models = OllamaClient.get_downloaded_models(host)
                        for model_info in downloaded_models:
                            model_name = model_info['name']
                            base_name = model_name.split(':')[0]
                            
                            if base_name not in comprehensive_status:
                                comprehensive_status[base_name] = {
                                    'downloaded': True,
                                    'variants': {}
                                }
                            
                            comprehensive_status[base_name]['variants'][model_name] = {
                                'downloaded': True,
                                'size': model_info.get('size', 0),
                                'modified_at': model_info.get('modified_at')
                            }
                    except Exception:
                        continue
                
                new_models = []
                total_found = 0
                
                # Count all available models (downloaded + downloadable)
                for model_base, status_info in comprehensive_status.items():
                    variants_dict = status_info.get('variants', {})
                    total_found += len(variants_dict)
                    
                    for variant_name, variant_info in variants_dict.items():
                        if variant_name not in existing_models:
                            # Determine strengths based on model name patterns
                            strengths = ['LOCAL']  # All Ollama models are local
                            model_lower = variant_name.lower()
                            if any(x in model_lower for x in ['vision', 'vl', 'visual']):
                                strengths.append('VISION')
                            if any(x in model_lower for x in ['uncensored', 'dolphin', 'wizard']):
                                strengths.append('UNCENSORED')
                            
                            # Get context window from variant info or estimate
                            context_window = variant_info.get('context_window')
                            if not context_window:
                                # Estimation logic similar to LlmSelector
                                if any(x in model_lower for x in ['llama3.3', 'llama3.2', 'llama3.1']):
                                    context_window = 128000
                                elif 'llama3' in model_lower:
                                    context_window = 8192
                                elif any(x in model_lower for x in ['qwen2.5', 'qwen3']):
                                    context_window = 128000
                                elif any(x in model_lower for x in ['mistral', 'mixtral']):
                                    context_window = 32768
                                else:
                                    context_window = 8192
                            
                            new_models.append({
                                'name': variant_name,
                                'context_window': context_window,
                                'strengths': strengths,
                                'provider': 'OllamaClient',
                                'downloaded': variant_info.get('downloaded', False),
                                'size_str': variant_info.get('size_str', ''),
                                'raw_info': variant_info
                            })
                
                with results_lock:
                    results['providers']['Ollama'] = {
                        'status': 'success',
                        'models_found': total_found,
                        'new_models': len(new_models),
                        'models': new_models
                    }
                    results['total_discovered'] += len(new_models)
                
                if progress_callback:
                    progress_callback('Ollama', 'success', len(new_models), total_found)
                    
            except ImportError:
                with results_lock:
                    results['providers']['Ollama'] = {
                        'status': 'skipped',
                        'reason': 'OllamaClient not available',
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                if progress_callback:
                    progress_callback('Ollama', 'skipped', 0)
            except Exception as e:
                error_msg = f"Ollama discovery failed: {str(e)}"
                with results_lock:
                    results['providers']['Ollama'] = {
                        'status': 'error',
                        'error': error_msg,
                        'models_found': 0,
                        'new_models': 0,
                        'models': []
                    }
                    results['errors'].append(error_msg)
                
                if progress_callback:
                    progress_callback('Ollama', 'error', 0)
        
        # Run all discovery tasks in parallel using threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            futures = [
                executor.submit(discover_google_models),
                executor.submit(discover_groq_models),
                executor.submit(discover_ollama_models)
            ]
            
            # Wait for all to complete
            concurrent.futures.wait(futures)
        
        return results
    
    @classmethod
    def _load_previously_discovered_models(cls) -> set:
        """Load previously discovered models from cache to avoid showing them as 'new'."""
        try:
            from pathlib import Path
            from core.globals import g
            
            # Use the same path as LlmSelector for consistency
            discovered_models_file = Path(g.LLM_CONFIG_PATH).parent / "discovered_models.json"
            if not discovered_models_file.exists():
                return set()
            
            import json
            with open(discovered_models_file, 'r') as f:
                data = json.load(f)
            
            if 'models' not in data:
                return set()
            
            previously_discovered = set()
            providers_data = data['models'].get('providers', {})
            
            # Collect all previously discovered model names
            for provider_name, provider_info in providers_data.items():
                if provider_info.get('status') == 'success':
                    for model_info in provider_info.get('models', []):
                        model_name = model_info.get('name', '')
                        if model_name:
                            previously_discovered.add(model_name)
            
            return previously_discovered
            
        except Exception:
            # If we can't load the cache, assume no previously discovered models
            return set()
    
    # Removed duplicate static method - now using get_available_llms(include_dynamic=False)
    
    @classmethod
    async def _discover_ollama_models_async(cls):
        """Asynchronously discover Ollama models and update the cache."""
        if cls._discovery_in_progress:
            return
        
        cls._discovery_in_progress = True
        try:
            import asyncio
            from core.globals import g
            
            # Get current static models
            existing_models = {model.model_key for model in cls._discovered_models_cache}
            new_models = []
            
            try:
                # Try different Ollama hosts with async timeouts
                tasks = []
                for host in g.DEFAULT_OLLAMA_HOSTS:
                    # Remove protocol prefix for host checking
                    clean_host = host.replace('http://', '').replace('https://', '')
                    tasks.append(cls._discover_from_host_async(clean_host, existing_models))
                
                # Wait for all host discoveries with timeout
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, list):
                            new_models.extend(result)
                
                # Update cache with discovered models
                if new_models:
                    cls._discovered_models_cache.extend(new_models)
                    cls._cache_timestamp = time.time()
                    logging.info(f"🔍 Discovered {len(new_models)} additional Ollama models")
                
            except ImportError:
                # OllamaClient not available
                pass
            except Exception as e:
                logging.debug(f"Ollama model discovery failed: {e}")
        
        finally:
            cls._discovery_in_progress = False
    
    @classmethod
    async def _discover_from_host_async(cls, host: str, existing_models: set) -> List["Llm"]:
        """Discover models from a single Ollama host asynchronously."""
        import asyncio
        
        try:
            from core.providers.cls_ollama_interface import OllamaClient
            
            # Use asyncio timeout for the entire discovery process
            async def discover_with_timeout():
                # Run the sync discovery in a thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: OllamaClient.get_downloaded_models(host)
                )
            
            # Timeout after 3 seconds per host
            downloaded_models = await asyncio.wait_for(
                discover_with_timeout(), 
                timeout=3.0
            )
            
            new_models = []
            for model_info in downloaded_models:
                model_name = model_info.get('name', '')
                if model_name and model_name not in existing_models:
                    # Verify this model is actually downloaded
                    if not (model_info.get('size', 0) > 0 or model_info.get('modified_at')):
                        continue
                    
                    # Determine strengths based on model name patterns
                    strengths = [AIStrengths.LOCAL]  # All Ollama models are local
                    
                    model_lower = model_name.lower()
                    if any(x in model_lower for x in ['vision', 'vl', 'visual']):
                        strengths.append(AIStrengths.VISION)
                    if any(x in model_lower for x in ['uncensored', 'dolphin', 'wizard']):
                        strengths.append(AIStrengths.UNCENSORED)
                    
                    new_models.append(Llm(
                        OllamaClient(), 
                        model_name, 
                        None, 
                        None,
                        strengths
                    ))
                    existing_models.add(model_name)
            
            return new_models
            
        except asyncio.TimeoutError:
            logging.debug(f"Ollama host {host} discovery timed out")
            return []
        except Exception as e:
            logging.debug(f"Failed to discover models from {host}: {e}")
            return []