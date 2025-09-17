from anthropic import Anthropic
import os
from typing import Optional, Any
from termcolor import colored
from core.chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface

from core.globals import g

class AnthropicAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Anthropic API.
    """

    @staticmethod
    def generate_response(chat: Chat, model_key: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, silent_reason: str = "", thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response using the Anthropic API.

        Args:
            chat (Chat): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for suppressing print statements.
            thinking_budget (Optional[int]): Token budget for reasoning (ignored by Anthropic).

        Returns:
            Any: A stream object that yields response chunks.
            
        Raises:
            RateLimitError: If the API rate limit is exceeded.
            TimeoutError: If the request times out.
            Exception: For other errors, to be handled by the router.
        """
        # Configure the client (let any error here bubble up to the router)
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Informational logging (not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)

        # Define system content (if any)
        system_content = ""
        # Copy image URLs
        image_urls = []

        if isinstance(chat, Chat):
            # Extract the system prompt if it exists
            if hasattr(chat, 'system_prompt') and chat.system_prompt:
                system_content = chat.system_prompt
            
            # Get any image URLs from the last user message
            if chat.messages and chat.messages[-1][0] == Role.USER:
                last_user_content = chat.messages[-1][1]
                if isinstance(last_user_content, list):
                    for item in last_user_content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            # Extract the image URL
                            image_url = item.get("image_url")
                            if image_url and isinstance(image_url, dict) and "url" in image_url:
                                image_urls.append(image_url["url"])
        
        # Create messages array in Anthropic format
        anthropic_messages = []
        
        if isinstance(chat, Chat):
            # Convert Chat object to Anthropic format
            for message in chat.messages:
                # Extract role and content from the message tuple (role, content)
                message_role = message[0]  # Role enum
                message_content = message[1]  # Content string or list
                
                if message_role == Role.USER:
                    # Handle user messages with potential images
                    if image_urls and chat.messages[-1] == message:
                        # This is the last user message and has images
                        content = []
                        
                        # First add the text content
                        text_content = message_content
                        if isinstance(text_content, str) and text_content:
                            content.append({"type": "text", "text": text_content})
                        
                        # Then add each image
                        for url in image_urls:
                            if url.startswith("data:image"):
                                # Handle base64 encoded images
                                mime_type = url.split(";")[0].split(":")[1]
                                base64_data = url.split(",")[1]
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": base64_data
                                    }
                                })
                            else:
                                # Handle URL images
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                })
                        
                        anthropic_messages.append({"role": "user", "content": content})
                    else:
                        # Regular text-only message
                        anthropic_messages.append({"role": "user", "content": str(message_content)})
                elif message_role == Role.ASSISTANT:
                    anthropic_messages.append({"role": "assistant", "content": str(message_content)})
                elif message_role == Role.SYSTEM:
                    # Store system message separately
                    system_content = str(message_content)
        else:
            # If chat is a string, just use it as a user message
            anthropic_messages.append({"role": "user", "content": chat})
        
        # Create the stream with system prompt if provided
        create_args = {
            "model": model_key,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": True
        }
        
        if system_content:
            create_args["system"] = system_content
            
        # Let errors bubble up to the router for centralized handling
        # RateLimitError and TimeoutError will be handled by the router's _handle_model_error method
        return client.messages.create(**create_args)

