import os
from typing import List, Optional, Any, Union
from openai import OpenAI
from termcolor import colored
from core.chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from core.globals import g

class NvidiaAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the NVIDIA NeMo API.
    """

    def __init__(self):
        """
        Initialize the NvidiaAPI with the NVIDIA-specific OpenAI client.
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv('NVIDIA_API_KEY')
        )

    def generate_response(self, chat: Union[Chat, str], model_key: str = "nvidia/llama-3.1-nemotron-70b-instruct", temperature: float = 0.7, silent_reason: str = "", base64_images: List[str] = [], thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response using the NVIDIA NeMo API.
        
        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Whether to suppress print statements.
            base64_images (List[str]): List of base64 encoded images (not used in this implementation).
            
        Returns:
            Any: A stream object that yields response chunks.
            
        Raises:
            Exception: If an error occurs during response generation, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj
            
        # Informational logging (not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"NVIDIA-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"NVIDIA-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)

        # Let errors bubble up to the router for centralized handling
        return self.client.chat.completions.create(
            model=model_key,
            messages=chat.to_openai(),
            temperature=temperature,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

    @staticmethod
    def transcribe_audio(audio_data, language: str = "", model: str = ""):
        """
        Transcribes an audio file using the NVIDIA NeMo API.
        This method is not implemented for the NVIDIA NeMo API in this example.
        """
        raise NotImplementedError("Audio transcription is not implemented for the NVIDIA NeMo API in this example.")