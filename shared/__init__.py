"""
Shared utilities for CLI-Agent tools

This module provides common utilities and helper functions that can be
used across all CLI-Agent tools and projects.
"""

# Core shared utilities
from .dia_helper import get_dia_model
from .common_utils import extract_blocks
from .path_resolver import PathResolver, setup_cli_agent_imports

# Import from structured utilities
from .utils import WebServer, BraveSearchAPI
from .audio import *

# Define what gets imported with "from shared import *"
__all__ = ['get_dia_model', 'extract_blocks', 'PathResolver', 'setup_cli_agent_imports', 'WebServer', 'BraveSearchAPI']