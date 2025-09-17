"""
Core CLI-Agent infrastructure

This module provides the core AI and infrastructure components that form
the foundation of the CLI-Agent system.
"""

# Import from both new core location and legacy py_classes for compatibility
try:
    # Try new core location first
    from .chat import Chat, Role
except ImportError:
    try:
        # Fall back to legacy location
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_classes'))
        from cls_chat import Chat, Role
    except ImportError:
        Chat = None
        Role = None

try:
    from .llm_router import LlmRouter
except ImportError:
    try:
        from cls_llm_router import LlmRouter
    except ImportError:
        LlmRouter = None

try:
    from .ai_strengths import AIStrengths
except ImportError:
    AIStrengths = None

try:
    from .globals import g
except ImportError:
    try:
        from globals import g
    except ImportError:
        g = None

# Define what gets imported with "from core import *"
__all__ = ['LlmRouter', 'Chat', 'Role', 'AIStrengths', 'g']