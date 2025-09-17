"""
Path Resolution Utility for CLI-Agent Tools

This utility helps standardize import paths across all CLI-Agent tools,
ensuring consistent access to shared infrastructure regardless of where
the tool is located in the directory structure.
"""

import sys
from pathlib import Path
from typing import Optional


class PathResolver:
    """Resolves import paths for CLI-Agent tools"""
    
    _cli_agent_root: Optional[Path] = None
    
    @classmethod
    def find_cli_agent_root(cls) -> Path:
        """
        Finds the CLI-Agent root directory by looking for key marker files.
        
        Returns:
            Path to CLI-Agent root directory
            
        Raises:
            RuntimeError: If CLI-Agent root cannot be found
        """
        if cls._cli_agent_root is not None:
            return cls._cli_agent_root
            
        # Start from the current file's location
        current = Path(__file__).parent.parent  # Go up from shared/
        
        # Look for marker files that indicate CLI-Agent root
        markers = ["pyproject.toml", "py_classes", "generate_podcast.py"]
        
        while current != current.parent:
            # Check if this directory has CLI-Agent markers
            marker_count = sum(1 for marker in markers if (current / marker).exists())
            
            if marker_count >= 2:  # At least 2 markers present
                cls._cli_agent_root = current
                return current
                
            current = current.parent
        
        raise RuntimeError("Could not find CLI-Agent root directory")
    
    @classmethod
    def setup_imports(cls, tool_name: Optional[str] = None) -> Path:
        """
        Sets up import paths for CLI-Agent tools
        
        Args:
            tool_name: Name of the tool (for future tool-specific config)
            
        Returns:
            Path to CLI-Agent root
        """
        cli_agent_root = cls.find_cli_agent_root()
        
        # Add to Python path if not already present
        root_str = str(cli_agent_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        
        return cli_agent_root
    
    @classmethod
    def get_tool_path(cls, tool_name: str) -> Path:
        """
        Get the path to a specific tool directory
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Path to the tool directory
        """
        root = cls.find_cli_agent_root()
        return root / "tools" / tool_name
    
    @classmethod
    def get_shared_path(cls) -> Path:
        """Get path to shared utilities directory"""
        root = cls.find_cli_agent_root()
        return root / "shared"
    
    @classmethod
    def get_core_path(cls) -> Path:
        """Get path to core infrastructure directory"""
        root = cls.find_cli_agent_root()
        return root / "core"


# Convenience function for simple usage
def setup_cli_agent_imports(tool_name: Optional[str] = None) -> Path:
    """
    Convenience function to set up CLI-Agent imports
    
    Args:
        tool_name: Optional tool name for future tool-specific features
        
    Returns:
        Path to CLI-Agent root directory
    """
    return PathResolver.setup_imports(tool_name)