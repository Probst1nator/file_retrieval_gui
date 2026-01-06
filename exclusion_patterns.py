"""
Glob-based pattern matching for file exclusion.

This module provides VSCode/gitignore-style pattern matching for excluding files and directories.
Supports glob patterns like *.log, **/.git/, .aider*, etc.
"""

import os
import re
from typing import List, Optional


# Default exclusion patterns in glob format
DEFAULT_EXCLUSION_PATTERNS = (
    "**/__pycache__/**, **/node_modules/**, **/venv/**, **/dist/**, **/build/**, "
    "**/.git/**, **/.idea/**, **/.vscode/**, **/.claude/**, **/.aider*/**, **/.mypy_cache/**, "
    "**/*.log, **/.DS_Store, **/.env, **/.gitignore, .file_copier_cache.json"
)


def parse_pattern_string(patterns: str) -> List[str]:
    """
    Parse comma-separated pattern string into list of patterns.

    Args:
        patterns: Comma-separated patterns like "*.log, **/.git/, .env"

    Returns:
        List of trimmed pattern strings

    Examples:
        "*.log, **/.git/" -> ["*.log", "**/.git/"]
        "  test*.py  ,  *.tmp  " -> ["test*.py", "*.tmp"]
    """
    if not patterns:
        return []

    return [p.strip() for p in patterns.split(',') if p.strip()]


def glob_to_regex(pattern: str) -> str:
    """
    Convert a single glob pattern to regex.

    Supported patterns:
    - ** matches any number of directories (including zero)
    - * matches anything except /
    - ? matches any single character except /
    - [abc] character class
    - / separates path components
    - Patterns ending with / are treated as directory patterns

    Examples:
        "**/*.log" -> matches any .log file in any subdirectory
        "**/node_modules/**" -> matches node_modules dir and all contents
        ".aider*/" -> matches directories starting with .aider
        "*.py" -> matches Python files anywhere

    Returns:
        Regex pattern string (not compiled)
    """
    original_pattern = pattern

    # Check if it's a directory pattern (ends with /)
    is_dir = pattern.endswith('/')
    if is_dir:
        pattern = pattern.rstrip('/')

    # Escape special regex chars but preserve glob wildcards
    regex_pattern = pattern
    regex_pattern = regex_pattern.replace('\\', '\\\\')
    regex_pattern = regex_pattern.replace('.', '\\.')
    regex_pattern = regex_pattern.replace('+', '\\+')
    regex_pattern = regex_pattern.replace('^', '\\^')
    regex_pattern = regex_pattern.replace('$', '\\$')
    regex_pattern = regex_pattern.replace('(', '\\(')
    regex_pattern = regex_pattern.replace(')', '\\)')
    regex_pattern = regex_pattern.replace('{', '\\{')
    regex_pattern = regex_pattern.replace('}', '\\}')
    regex_pattern = regex_pattern.replace('|', '\\|')

    # Handle ** glob patterns
    # **/ at the start means "zero or more directories"
    if regex_pattern.startswith('**/'):
        regex_pattern = '((.*/)?|^)' + regex_pattern[3:]
    else:
        # ** in the middle or end
        regex_pattern = regex_pattern.replace('/**', '/.*')
        regex_pattern = regex_pattern.replace('**', '.*')

    # Handle * (matches anything except /)
    regex_pattern = regex_pattern.replace('*', '[^/]*')

    # Handle ? (matches any single character except /)
    regex_pattern = regex_pattern.replace('?', '[^/]')

    # Determine if pattern should match in any directory or just root
    has_double_star = '**' in original_pattern
    has_slash = '/' in original_pattern and not original_pattern.startswith('**/')

    if original_pattern.startswith('/'):
        # Rooted pattern - match from beginning
        if regex_pattern.startswith('/'):
            regex_pattern = '^' + regex_pattern[1:]
        else:
            regex_pattern = f'^{regex_pattern}'
    elif not has_double_star and not has_slash:
        # Simple pattern without ** or / - match at any depth (gitignore behavior)
        # E.g., .DS_Store, *.log match anywhere
        regex_pattern = f'(^|/)({regex_pattern})'
    elif not has_double_star and has_slash:
        # Pattern with / but no ** - match from root
        regex_pattern = f'^{regex_pattern}'

    # If it's a directory pattern, match the directory and its contents
    if is_dir or original_pattern.endswith('/**'):
        # Match directory itself and anything inside it
        if not regex_pattern.endswith('.*'):
            regex_pattern = f'{regex_pattern}(/.*)?'
    else:
        # For file patterns, match end of string
        if not regex_pattern.endswith('$'):
            regex_pattern = f'{regex_pattern}$'

    return regex_pattern


def parse_gitignore(gitignore_path: str, base_dir: str = None) -> List[str]:
    """
    Parse .gitignore file and convert patterns to regex strings.

    Args:
        gitignore_path: Path to .gitignore file
        base_dir: Base directory for relative path resolution (optional)

    Returns:
        List of regex pattern strings

    Notes:
        - Skips comments (lines starting with #) and empty lines
        - Handles directory patterns (trailing /)
        - Handles rooted patterns (leading /)
        - Does not support negation patterns (!) yet
    """
    patterns = []

    if not os.path.exists(gitignore_path):
        return patterns

    try:
        with open(gitignore_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # TODO: Support negation patterns (!)
                if line.startswith('!'):
                    continue

                # Convert gitignore pattern to regex using our glob_to_regex
                regex_pattern = glob_to_regex(line)
                patterns.append(regex_pattern)

    except (IOError, OSError):
        # If we can't read the file, just return empty list
        pass

    return patterns


def create_exclusion_regex(
    pattern_strings: List[str] = None,
    gitignore_patterns: List[str] = None,
    use_defaults: bool = True
) -> Optional[re.Pattern]:
    """
    Create compiled regex from multiple pattern sources.

    Args:
        pattern_strings: List of glob patterns to convert
        gitignore_patterns: List of pre-converted regex patterns from .gitignore
        use_defaults: Whether to include DEFAULT_EXCLUSION_PATTERNS

    Returns:
        Compiled regex pattern or None if no patterns

    Priority order:
        1. Custom pattern_strings (if provided)
        2. DEFAULT_EXCLUSION_PATTERNS (if use_defaults=True)
        3. gitignore_patterns (if provided)

    Example:
        regex = create_exclusion_regex(
            pattern_strings=["*.tmp", "test_*"],
            gitignore_patterns=parse_gitignore(".gitignore"),
            use_defaults=True
        )
    """
    all_regex_patterns = []

    # Add custom patterns
    if pattern_strings:
        for pattern in pattern_strings:
            regex_pattern = glob_to_regex(pattern)
            all_regex_patterns.append(regex_pattern)

    # Add default patterns
    if use_defaults and not pattern_strings:  # Only use defaults if no custom patterns
        default_patterns = parse_pattern_string(DEFAULT_EXCLUSION_PATTERNS)
        for pattern in default_patterns:
            regex_pattern = glob_to_regex(pattern)
            all_regex_patterns.append(regex_pattern)

    # Add gitignore patterns (already converted to regex)
    if gitignore_patterns:
        all_regex_patterns.extend(gitignore_patterns)

    if not all_regex_patterns:
        return None

    # Combine all patterns with OR
    final_regex_str = "|".join(f"({p})" for p in all_regex_patterns)

    try:
        return re.compile(final_regex_str, re.IGNORECASE)
    except re.error:
        # If regex compilation fails, return None
        return None


class PatternMatcher:
    """
    Efficient pattern matcher that compiles patterns once and reuses.

    Usage:
        matcher = PatternMatcher(
            patterns="*.log, **/.git/",
            gitignore_path=".gitignore",
            base_dir="/path/to/project"
        )

        if matcher.should_exclude("node_modules/package.json"):
            # Skip this file
    """

    def __init__(
        self,
        patterns: str = None,
        gitignore_path: str = None,
        base_dir: str = None,
        use_defaults: bool = True
    ):
        """
        Initialize pattern matcher with pattern sources.

        Args:
            patterns: Comma-separated glob patterns
            gitignore_path: Path to .gitignore file
            base_dir: Base directory for relative path resolution
            use_defaults: Whether to include DEFAULT_EXCLUSION_PATTERNS
        """
        self.base_dir = base_dir
        self._compiled_regex = None

        # Parse custom patterns
        pattern_list = None
        if patterns:
            pattern_list = parse_pattern_string(patterns)

        # Parse gitignore patterns
        gitignore_patterns = None
        if gitignore_path:
            gitignore_patterns = parse_gitignore(gitignore_path, base_dir)

        # Compile combined regex
        self._compiled_regex = create_exclusion_regex(
            pattern_strings=pattern_list,
            gitignore_patterns=gitignore_patterns,
            use_defaults=use_defaults
        )

    def should_exclude(self, path: str) -> bool:
        """
        Check if path should be excluded.

        Args:
            path: Relative path to check (use / separator)

        Returns:
            True if path matches any exclusion pattern

        Note:
            Automatically normalizes path separators to /
            Handles both file and directory patterns
        """
        if self._compiled_regex is None:
            return False

        # Normalize path separators to /
        normalized_path = path.replace(os.sep, '/')

        # Remove leading ./ if present
        if normalized_path.startswith('./'):
            normalized_path = normalized_path[2:]

        return bool(self._compiled_regex.search(normalized_path))

    def matches(self, path: str) -> bool:
        """Alias for should_exclude() for readability."""
        return self.should_exclude(path)
