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


def _glob_body_to_regex(glob: str) -> str:
    """Translate the glob wildcards in a pattern body to a regex fragment.

    Scans character-by-character so ``**`` is never corrupted by a later pass
    over single ``*`` (the bug in the original implementation, where line
    ``replace('*', '[^/]*')`` ran after ``**`` had already become ``.*`` and
    turned it into ``.[^/]*``):

    - ``**/`` -> ``(?:.*/)?``  (zero or more leading directory segments)
    - ``**``  -> ``.*``        (anything, including ``/``)
    - ``*``   -> ``[^/]*``     (anything except ``/`` — a single segment)
    - ``?``   -> ``[^/]``      (a single character except ``/``)
    - ``[...]`` -> regex character class (a leading ``[!`` becomes ``[^``)
    - everything else is escaped as a literal
    """
    out: List[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == '*':
            if i + 1 < n and glob[i + 1] == '*':
                i += 2
                if i < n and glob[i] == '/':
                    out.append('(?:.*/)?')
                    i += 1
                else:
                    out.append('.*')
            else:
                out.append('[^/]*')
                i += 1
        elif c == '?':
            out.append('[^/]')
            i += 1
        elif c == '[':
            j = i + 1
            if j < n and glob[j] in '!^':
                j += 1
            if j < n and glob[j] == ']':
                j += 1
            while j < n and glob[j] != ']':
                j += 1
            if j < n:  # found a closing ]
                cls = glob[i:j + 1]
                if cls.startswith('[!'):
                    cls = '[^' + cls[2:]
                out.append(cls)
                i = j + 1
            else:  # unmatched '[' -> literal
                out.append('\\[')
                i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return ''.join(out)


def glob_to_regex(pattern: str) -> str:
    """
    Convert a single glob pattern to a regex string (uncompiled).

    Supported syntax:
    - ``**`` matches any number of directories (including zero)
    - ``*``  matches anything except ``/`` (a single path segment)
    - ``?``  matches any single character except ``/``
    - ``[abc]`` character class
    - a trailing ``/`` marks a directory pattern (matches the directory and
      everything inside it)
    - a leading ``/`` roots the pattern at the start of the path

    Anchoring (the result is used with ``re.search``):
    - rooted patterns, and patterns containing ``**`` or an internal ``/``,
      are anchored at the start (``^``)
    - a bare literal name (``.DS_Store``) matches at any depth (``(^|/)``),
      mirroring gitignore
    - a bare *wildcard* name (``*.log``) matches a single segment only, so it
      does NOT cross ``/`` — use ``**/*.log`` to match at any depth

    Examples:
        "**/*.log"            -> any .log file at any depth
        "**/node_modules/**"  -> node_modules dir and all of its contents
        "*.log"               -> a .log file in the top segment only
        ".DS_Store"           -> that file at any depth

    Returns:
        Regex pattern string (not compiled).
    """
    is_dir = pattern.endswith('/')
    rooted = pattern.startswith('/')

    core = pattern[:-1] if is_dir else pattern
    if rooted:
        core = core[1:]

    body = _glob_body_to_regex(core)

    has_double = '**' in core
    has_slash = '/' in core
    has_wildcard = '*' in core or '?' in core

    if rooted or has_double or has_slash or has_wildcard:
        # Rooted, recursive, multi-segment, or single-segment wildcard: anchor
        # at the start so a bare "*.log" cannot cross a "/".
        prefix = '^'
    else:
        # Bare literal name (e.g. .DS_Store, .file_copier_cache.json): match at
        # any depth, like gitignore.
        prefix = '(^|/)'

    # Directory patterns match the directory itself and anything beneath it.
    suffix = '(/.*)?$' if is_dir else '$'

    return f'{prefix}{body}{suffix}'


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
