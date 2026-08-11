"""
Unit tests for exclusion_patterns module.
"""

import re
import os
import sys
import tempfile
import pytest

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exclusion_patterns import (
    parse_pattern_string,
    glob_to_regex,
    parse_gitignore,
    create_exclusion_regex,
    PatternMatcher,
    DEFAULT_EXCLUSION_PATTERNS
)


class TestParsePatternString:
    """Tests for parse_pattern_string function"""

    def test_simple_patterns(self):
        """Test parsing simple comma-separated patterns"""
        result = parse_pattern_string("*.log, **/.git/, .env")
        assert result == ["*.log", "**/.git/", ".env"]

    def test_with_spaces(self):
        """Test parsing patterns with extra spaces"""
        result = parse_pattern_string("  test*.py  ,  *.tmp  ")
        assert result == ["test*.py", "*.tmp"]

    def test_empty_string(self):
        """Test parsing empty string"""
        result = parse_pattern_string("")
        assert result == []

    def test_trailing_commas(self):
        """Test patterns with trailing commas"""
        result = parse_pattern_string("*.log, *.tmp,")
        assert result == ["*.log", "*.tmp"]


class TestGlobToRegex:
    """Tests for glob_to_regex function"""

    def test_simple_wildcard(self):
        """Test simple * wildcard"""
        pattern = "*.log"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search("app.log")
        assert not regex.search("logs/app.log")

    def test_recursive_wildcard(self):
        """Test ** recursive wildcard"""
        pattern = "**/*.log"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search("app.log")
        assert regex.search("logs/app.log")
        assert regex.search("a/b/c/d.log")

    def test_directory_pattern(self):
        """Test directory patterns (ending with /)"""
        pattern = "**/node_modules/**"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search("node_modules/")
        assert regex.search("node_modules/package.json")
        assert regex.search("lib/node_modules/axios/index.js")

    def test_wildcard_directory(self):
        """Test wildcard in directory names"""
        pattern = "**/.aider*/"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search(".aider/")
        assert regex.search(".aider.conf/")
        assert regex.search("src/.aider-backup/")
        assert not regex.search("aider/")

    def test_question_mark(self):
        """Test ? wildcard (single character)"""
        pattern = "test?.py"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search("test1.py")
        assert regex.search("testa.py")
        assert not regex.search("test.py")
        assert not regex.search("test12.py")

    def test_exact_filename(self):
        """Test exact filename match"""
        pattern = ".DS_Store"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search(".DS_Store")
        assert regex.search("src/.DS_Store")
        assert not regex.search(".DS_Store.backup")

    def test_rooted_pattern(self):
        """Test patterns starting with / (rooted)"""
        pattern = "/src/*.py"
        regex = re.compile(glob_to_regex(pattern))
        assert regex.search("src/app.py")
        # Note: In real implementation, rooted patterns might need adjustment


class TestParseGitignore:
    """Tests for parse_gitignore function"""

    def test_basic_gitignore(self):
        """Test parsing basic .gitignore file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gitignore', delete=False) as f:
            f.write("*.tmp\n")
            f.write("test_*\n")
            f.write("__pycache__/\n")
            f.flush()
            temp_path = f.name

        try:
            patterns = parse_gitignore(temp_path)
            assert len(patterns) >= 3

            # Compile and test patterns
            combined_regex = re.compile("|".join(f"({p})" for p in patterns))
            assert combined_regex.search("file.tmp")
            assert combined_regex.search("test_data.py")
            assert combined_regex.search("__pycache__/")
        finally:
            os.unlink(temp_path)

    def test_gitignore_with_comments(self):
        """Test .gitignore with comments and empty lines"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gitignore', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("*.log\n")
            f.write("# Another comment\n")
            f.write("*.tmp\n")
            f.flush()
            temp_path = f.name

        try:
            patterns = parse_gitignore(temp_path)
            # Should only include non-comment, non-empty patterns
            assert len(patterns) == 2
        finally:
            os.unlink(temp_path)

    def test_nonexistent_gitignore(self):
        """Test handling of non-existent .gitignore file"""
        patterns = parse_gitignore("/nonexistent/path/.gitignore")
        assert patterns == []


class TestCreateExclusionRegex:
    """Tests for create_exclusion_regex function"""

    def test_with_custom_patterns(self):
        """Test creating regex with custom patterns"""
        patterns = ["*.tmp", "test_*"]
        regex = create_exclusion_regex(pattern_strings=patterns, use_defaults=False)

        assert regex is not None
        assert regex.search("file.tmp")
        assert regex.search("test_data.py")
        assert not regex.search("app.py")

    def test_with_defaults(self):
        """Test creating regex with default patterns"""
        regex = create_exclusion_regex(use_defaults=True)

        assert regex is not None
        assert regex.search("__pycache__/")
        assert regex.search("node_modules/package.json")
        assert regex.search(".git/config")
        assert regex.search("app.log")

    def test_combined_patterns(self):
        """Test combining custom and gitignore patterns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gitignore', delete=False) as f:
            f.write("*.tmp\n")
            f.flush()
            temp_path = f.name

        try:
            gitignore_patterns = parse_gitignore(temp_path)
            regex = create_exclusion_regex(
                pattern_strings=["test_*"],
                gitignore_patterns=gitignore_patterns,
                use_defaults=False
            )

            assert regex is not None
            assert regex.search("file.tmp")
            assert regex.search("test_data.py")
        finally:
            os.unlink(temp_path)


class TestPatternMatcher:
    """Tests for PatternMatcher class"""

    def test_should_exclude_with_defaults(self):
        """Test PatternMatcher with default patterns"""
        matcher = PatternMatcher(use_defaults=True)

        assert matcher.should_exclude("__pycache__/")
        assert matcher.should_exclude("__pycache__/module.pyc")
        assert matcher.should_exclude("node_modules/package.json")
        assert matcher.should_exclude(".git/config")
        assert matcher.should_exclude("app.log")
        assert matcher.should_exclude(".DS_Store")
        assert matcher.should_exclude(".env")

        assert not matcher.should_exclude("app.py")
        assert not matcher.should_exclude("src/main.py")

    def test_should_exclude_with_custom_patterns(self):
        """Test PatternMatcher with custom patterns"""
        matcher = PatternMatcher(patterns="*.tmp, test_*", use_defaults=False)

        assert matcher.should_exclude("file.tmp")
        assert matcher.should_exclude("test_data.py")
        assert not matcher.should_exclude("app.py")

    def test_should_exclude_with_gitignore(self):
        """Test PatternMatcher with .gitignore integration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gitignore', delete=False) as f:
            f.write("*.tmp\n")
            f.write("test_*\n")
            f.flush()
            temp_path = f.name

        try:
            matcher = PatternMatcher(
                gitignore_path=temp_path,
                use_defaults=False
            )

            assert matcher.should_exclude("file.tmp")
            assert matcher.should_exclude("test_data.py")
            assert not matcher.should_exclude("app.py")
        finally:
            os.unlink(temp_path)

    def test_matches_alias(self):
        """Test that matches() is an alias for should_exclude()"""
        matcher = PatternMatcher(patterns="*.log", use_defaults=False)

        assert matcher.matches("app.log") == matcher.should_exclude("app.log")
        assert matcher.matches("app.py") == matcher.should_exclude("app.py")

    def test_path_normalization(self):
        """Test that PatternMatcher normalizes path separators"""
        matcher = PatternMatcher(patterns="**/test/**", use_defaults=False)

        # Test with different separators
        assert matcher.should_exclude("test/file.py")
        assert matcher.should_exclude("src/test/data.py")

    def test_empty_matcher(self):
        """Test PatternMatcher with no patterns"""
        matcher = PatternMatcher(use_defaults=False)

        # Should not exclude anything
        assert not matcher.should_exclude("any/file.py")
        assert not matcher.should_exclude("__pycache__/")


class TestDefaultExclusionPatterns:
    """Tests for DEFAULT_EXCLUSION_PATTERNS"""

    def test_default_patterns_exist(self):
        """Test that default patterns are defined"""
        assert DEFAULT_EXCLUSION_PATTERNS is not None
        assert len(DEFAULT_EXCLUSION_PATTERNS) > 0

    def test_default_patterns_format(self):
        """Test that default patterns can be parsed"""
        patterns = parse_pattern_string(DEFAULT_EXCLUSION_PATTERNS)
        assert len(patterns) > 0

        # Check that common patterns are included
        pattern_strings = ", ".join(patterns)
        assert any("pycache" in p.lower() for p in patterns)
        assert any("node_modules" in p for p in patterns)
        assert any("git" in p.lower() for p in patterns)

    def test_default_patterns_compile(self):
        """Test that default patterns compile to valid regex"""
        patterns = parse_pattern_string(DEFAULT_EXCLUSION_PATTERNS)
        regex = create_exclusion_regex(pattern_strings=patterns, use_defaults=False)

        assert regex is not None
        # Should be able to search without errors
        assert regex.search("__pycache__/") is not None


class TestEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_special_regex_characters(self):
        """Test that special regex characters are properly escaped"""
        pattern = "file.with.dots.txt"
        regex = re.compile(glob_to_regex(pattern))

        assert regex.search("file.with.dots.txt")
        assert not regex.search("fileXwithXdotsXtxt")

    def test_empty_pattern(self):
        """Test handling of empty patterns"""
        regex = create_exclusion_regex(pattern_strings=[], use_defaults=False)
        assert regex is None

    def test_cache_filename_excluded(self):
        """Test that .file_copier_cache.json is in defaults"""
        matcher = PatternMatcher(use_defaults=True)
        assert matcher.should_exclude(".file_copier_cache.json")

    def test_dotfile_patterns(self):
        """Test patterns starting with dot"""
        matcher = PatternMatcher(patterns="**/.aider*/, **/.claude/", use_defaults=False)

        assert matcher.should_exclude(".aider/")
        assert matcher.should_exclude(".aider.conf/")
        assert matcher.should_exclude(".claude/")
        assert matcher.should_exclude("src/.claude/config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
