"""Tests for the file_retrieval_gui "Apply Changes to Files" parse/apply path.

Covers smart_paster only — no tkinter, no network, everything against tmp_path.
The module is loaded under a unique name because other tools in this monorepo
also ship a `smart_paster`-adjacent module namespace (see tools/CLAUDE.md
§ Test conventions).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parent

# smart_paster does plain `import exclusion_patterns`-style sibling imports, the
# same way gui.py sets itself up at startup.
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


def _load_smart_paster():
    spec = importlib.util.spec_from_file_location(
        "frg_smart_paster", TOOL_DIR / "smart_paster.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["frg_smart_paster"] = module
    spec.loader.exec_module(module)
    return module


sp = _load_smart_paster()


@pytest.fixture
def project(tmp_path):
    """A project root with one nested file."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    return tmp_path


def _patch(path, search, replace):
    return (
        f'<patch file="{path}">\n'
        f"<search>\n{search}\n</search>\n"
        f"<replace>\n{replace}\n</replace>\n"
        f"</patch>\n"
    )


# --- duplicate <patch> blocks chain instead of clobbering -------------------


def test_duplicate_patch_blocks_chain_into_one_change(project):
    blob = _patch("sub/a.py", "one", "ONE") + _patch("sub/a.py", "three", "THREE")

    changes = sp.preview_changes_to_files(blob, str(project))

    assert len(changes) == 1, "two patches to one file must merge into one change"
    change = changes[0]
    assert change.change_type is sp.ChangeType.MODIFY_FILE
    assert change.content == "ONE\ntwo\nTHREE\n", "both edits must survive"
    assert change.original_content == "one\ntwo\nthree\n"

    results = sp.apply_selected_changes(changes)
    assert results["errors"] == []
    assert (project / "sub" / "a.py").read_text(encoding="utf-8") == "ONE\ntwo\nTHREE\n"


def test_failing_second_patch_invalidates_the_whole_file(project):
    blob = _patch("sub/a.py", "one", "ONE") + _patch("sub/a.py", "nowhere", "X")

    changes = sp.preview_changes_to_files(blob, str(project))

    assert len(changes) == 1
    change = changes[0]
    assert change.change_type is sp.ChangeType.INVALID_PATH
    assert "Patch Failed" in change.error_message
    assert change.selected is False

    sp.apply_selected_changes(changes)
    assert (project / "sub" / "a.py").read_text(encoding="utf-8") == "one\ntwo\nthree\n", (
        "an all-or-nothing failure must leave the file untouched"
    )


def test_full_overwrite_patch_replaces_a_pending_surgical_patch(project):
    blob = _patch("sub/a.py", "one", "ONE") + (
        '<patch file="sub/a.py">\n<replace>\nwholly new\n</replace>\n</patch>\n'
    )

    changes = sp.preview_changes_to_files(blob, str(project))

    assert len(changes) == 1
    assert changes[0].content == "wholly new\n"
    assert changes[0].change_type is sp.ChangeType.MODIFY_FILE


# --- history is filed under the project root -------------------------------


def test_history_is_keyed_to_the_project_root_not_the_subdirectory(project):
    blob = "**File:** `sub/a.py`\n\n```python\nreplaced\n```\n"
    changes = sp.preview_changes_to_files(blob, str(project))

    results, operation = sp.apply_selected_with_history(changes, str(project))

    assert results["success"] == ["sub/a.py"]
    assert operation is not None
    assert operation.root_directory == str(project)

    history = sp.ApplyHistoryManager()
    history.add_operation(operation)
    assert history.can_undo(str(project)) is True, "Undo must be offered for the root the GUI queries"


def test_undo_restores_a_nested_file(project):
    blob = "**File:** `sub/a.py`\n\n```python\nreplaced\n```\n"
    changes = sp.preview_changes_to_files(blob, str(project))
    _, operation = sp.apply_selected_with_history(changes, str(project))

    assert (project / "sub" / "a.py").read_text(encoding="utf-8") == "replaced\n"

    sp.apply_snapshots(operation.before_snapshots)
    assert (project / "sub" / "a.py").read_text(encoding="utf-8") == "one\ntwo\nthree\n"


# --- the heading pattern must not reach across prose ------------------------


def test_prose_mentioning_a_filename_does_not_capture_a_later_code_block(project):
    blob = """
## 2. Rationale

We refactored things. See the note about config.yaml for details.
It was tricky.

Here is an unrelated illustrative snippet, NOT a file to write:

```bash
rm -rf /tmp/foo
```

Regular prose continues.
"""
    assert sp.preview_changes_to_files(blob, str(project)) == []


def test_supported_markdown_formats_still_parse(project):
    annotation = "**File:** `x.txt`\n\n```\nfrom annotation\n```\n"
    heading = "# y.txt\n```\nfrom heading\n```\n"
    inline = "```python\n# z.py\nfrom inline\n```\n"

    for blob, expected_path, expected_content in [
        (annotation, "x.txt", "from annotation\n"),
        (heading, "y.txt", "from heading\n"),
        (inline, "z.py", "from inline\n"),
    ]:
        changes = sp.preview_changes_to_files(blob, str(project))
        assert [c.file_path for c in changes] == [expected_path], blob
        assert changes[0].content == expected_content
        assert changes[0].change_type is sp.ChangeType.NEW_FILE


# --- diff rendering ---------------------------------------------------------


def test_diff_lines_carry_no_trailing_newlines(project):
    blob = "**File:** `sub/a.py`\n\n```python\none\nTWO\nthree\n```\n"
    change = sp.preview_changes_to_files(blob, str(project))[0]

    assert change.diff_lines, "a modification must produce a diff"
    assert not any(line.endswith("\n") for line in change.diff_lines), (
        "the renderer appends its own newline; keepends here double-spaced the view"
    )
    assert change.lines_added == 1
    assert change.lines_removed == 1


# --- invalid changes explain themselves and are never pre-selected ----------


def test_invalid_changes_are_unselected_and_carry_their_reason(project):
    blob = _patch("sub/missing.py", "anything", "X")
    change = sp.preview_changes_to_files(blob, str(project))[0]

    assert change.change_type is sp.ChangeType.INVALID_PATH
    assert change.selected is False
    assert change.error_message == "Cannot patch: File does not exist"
    assert change.status_summary == "Cannot patch: File does not exist", (
        "the dialog shows status_summary; it must not swallow the reason"
    )


def test_status_summary_is_a_single_line(project):
    """The tree cell is one line; a multi-line patch error must not break it."""
    change = sp.preview_changes_to_files(
        _patch("sub/a.py", "nowhere", "X"), str(project)
    )[0]

    assert "\n" in change.error_message, "the raw message spans lines"
    assert "\n" not in change.status_summary
    assert change.status_summary.startswith("Patch Failed:")


def test_unsafe_paths_are_rejected(project):
    for path in ["../escape.py", "/etc/passwd"]:
        change = sp.preview_changes_to_files(
            f'<patch file="{path}">\n<replace>\nx\n</replace>\n</patch>\n', str(project)
        )[0]
        assert change.change_type is sp.ChangeType.INVALID_PATH
        assert change.error_message == "Unsafe path"
        assert change.selected is False


def test_applying_an_invalid_change_reports_an_error(project):
    change = sp.FileChange(
        file_path="sub/a.py",
        content="",
        change_type=sp.ChangeType.INVALID_PATH,
        full_path=str(project / "sub" / "a.py"),
        error_message="Patch Failed: nope",
        selected=True,
    )

    results = sp.apply_selected_changes([change])

    assert results["success"] == []
    assert len(results["errors"]) == 1
    assert "Patch Failed: nope" in results["errors"][0]


def test_apply_changes_to_files_surfaces_a_failed_patch(project):
    """The no-preview path has no dialog, so the reason must reach the log."""
    blob = _patch("sub/a.py", "nowhere", "X")

    results = sp.apply_changes_to_files(blob, str(project))

    assert results["success"] == []
    assert len(results["errors"]) == 1
    assert "Patch Failed" in results["errors"][0]
