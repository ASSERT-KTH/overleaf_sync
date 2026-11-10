"""
Exhaustive tests for compute_ot_ops and apply_sharejs_ops.

The invariant: applying compute_ot_ops(old, new) to old one-op-at-a-time
(simulating the Overleaf server receiving each applyOtUpdate in sequence)
must yield exactly new.
"""
from __future__ import annotations

import pytest
from overleaf_sync import compute_ot_ops, apply_sharejs_ops


def server_apply(content: str, ops: list[dict]) -> str:
    """Simulate Overleaf server applying ops one by one in order."""
    for op in ops:
        p = op["p"]
        if "i" in op:
            content = content[:p] + op["i"] + content[p:]
        elif "d" in op:
            content = content[:p] + content[p + len(op["d"]):]
    return content


def roundtrip(old: str, new: str):
    """Assert that ops computed from old→new reproduce new when applied by server."""
    ops = compute_ot_ops(old, new)
    result = server_apply(old, ops)
    assert result == new, (
        f"FAIL: expected {new!r}, got {result!r}\n"
        f"  old={old!r}\n"
        f"  ops={ops}"
    )


# ── pure inserts ────────────────────────────────────────────────────────────

def test_insert_at_beginning():
    roundtrip("world", "hello world")

def test_insert_at_end():
    roundtrip("hello", "hello world")

def test_insert_in_middle():
    roundtrip("helloworld", "hello beautiful world")

def test_insert_empty_old():
    roundtrip("", "hello")

def test_insert_newline():
    roundtrip("line1\nline3", "line1\nline2\nline3")


# ── pure deletes ─────────────────────────────────────────────────────────────

def test_delete_at_beginning():
    roundtrip("hello world", "world")

def test_delete_at_end():
    roundtrip("hello world", "hello")

def test_delete_in_middle():
    roundtrip("hello beautiful world", "hello world")

def test_delete_all():
    roundtrip("hello", "")

def test_delete_newline():
    roundtrip("line1\nline2\nline3", "line1\nline3")


# ── replace (delete + insert at same position) ───────────────────────────────
# This is the class of operations that triggers the ordering bug.

def test_replace_at_beginning():
    roundtrip("Hello World", "Hi World")

def test_replace_at_end():
    roundtrip("Hello World", "Hello Earth")

def test_replace_in_middle():
    roundtrip("Hello World", "Hello Beautiful World")

def test_replace_longer_with_shorter():
    roundtrip("Hello World", "Hi W")

def test_replace_shorter_with_longer():
    roundtrip("Hi W", "Hello World")

def test_replace_whole_word():
    roundtrip("foo bar baz", "foo qux baz")

def test_replace_latex_command():
    old = r"\section{Introduction}"
    new = r"\section{Related Work}"
    roundtrip(old, new)

def test_replace_multiline():
    old = "line1\nold content here\nline3"
    new = "line1\nnew content here\nline3"
    roundtrip(old, new)

def test_replace_grows():
    roundtrip("Hello World", "Hi Earth and Beyond")

def test_replace_shrinks():
    roundtrip("Hi Earth and Beyond", "Hello World")


# ── realistic LaTeX edits ────────────────────────────────────────────────────

SAMPLE = r"""\documentclass{article}
\begin{document}
\section{Introduction}
This is an introduction.
\end{document}
"""

def test_latex_change_section_title():
    new = SAMPLE.replace("Introduction", "Background")
    roundtrip(SAMPLE, new)

def test_latex_add_paragraph():
    new = SAMPLE.replace(
        "This is an introduction.",
        "This is an introduction.\n\nMore text here."
    )
    roundtrip(SAMPLE, new)

def test_latex_delete_sentence():
    new = SAMPLE.replace("\nThis is an introduction.", "")
    roundtrip(SAMPLE, new)

def test_latex_replace_sentence():
    new = SAMPLE.replace("This is an introduction.", "This section motivates the work.")
    roundtrip(SAMPLE, new)

def test_latex_add_usepackage():
    new = SAMPLE.replace(r"\begin{document}", r"\usepackage{amsmath}" + "\n" + r"\begin{document}")
    roundtrip(SAMPLE, new)


# ── edge cases ───────────────────────────────────────────────────────────────

def test_identical():
    assert compute_ot_ops("abc", "abc") == []

def test_single_char_replace():
    roundtrip("a", "b")

def test_single_char_insert():
    roundtrip("", "x")

def test_single_char_delete():
    roundtrip("x", "")

def test_trailing_newline_add():
    roundtrip("hello", "hello\n")

def test_trailing_newline_remove():
    roundtrip("hello\n", "hello")

def test_only_newlines():
    roundtrip("\n\n", "\n\n\n")

def test_unicode():
    roundtrip("café", "café latte")

def test_unicode_replace():
    roundtrip("naïve", "naive")


# ── git backup comparison ────────────────────────────────────────────────────

def test_git_backup_roundtrip():
    """Verify ops roundtrip on the actual project file from git backup."""
    import subprocess, pathlib
    backup = pathlib.Path.home() / "workspace/overleaf_backup/69d7ba24e392a5121b52a73f/main.tex"
    if not backup.exists():
        pytest.skip("git backup not present")
    content = backup.read_text(encoding="utf-8")
    # Simulate a realistic edit: change a word in the content
    if "Introduction" in content:
        edited = content.replace("Introduction", "Background", 1)
        roundtrip(content, edited)
    # Simulate appending a line
    roundtrip(content, content + "\n% added line\n")
    # Simulate deleting first line
    lines = content.split("\n")
    if len(lines) > 1:
        roundtrip(content, "\n".join(lines[1:]))


# ── no replace of unchanged text ─────────────────────────────────────────────

def _max_op_len(ops: list[dict]) -> int:
    return max(len(op.get("i", "")) + len(op.get("d", "")) for op in ops)


def test_distant_edits_give_small_ops():
    """Two 1-char edits at both ends must not delete+reinsert the middle."""
    old = "A" * 50 + "\n" + "x\n" * 5000 + "B" * 50
    new = "Z" + old[1:-1] + "Y"
    ops = compute_ot_ops(old, new)
    assert server_apply(old, ops) == new
    assert _max_op_len(ops) <= 2, ops


def test_many_scattered_line_edits_stay_local():
    lines = [f"line {k} with some text\n" for k in range(3000)]
    new_lines = list(lines)
    for k in range(0, 3000, 300):
        new_lines[k] = new_lines[k].replace("some", "other")
    del new_lines[1500]
    new_lines.insert(2000, "brand new line\n")
    old, new = "".join(lines), "".join(new_lines)
    ops = compute_ot_ops(old, new)
    assert server_apply(old, ops) == new
    assert _max_op_len(ops) <= len("line 1500 with some text\n"), ops


def test_ops_descending_positions():
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nC\nd\nE\n"
    ops = compute_ot_ops(old, new)
    ps = [op["p"] for op in ops]
    assert ps == sorted(ps, reverse=True)
    assert server_apply(old, ops) == new


def test_large_near_identical_is_fast():
    import time
    old = "".join(f"\\item paragraph {k} lorem ipsum dolor\n" for k in range(20000))
    new = old.replace("paragraph 7 ", "PARAGRAPH 7 ").replace("paragraph 19999", "p19999")
    t = time.monotonic()
    ops = compute_ot_ops(old, new)
    assert time.monotonic() - t < 2.0
    assert server_apply(old, ops) == new


# ── word-level splitting of changed paragraphs ──────────────────────────────

PARA = ("We present a novel approach to program repair that leverages large "
        "language models and a feedback loop with the test suite, achieving "
        "state of the art results on Defects4J and QuixBugs benchmarks.\n")


def test_reflowed_paragraph_keeps_words():
    """Rewrapping lines changes only whitespace: no word is deleted."""
    old = "intro\n" + PARA + "outro\n"
    reflowed = PARA.replace(" leverages large ", " leverages large\n").replace(" suite, ", " suite,\n")
    new = "intro\n" + reflowed + "outro\n"
    ops = compute_ot_ops(old, new)
    assert server_apply(old, ops) == new
    assert all(not op.get("d", "").strip() for op in ops), ops


def test_scattered_word_edits_in_one_line():
    old = "intro\n" + PARA + "outro\n"
    new = old.replace("novel", "new").replace("Defects4J", "Defects4J v2").replace("loop", "cycle")
    ops = compute_ot_ops(old, new)
    assert server_apply(old, ops) == new
    assert _max_op_len(ops) <= len("loop") + len("cycle"), ops


def test_full_rewrite_stays_one_replace():
    old = "intro\n" + PARA + "outro\n"
    new = "intro\nCompletely different sentence about the weather in Stockholm.\noutro\n"
    ops = compute_ot_ops(old, new)
    assert server_apply(old, ops) == new
    assert len(ops) == 2, ops


@pytest.mark.parametrize("seed", range(200))
def test_random_word_edits_roundtrip(seed):
    import random
    rnd = random.Random(seed)
    words = ["a", "b", "foo", " ", "\n", ",", "\\cite{x}", "é", "  "]
    old = "".join(rnd.choice(words) for _ in range(rnd.randint(0, 60)))
    new = list(_TOKEN_SPLIT(old))
    for _ in range(rnd.randint(1, 5)):
        k = rnd.randint(0, len(new))
        if new and rnd.random() < 0.5:
            del new[min(k, len(new) - 1)]
        else:
            new.insert(k, rnd.choice(words))
    roundtrip(old, "".join(new))


def _TOKEN_SPLIT(s):
    import re
    return re.findall(r"\w+|\s+|[^\w\s]", s)
