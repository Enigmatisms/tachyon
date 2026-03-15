"""Kernel name filtering — shared utility used across all CLI commands.

Supports three matching modes:
  - glob:    fnmatch patterns (e.g. ``matmul*``, ``*reduce*``)
  - regex:   Python re patterns (e.g. ``matmul_kernel_\\d+``)
  - substr:  Case-insensitive substring match

For NCU ``--kernel-name`` (regex pass-through), use ``to_ncu_regex()``.
"""
from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tachyon.models.kernel import KernelReport


def match_kernel_name(
    kernel_name: str,
    demangled_name: str | None,
    pattern: str,
) -> bool:
    """Check if a kernel matches a filter pattern.

    Tries in order: fnmatch glob, regex, then case-insensitive substring.
    Checks both the mangled ``kernel_name`` and ``demangled_name``.
    """
    names = [kernel_name]
    if demangled_name:
        names.append(demangled_name)

    for name in names:
        # 1. fnmatch glob
        if fnmatch.fnmatch(name, pattern):
            return True
        # 2. regex
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            pass
        # 3. case-insensitive substring
        if pattern.lower() in name.lower():
            return True

    return False


def filter_kernels(
    kernels: list[KernelReport],
    pattern: str,
) -> list[KernelReport]:
    """Filter a list of KernelReport objects by name pattern.

    Returns matching kernels preserving original order.
    """
    return [
        k for k in kernels
        if match_kernel_name(k.kernel_name, k.demangled_name, pattern)
    ]


def to_ncu_regex(pattern: str) -> str:
    """Convert a user-supplied pattern to an NCU-compatible kernel-name regex.

    NCU's ``--kernel-name`` expects a regex, not a glob.
    If the pattern looks like a glob (contains ``*`` or ``?`` but no regex chars),
    convert it. Otherwise pass through as-is.
    """
    # If it looks like a glob pattern (has * or ? but no regex anchors/groups)
    glob_chars = {"*", "?"}
    regex_only_chars = {"(", ")", "|", "+", "{", "}"}

    has_glob = any(c in pattern for c in glob_chars)
    has_regex = any(c in pattern for c in regex_only_chars) or pattern.startswith("^")

    if has_glob and not has_regex:
        # Convert glob to regex: * -> .*, ? -> .
        regex = fnmatch.translate(pattern)
        # fnmatch.translate appends \Z — strip it since ncu does partial match
        regex = regex.removesuffix(r"\Z")
        # Also strip leading (?s:) wrapper if present
        if regex.startswith("(?s:") and regex.endswith(")"):
            regex = regex[4:-1]
        return regex

    return pattern
