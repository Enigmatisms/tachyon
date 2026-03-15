"""Shared analysis pipeline — the core of Tachyon's analytical output.

Used by:
  - ``tachyon profile``  (E2E: profile → analyze → AI)
  - ``tachyon analyze``  (offline: load → analyze → AI)

Responsibilities:
  1. Merge duplicate kernel runs (same name + config → averaged metrics)
  2. Rule Engine analysis (6 built-in Analyzers)
  3. Terminal / Markdown / JSON rendering
  4. AI-enhanced analysis (optional, graceful fallback)
"""
