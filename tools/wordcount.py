# -*- coding: utf-8 -*-
"""Count the words that Elsevier counts for Computers & Geosciences.

The journal limits Research/Application Articles to 5,500 words at initial
submission (6,500 after revision) and allows a further 10% on top.  Abstract,
keywords, highlights, references, tables and figure captions are excluded from
the limit, so they are counted separately here and kept out of the body total.

Writes manuscript/wordcount.txt and exits 1 if the body exceeds the limit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
TEX = (MS / "manuscript.tex").read_text(encoding="utf-8")

LIMIT = 5500
ALLOWANCE = 1.10  # 10% tolerance granted to non-review articles

# Drop comment lines, table inputs and figure placeholders (captions excluded).
text = "\n".join(ln for ln in TEX.splitlines() if not ln.lstrip().startswith("%"))
text = re.sub(r"\\input\{tables/[^}]*\}", " ", text)
text = re.sub(r"\\figplaceholder\{[^}]*\}\{[^}]*\}", " ", text, flags=re.S)
text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", text, flags=re.S)
text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", text, flags=re.S)


def words(chunk: str) -> int:
    chunk = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", chunk)
    chunk = re.sub(r"[{}$\\]", " ", chunk)
    return len([w for w in chunk.split() if re.search(r"[A-Za-z0-9]", w)])


body = text.split(r"\end{frontmatter}", 1)[1]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", text, re.S)

# Back matter starts at the first unnumbered section after the conclusions.
back = re.search(r"\\section\*\{", body)
core = body[: back.start()] if back else body
backmatter = body[back.start():] if back else ""

body_words = words(core)
back_words = words(backmatter)
abs_words = words(abstract.group(1)) if abstract else 0
kw_words = words(keywords.group(1)) if keywords else 0

limit_with_allowance = int(LIMIT * ALLOWANCE)
ok = body_words <= limit_with_allowance

lines = [
    "Word count (Computers & Geosciences rule)",
    "=" * 44,
    f"body (Introduction .. Conclusions + Appendix) : {body_words}",
    f"  limit 5,500 (+10% = {limit_with_allowance})        : "
    + ("OK" if ok else f"OVER by {body_words - limit_with_allowance}"),
    f"abstract (limit 300)                          : {abs_words}",
    f"keywords (limit 6 items, counted words)       : {kw_words}",
    f"back matter (declarations, not counted)       : {back_words}",
    "",
    "Excluded from the body count: abstract, keywords, highlights, references,",
    "tables and figure captions, as specified in the guide for authors.",
    "RESULT: " + ("PASS" if ok and abs_words <= 300 else "FAIL"),
]
(MS / "wordcount.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(0 if ok and abs_words <= 300 else 1)
