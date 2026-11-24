# -*- coding: utf-8 -*-
"""Audit the manuscript for provenance violations.

Checks:
  1. every CamelCase macro used in manuscript.tex is defined in
     manuscript/tables/key_numbers.tex (i.e. generated from a script product);
  2. every \\cite{key} exists in manuscript/references.bib;
  3. no hand-typed 4+-decimal performance number appears in the prose
     (constants such as 0.0005 in an equation are allowed via --allow);
  4. every table \\input{...} file exists and is non-empty;
  5. every numeric macro in the prose resolves to a value in numbers.json.

Writes manuscript/audit_report.txt and exits 1 if any check fails.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
TEX = (MS / "manuscript.tex").read_text(encoding="utf-8")
KEYS = (MS / "tables" / "key_numbers.tex").read_text(encoding="utf-8")
BIB = (MS / "references.bib").read_text(encoding="utf-8")
NUMBERS = json.loads((MS / "tables" / "numbers.json").read_text(encoding="utf-8"))

defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", KEYS))
# Standard LaTeX/math commands with an initial capital are not numeric macros.
STANDARD = {"Delta", "LaTeX", "TeX", "Gamma", "Theta", "Phi", "Omega",
            "mathrm", "mathcal", "text", "Rightarrow", "ldots"}
used = set(re.findall(r"\\([A-Z][A-Za-z]*)", TEX)) - STANDARD
missing_macros = sorted(used - defined)

cites = set()
for block in re.findall(r"\\cite[pt]?\{([^}]*)\}", TEX):
    cites.update(k.strip() for k in block.split(",") if k.strip())
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIB))
missing_cites = sorted(cites - bibkeys)

# Hand-typed performance-looking numbers. Strip comments and table inputs first.
prose = "\n".join(ln for ln in TEX.splitlines() if not ln.lstrip().startswith("%"))
hard = re.findall(r"(?<![\w.])(0\.\d{4,})(?![\w])", prose)
# Decimal constants that are method hyperparameters or criterion thresholds.
ALLOW = {"0.0005", "0.0050", "0.0001"}
hard = sorted(set(hard) - ALLOW)

inputs = re.findall(r"\\input\{([^}]*)\}", TEX)
input_paths = [p if p.endswith(".tex") else p + ".tex" for p in inputs]
missing_inputs = [p for p in input_paths if not (MS / p).is_file()
                  or (MS / p).stat().st_size == 0]

unresolved = sorted(m for m in used & defined if m not in NUMBERS)

lines = [
    "Manuscript provenance audit",
    "=" * 40,
    f"numeric macros defined : {len(defined)}",
    f"CamelCase macros used  : {len(used)}",
    f"citation keys used     : {len(cites)}",
    f"bibliography entries   : {len(bibkeys)}",
    f"table inputs           : {len(inputs)}",
    "",
    f"[1] undefined numeric macros : {len(missing_macros)}",
] + [f"    - {m}" for m in missing_macros] + [
    f"[2] missing bibliography keys : {len(missing_cites)}",
] + [f"    - {c}" for c in missing_cites] + [
    f"[3] hand-typed 4+-decimal numbers in prose : {len(hard)}",
] + [f"    - {h}" for h in hard] + [
    f"[4] missing/empty table inputs : {len(missing_inputs)}",
] + [f"    - {p}" for p in missing_inputs] + [
    f"[5] macros without a value in numbers.json : {len(unresolved)}",
] + [f"    - {m}" for m in unresolved] + [""]

failed = any([missing_macros, missing_cites, hard, missing_inputs, unresolved])
lines.append("RESULT: " + ("FAIL" if failed else "PASS"))
(MS / "audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
return_code = 1 if failed else 0
sys.exit(return_code)
