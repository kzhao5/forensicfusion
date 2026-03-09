from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def make_cvpr_table(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    caption: str,
    label: str,
    align: str,
    size: str = "\\scriptsize",
    tabcolsep_pt: float = 3.5,
    resize_to_linewidth: bool = False,
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(size)
    lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep_pt}pt}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize_to_linewidth:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(_fmt(row[c]) for c in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize_to_linewidth:
        lines.append(r"}")
    lines.append(r"\vspace{-2mm}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
