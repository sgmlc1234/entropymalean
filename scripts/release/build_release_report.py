"""Render the release and its audit trail as a LaTeX article.

The gallery this sits beside shows the accepted rows and their Lean. That answers
"what did you release" and not "how do you know it is any good", which is the
question actually asked. Three things were missing and are the point of this
renderer:

  * the judge's reasoning, in full, including the passes that disagreed;
  * the rows that did not make it, with the reason attached to each;
  * who decided what -- Lean, a hash, or a model -- and how often the judge
    agrees with itself, which is the honest ceiling on any verdict it gives.

The site carries the same material as a browsable page; this is the form that
goes into a paper appendix. Both are built from `eml1_release.jsonl` and
`eml1_rejected.jsonl`, so neither can drift from the other.

Compile with lualatex or xelatex -- the Lean in it is Unicode throughout and
pdflatex cannot set it.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List

#: Pass identifiers as a reader meets them. `rejudge_1` is a field name; what
#: matters on the page is that two of these ran after the fact, with more in
#: front of them than the one that ran while the row was being written.
PASS_NAMES = {
    "at_generation": "At generation",
    "rejudge_1": "Re-judged, first pass",
    "rejudge_2": "Re-judged, second pass",
}

#: What a judge was shown, as recorded. The field arrives as a stringified list
#: on some rows and a real list on others, which is why this parses rather than
#: formats.
def _saw(value: Any) -> str:
    if isinstance(value, str):
        parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value)
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        return ""
    return ", ".join(tex(p.replace("_", " ")) for p in parts) if parts else ""


PIPELINE = [
    ("statement_type_check", "Lean", "The statement elaborates under Mathlib with \\texttt{autoImplicit false}, so no free variable is silently invented."),
    ("proof_accepted", "Lean", "\\texttt{lake env lean} accepts the proof and it contains no \\texttt{sorry}."),
    ("axiom_audit", "Lean", "\\texttt{\\#print axioms} lists what the proof depends on; anything outside Lean's three standard axioms fails the row."),
    ("vacuity", "Lean", "The hypotheses are jointly satisfiable, so the theorem is not true merely by having no instances."),
    ("dead_hypotheses", "Lean", "Each hypothesis whose name the proof never writes is removed and the file recompiled; a proof that still closes proves the hypothesis was dead. Repeated to a fixpoint, because removing one can make another dead."),
    ("corpus_dedup", "hash", "The statement, alpha-normalised, is absent from every earlier campaign and from this one."),
    ("redundancy", "Lean", "Whether one parent already carries the child: for a crossover, that one parent proves it alone; for a mutation, that the parent gives it directly. Finding such a proof is decisive; failing to find one is not, and the inline probe runs without a prover so it reaches only what the tactic ladder can close."),
    ("hypothesis_preservation", "parser", "Silent mutations only. The parent's and child's binders are compared after alpha-normalisation. It cannot tell a hypothesis that was re-encoded from one that was dropped, so a finding is read, not obeyed."),
    ("comparator", "Lean", "An independent runner rebuilds the statement from the row alone, compares it against the submitted proof with lean4export, checks the axiom allowlist, and replays the proof through the Lean kernel. This is what separates \\texttt{proof\\_checked} from \\texttt{kernel\\_replayed}."),
    ("goal_roundtrip", "model", "Lean's elaborated goal goes to an informalizer that sees no prose, and a separate judge compares its output with the stated problem. Neither model sees the other's input."),
]

FAILURE_GLOSS = {
    "recall": "solvable by recalling the parent",
    "parallel": "the parents are proved separately and joined at the end",
    "parallel_crossover": "the parents are proved separately and joined at the end",
    "decoration": "the change is cosmetic; the parent's proof skeleton still closes it",
    "universal_supplier": "one parent proves the child on its own, so the other contributes nothing",
    "constant_supplier": "a parent is used only as a constant, not as an argument",
    "repeated_device": "the same device as a sibling already kept from these parents",
    "weakening": "the child is weaker than its parent",
    "conjunction": "two unrelated claims stapled together",
    "exists_unique_wrapper": "an existing result rewrapped as an existence-and-uniqueness claim",
    "tractability_bound": "the bound makes the statement true for reasons unrelated to the mathematics",
}

_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

#: The judge quotes Lean in its reasoning, so mathematical symbols land in
#: running prose where the text face does not carry them: one build dropped 899
#: characters across 35 code points, and every `≤` and `∣` in the reasoning
#: became a blank in exactly the sentences the report exists to show.
#:
#: They are mapped here rather than with `newunicodechar` in the preamble,
#: because that makes the characters active *everywhere* -- and they appear
#: 3,100 times inside the verbatim Lean blocks against 261 times in prose, where
#: activating them would both cost a macro expansion each and render Lean source
#: through math mode. Verbatim never passes through this function.
_MATH_IN_PROSE = {
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≡": r"\equiv", "≃": r"\simeq",
    "∣": r"\mid", "∤": r"\nmid", "‖": r"\Vert", "→": r"\to", "↦": r"\mapsto",
    "∈": r"\in", "∉": r"\notin", "⊆": r"\subseteq", "∩": r"\cap", "∪": r"\cup",
    "⋂": r"\bigcap", "⋃": r"\bigcup", "⊓": r"\sqcap", "⨅": r"\bigsqcap",
    "⊤": r"\top", "⊥": r"\bot", "⊢": r"\vdash", "∘": r"\circ",
    "√": r"\surd", "∞": r"\infty", "∑": r"\sum", "∏": r"\prod",
    "∀": r"\forall", "∃": r"\exists", "∧": r"\wedge", "∨": r"\vee",
    "∅": r"\emptyset", "⟨": r"\langle", "⟩": r"\rangle", "⧸": "/",
    "𝓝": r"\mathcal{N}", "ₗ": r"_{\mathrm{l}}", "ₙ": r"_{\mathrm{n}}",
    "×": r"\times", "∼": r"\sim", "≅": r"\cong", "⁻": r"^{-}", "∖": r"\setminus",
    # Added when the ProofNet corpus grew: 27 characters were dropped silently
    # from the built PDF because the prose font has no glyph for them. A symbol
    # missing from this table does not fail the build -- it disappears from the
    # page, so `Missing character` in the log is the only evidence.
    "↔": r"\leftrightarrow", "⟪": r"\langle\!\langle", "⟫": r"\rangle\!\rangle",
    "ᵢ": r"_{i}", "ⱼ": r"_{j}", "ₖ": r"_{k}", "ₘ": r"_{m}", "ₚ": r"_{p}",
    "ₛ": r"_{s}", "ₜ": r"_{t}", "↑": r"\uparrow", "↓": r"\downarrow",
    "⇑": r"\Uparrow", "≫": r"\gg", "≪": r"\ll", "∫": r"\int", "∂": r"\partial",
    "∇": r"\nabla", "⊕": r"\oplus", "⊗": r"\otimes", "≺": r"\prec", "≻": r"\succ",
}


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def consistency(run1: Path, run2: Path) -> Dict[str, Any]:
    a = {r["problem_id"]: r for r in json.loads(run1.read_text(encoding="utf-8"))}
    b = {r["problem_id"]: r for r in json.loads(run2.read_text(encoding="utf-8"))}
    ids = [k for k in a if k in b]
    kept = lambda r: r["new_verdict"] == "keep"
    both_reject = [k for k in ids if not kept(a[k]) and not kept(b[k])]
    return {
        "n": len(ids),
        "verdict_agreement": sum(1 for k in ids if kept(a[k]) == kept(b[k])),
        "quality_agreement": sum(1 for k in ids if a[k]["new_quality"] == b[k]["new_quality"]),
        "both_reject": len(both_reject),
        "failure_agreement": sum(1 for k in both_reject if a[k]["new_failure"] == b[k]["new_failure"]),
        "matrix": collections.Counter((a[k]["new_quality"], b[k]["new_quality"]) for k in ids),
    }


def tex(text: Any) -> str:
    """Escape prose for LaTeX.

    Latin text passes through untouched -- fontspec sets it. A mathematical
    symbol the text face lacks is sent to math mode instead of being dropped.
    """
    out = []
    for char in str(text or ""):
        if char in _SPECIALS:
            out.append(_SPECIALS[char])
        elif char in _MATH_IN_PROSE:
            out.append(f"\\ensuremath{{{_MATH_IN_PROSE[char]}}}")
        else:
            out.append(char)
    return "".join(out)


_MATH_SPAN = re.compile(r"\$[^$]*\$")


def prose(text: Any) -> str:
    r"""Escape a statement while leaving its mathematics alone.

    98 of the 161 parent statements arrive carrying LaTeX already -- the seed
    sheets write `$y = ax^2 + bx + c$` -- and the ordinary escaper turns every
    one of those into `\$y = ax\^{}2 ...\$`, which is how the most important
    line on the card became unreadable. Spans between dollars are passed
    through untouched; everything outside them is escaped as usual.
    """
    raw = str(text or "")
    out, index = [], 0
    for match in _MATH_SPAN.finditer(raw):
        out.append(tex(raw[index:match.start()]))
        out.append(match.group(0))
        index = match.end()
    out.append(tex(raw[index:]))
    return "".join(out)


def mono(text: Any) -> str:
    """An identifier in the monospace face.

    A crossover id concatenates both parents and every generation, runs past a
    hundred characters, and contains no space to break at. `seqsplit` breaks it
    anywhere rather than letting it run into the margin.
    """
    return r"\texttt{\seqsplit{" + tex(text) + "}}"


def verbatim(text: Any, environment: str = "leanbox", title: str = "") -> str:
    body = str(text or "").replace("\r", "").strip()
    return f"\\begin{{{environment}}}{{{title}}}\n{body}\n\\end{{{environment}}}"


def longtable(head: List[str], spec: str, rows: List[List[str]]) -> str:
    out = [r"\begin{longtable}{" + spec + "}", r"\toprule",
           " & ".join(rf"\textbf{{{h}}}" for h in head) + r" \\", r"\midrule", r"\endhead"]
    for row in rows:
        out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    return "\n".join(out)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{fontspec}
\usepackage{amsmath}
\usepackage{unicode-math}
\IfFontExistsTF{STIX Two Text}{\setmainfont{STIX Two Text}}{\setmainfont{Palatino}}
\IfFontExistsTF{Helvetica Neue}{\setsansfont{Helvetica Neue}}{\setsansfont{Arial}}
% JuliaMono over Menlo for anything that sets Lean. Menlo is missing the
% symbols Lean actually uses -- one build dropped 678 characters, 110 of them
% `∣`, so `3 ∣ n` printed as `3  n` in the released listings. JuliaMono exists
% for this alphabet. Menlo stays as the fallback so the document still builds
% where JuliaMono is absent, and the log then says what went missing.
\IfFontExistsTF{JuliaMono}{\setmonofont{JuliaMono}[Scale=0.78]}{%
  \IfFontExistsTF{Menlo}{\setmonofont{Menlo}[Scale=0.80]}{\setmonofont{Latin Modern Mono}[Scale=0.80]}}
\IfFontExistsTF{STIX Two Math}{\setmathfont{STIX Two Math}}{\setmathfont{Latin Modern Math}}
\IfFontExistsTF{JuliaMono}{\newfontfamily\leanfont{JuliaMono}[Scale=0.74]}{%
  \IfFontExistsTF{Menlo}{\newfontfamily\leanfont{Menlo}[Scale=0.76]}{\newfontfamily\leanfont{Latin Modern Mono}[Scale=0.76]}}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fvextra}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{seqsplit}
\usepackage[most]{tcolorbox}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!50!black}
\definecolor{emgborder}{HTML}{CBD5E1}
\definecolor{emggray}{HTML}{F7F7F7}
\definecolor{emggreen}{HTML}{1F6F50}
\definecolor{emglightgreen}{HTML}{EFF8F1}
\definecolor{emgred}{HTML}{A3323C}
\definecolor{emglightred}{HTML}{FCEFF0}
\tcbset{boxrule=0.45pt, arc=1.5mm, left=1.3mm, right=1.3mm, top=1mm, bottom=1mm,
        before skip=0.5em, after skip=0.5em}
\newtcolorbox{reviewbox}[1]{enhanced, breakable, colback=emggray, colframe=emgborder,
  title={#1}, coltitle=black, fonttitle=\sffamily\bfseries}
\newtcolorbox{problembox}[1]{enhanced, breakable, colback=white, colframe=black!55,
  title={#1}, coltitle=black, colbacktitle=black!8, fonttitle=\sffamily\bfseries,
  fontupper=\large, left=2.5mm, right=2.5mm, top=1.8mm, bottom=1.8mm}
\newtcolorbox{keepbox}[1]{enhanced, breakable, colback=emglightgreen, colframe=emggreen!65!black,
  title={#1}, coltitle=white, colbacktitle=emggreen!70!black, fonttitle=\sffamily\bfseries}
\newtcolorbox{rejectbox}[1]{enhanced, breakable, colback=emglightred, colframe=emgred!65!black,
  title={#1}, coltitle=white, colbacktitle=emgred!70!black, fonttitle=\sffamily\bfseries}

% Every row carries three independent judgements, and they were being set as one
% run of bold-then-italic paragraphs inside a single box -- so a reader could not
% see where one judge stopped and the next began, nor that the three had been
% given different information. Each pass now gets its own panel: a rule in the
% colour of its verdict, a header naming the judge and what it was shown, and the
% reasoning as body text.
\definecolor{emgslate}{HTML}{475569}
% The header lives inside the panel rather than in tcolorbox's boxed title: an
% attached title floats above the frame, so the coloured rule began below the
% judge's name and the two read as separate objects. One box, first line is the
% header, and the rule runs the full height of the judgement it marks.
\newtcolorbox{judgepanel}[1]{enhanced, breakable, sharp corners,
  colback=white, colframe=white, boxrule=0pt,
  borderline west={2.2pt}{0pt}{#1},
  left=3mm, right=2mm, top=1.2mm, bottom=1.2mm,
  before skip=0.45em, after skip=0.45em}
% A verdict worn as a label rather than spelled out mid-sentence, so the three
% panels can be compared down the page without reading them.
\newcommand{\verdictchip}[2]{{\small\sffamily\bfseries\textcolor{#1}{#2}}}
\newcommand{\judgehead}[2]{{\small\sffamily\bfseries\textcolor{#1}{#2}}\par}
\newcommand{\judgesaw}[1]{{\footnotesize\sffamily\textcolor{emgslate}{#1}\par\vspace{0.6mm}}}
\newenvironment{leanbox}[1]{\VerbatimEnvironment
  \begin{tcolorbox}[enhanced, breakable, colback=black!2, colframe=black!35, title={#1},
    coltitle=black, colbacktitle=black!12, fonttitle=\sffamily\bfseries]
  \begin{Verbatim}[breaklines=true, breakanywhere=true, fontsize=\footnotesize,
    formatcom=\leanfont, tabsize=2]}{\end{Verbatim}\end{tcolorbox}}
\setlist[itemize]{topsep=0.2em, parsep=0pt, leftmargin=1.1em}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\setcounter{tocdepth}{1}
\begin{document}
\title{TITLEPLACEHOLDER}
\date{}
\maketitle
"""


#: `Prove the theorem <id>.` is what the seed CSVs carry where the source
#: benchmark ships no informal statement -- 109 of 161 parent entries in the
#: release. It is not a description, and printing it as one tells a reader the
#: problem has been explained when it has not.
_PLACEHOLDER = re.compile(r"^\s*Prove the theorem\b|^\s*$")


def _describes(text: Any) -> bool:
    return bool(str(text or "").strip()) and not _PLACEHOLDER.match(str(text))


def _theorem_name(row: dict) -> str:
    """The declared name, which is what a reader actually cites."""
    match = re.search(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)", str(row.get("formal_statement") or ""))
    return match.group(1) if match else ""


def _card(number: int, row: dict) -> str:
    """One problem, ordered so a reader learns what it is before how it was made.

    The first version led with the identifier, the operator, the lineage depth,
    the certificate level, the slot's internal goal and both parents' Lean, and
    reached the prose statement fifteen lines in. Everything above it is
    machinery -- true, and useless to someone trying to find out what the
    theorem says. The statement now comes first and the provenance last, and the
    parents carry their own prose rather than Lean alone.
    """
    out: List[str] = []
    out.append(r"\clearpage")
    theorem = _theorem_name(row) or str(row["problem_id"])[-32:]
    out.append(rf"\subsection{{{number:03d}. \texttt{{{tex(theorem)}}}}}")

    # What the problem says, before anything about where it came from.
    if _describes(row.get("statement")):
        out.append(r"\begin{problembox}{The problem}" + "\n"
                   + prose(row["statement"]) + "\n" + r"\end{problembox}")
    out.append(verbatim(row.get("formal_statement"), title="Lean statement"))

    for index, parent in enumerate(row.get("parents") or [], 1):
        label = "Parent" if len(row.get("parents") or []) == 1 else f"Parent {index}"
        if _describes(parent.get("statement")):
            out.append(rf"\textbf{{{label}.}} {prose(parent['statement'])}\par")
        else:
            # No informal statement exists for this seed in the source benchmark.
            # Say that, rather than printing the stand-in as though it described
            # something.
            out.append(rf"\textbf{{{label}.}} {mono(parent['parent_id'])} "
                       r"\textit{(no informal statement in the source benchmark)}\par")
        if parent.get("formal_statement"):
            out.append(verbatim(parent["formal_statement"], title=f"{label} in Lean"))
        contribution = str(parent.get("contribution") or "")
        # Reserve slots fill this with their own template — "reserve mutation
        # must add a semantic proof obligation" — which describes the slot, not
        # the parent. Printing it as though it said something about this problem
        # is worse than leaving it out.
        if contribution and not contribution.lstrip().lower().startswith("reserve"):
            out.append(rf"\textit{{What it contributes.}} {tex(contribution)}\par")

    body = [r"Every judgement this row received. Admission required \texttt{strong} from both "
            r"passes of the current judge; the generation-time judge saw neither the plan, nor "
            r"the siblings already kept from these parents, nor which tier the slot asked for, "
            r"and is recorded rather than counted."]
    for record in row["review"]["passes"]:
        colour = "emggreen" if record.get("verdict") == "keep" else "emgred"
        header = PASS_NAMES.get(str(record.get("pass") or ""), tex(record.get("pass") or ""))
        chip = rf"\verdictchip{{{colour}}}{{{tex(str(record.get('quality') or '')).upper()}}}"
        if record.get("failure"):
            chip += rf" \verdictchip{{emgred}}{{/ {tex(record['failure'])}}}"
        body.append(rf"\begin{{judgepanel}}{{{colour}}}")
        body.append(rf"\judgehead{{{colour}}}{{{header} \quad {chip}}}")
        # What the judge was shown is the reason two of these passes are not
        # interchangeable, so it belongs beside the verdict rather than in a
        # caveat further up the page.
        seen = _saw(record.get("saw"))
        provenance = [p for p in (
            rf"model \texttt{{{tex(record['model'])}}}" if record.get("model") else "",
            rf"saw {seen}" if seen else "",
        ) if p]
        if provenance:
            body.append(rf"\judgesaw{{{' \\; · \\; '.join(provenance)}}}")
        if record.get("reason"):
            # Set roman. These run to four or five sentences of close argument
            # and italic across that length is tiring to read; the coloured rule
            # already marks the text as the judge's rather than ours, which is
            # the job the italic was doing.
            body.append(rf"{tex(record['reason'])}\par")
        body.append(r"\end{judgepanel}")
    out.append(r"\begin{keepbox}{Review}" + "\n" + "\n".join(body) + "\n" + r"\end{keepbox}")

    out.append(longtable(
        ["Check", "By", "Result"], r"@{}p{0.24\linewidth}p{0.07\linewidth}p{0.60\linewidth}@{}",
        [[rf"\texttt{{{tex(name)}}}", tex((row['checks'].get(name) or {}).get('by')),
          tex((row["checks"].get(name) or {}).get("result") or "not run")]
         for name, _, _ in PIPELINE
         # A crossover probe on a mutation row is not a check that failed to
         # run; it is a check that does not apply, and listing it as "not run"
         # invites the reader to count it as missing.
         if (row["checks"].get(name) or {}).get("applies", True) is not False]))

    goal = row["checks"].get("goal_roundtrip") or {}
    if goal.get("ran") and goal.get("elaborated_goal"):
        out.append(verbatim(goal["elaborated_goal"], title="Goal round-trip: what Lean elaborated"))
        out.append(rf"\textbf{{Read back by a model that saw only the Lean.}} "
                   rf"\textit{{{tex(goal['read_back_as'])}}}\par")

    out.append(verbatim(row.get("lean_code"), title="Lean certificate"))

    plan = row.get("plan") or {}
    provenance = [
        rf"\textbf{{Operator.}} \texttt{{{tex(row['op_type'])}}} / \texttt{{{tex(row['operator_variant'])}}}",
        rf"\textbf{{Lineage depth.}} {row['lineage_depth']}",
        rf"\textbf{{Certificate.}} \texttt{{{tex(row['certificate']['level'])}}}",
    ]
    out.append(" \\quad ".join(provenance) + r"\par")
    out.append(rf"\textbf{{Problem id.}} {mono(row['problem_id'])}\par")
    if plan.get("fusion_mechanism"):
        out.append(rf"\textbf{{Fusion mechanism.}} \texttt{{{tex(plan['fusion_mechanism'])}}} "
                   rf"{tex(plan.get('fusion_goal') or '')}\par")
    certificate = row["certificate"]
    out.append(
        rf"toolchain \texttt{{{tex(certificate['lean_toolchain'])}}}, "
        rf"mathlib \texttt{{{tex((certificate['mathlib_revision'] or '')[:12])}}}, "
        rf"axioms \texttt{{{tex(', '.join(certificate['axioms']) or 'none recorded')}}}, "
        rf"fingerprint \texttt{{{tex(row['hashes']['dedup_fingerprint'][:16])}}}\par"
    )
    return "\n\n".join(out)


def render_tex(accepted: List[dict], held: List[dict], stats: Dict[str, Any],
               family: str = "") -> str:
    both = [r for r in held if r["admission"]["why_not"] == "rejected by both passes"]
    considered = len(accepted) + len(held)
    title = ("EML-1 release: the corpus, the judgements, and what was thrown away"
             if not family else
             f"EML-1 release, {family}: the corpus, the judgements, and what was thrown away")
    out: List[str] = [PREAMBLE.replace("TITLEPLACEHOLDER", title)]

    out.append(r"\begin{reviewbox}{What this is}" + "\n"
               + rf"{len(accepted)} generated theorems, each with a complete Lean proof. Every one "
                 r"carries the reasoning behind each judgement it received, including the passes that "
                 r"disagreed, and the result of every mechanical check. The rows that did not make it "
                 r"are summarised here and listed in full in \texttt{eml\_v1\_rejected.jsonl}, because a "
                 r"corpus that shows only its survivors cannot be checked." + "\n"
               + r"\par\textbf{Canonical artifact.} The Lean statements and certificates are the source "
                 r"of truth; this document is a review rendering." + "\n" + r"\end{reviewbox}")
    out.append(r"\tableofcontents")

    out.append(r"\clearpage\section{What was thrown away}")
    out.append(rf"{considered} certified rows were considered; {len(accepted)} were admitted. "
               r"Missing the bar happens three ways, and they are not the same finding: a row both "
               r"passes reject is a defect the pipeline caught, a row they split on is a fact about "
               r"the judge, and a row both keep but neither calls strong is simply below the bar this "
               r"release sets.\par")
    out.append(longtable(["Outcome", "Rows"], r"@{}p{0.72\linewidth}r@{}",
                         [[r"admitted \textemdash{} \texttt{strong} from both passes", str(len(accepted))]]
                         + [[tex(why), str(count)] for why, count in
                            collections.Counter(r["admission"]["why_not"] for r in held).most_common()]))
    out.append(rf"Among the {len(both)} rows both passes rejected, the named failure modes were:\par")
    out.append(longtable(["Failure", "Rows", "What it means"],
                         r"@{}p{0.24\linewidth}rp{0.60\linewidth}@{}",
                         [[rf"\texttt{{{tex(failure)}}}", str(count), tex(FAILURE_GLOSS.get(failure, ""))]
                          for failure, count in collections.Counter(
                              (r["review"]["passes"][-1].get("failure") or "unlabelled") for r in both
                          ).most_common()]))

    out.append(r"\clearpage\section{How a row got here}")
    out.append(r"Each gate, and what kind of thing settles it. Lean decides the questions that have an "
               r"answer; a model is asked only the questions that do not.\par")
    # Coverage is over the rows a check applies to. Reported corpus-wide,
    # `redundancy` read 7/135 -- which looks like a gap and is in fact the
    # number of crossovers, since a one-parent mutation cannot be asked whether
    # it follows from one parent alone.
    coverage_rows = []
    scoped = []
    for name, by, description in PIPELINE:
        scope = [r for r in accepted
                 if (r["checks"].get(name) or {}).get("applies", True) is not False]
        ran = sum(1 for r in scope if (r["checks"].get(name) or {}).get("ran"))
        if not scope:
            # No row in this release uses the operator the check belongs to.
            scoped.append(name)
            count = "not applicable here"
        elif len(scope) != len(accepted):
            scoped.append(name)
            count = rf"{ran}/{len(scope)} applicable"
        else:
            count = f"{ran}/{len(accepted)}"
        coverage_rows.append([rf"\texttt{{{tex(name)}}}", by, count, description])
    out.append(longtable(["Check", "By", "Coverage", "What it establishes"],
                         r"@{}p{0.22\linewidth}p{0.06\linewidth}p{0.12\linewidth}p{0.54\linewidth}@{}",
                         coverage_rows))
    ready = sum(1 for r in accepted
                if "prepared" in ((r["checks"].get("comparator") or {}).get("result") or ""))
    if ready:
        out.append(rf"\textbf{{Comparator.}} Its coverage above is {sum(1 for r in accepted if (r['checks'].get('comparator') or {{}}).get('ran'))}"
                   rf"/{len(accepted)} and that is the honest number: a workspace has been built for "
                   rf"{ready} of {len(accepted)} rows and none has been replayed. Comparator's sandbox is "
                   r"Landlock through \texttt{landrun}, so it runs on Linux and the corpus was assembled "
                   r"on macOS. Every released row therefore sits at \texttt{proof\_checked}; none claims "
                   r"\texttt{kernel\_replayed}. The workspaces ship with the release so the claim can be "
                   r"settled by anyone with a Linux runner.\par")

    # A check that applies to a subset is not a gap; only one that applies to
    # every row and ran on few of them is.
    thin = [name for name, _, _ in PIPELINE
            if name not in scoped
            and sum(1 for r in accepted if (r["checks"].get(name) or {}).get("ran")) < len(accepted) * 0.5]
    if thin:
        out.append("Coverage is uneven and the table says so. "
                   + ", ".join(rf"\texttt{{{tex(t)}}}" for t in thin)
                   + r" ran on a minority of rows, because they were switched on partway through the "
                     r"campaigns that produced this corpus. A row without a given check is not a row "
                     r"that failed it.\par")

    out.append(r"\clearpage\section{How much a verdict is worth}")
    out.append(rf"The judge was run twice over all {stats['n']} rows \textemdash{{}} same inputs, same "
               r"order, no shared state \textemdash{} so that its disagreement with itself could be "
               r"measured rather than assumed.\par")
    out.append(longtable(["Measure", "Agreement"], r"@{}p{0.66\linewidth}r@{}",
                         [[tex(label), f"{num}/{den} ({100*num//den}\\%)"]
                          for label, num, den in (
                              ("keep or reject", stats["verdict_agreement"], stats["n"]),
                              ("exact quality grade", stats["quality_agreement"], stats["n"]),
                              ("failure label, when both rejected", stats["failure_agreement"],
                               max(1, stats["both_reject"])))]))
    grades = ["strong", "acceptable", "weak"]
    out.append(longtable([r"pass 1 $\downarrow$ / pass 2 $\rightarrow$"] + grades,
                         r"@{}l" + "r" * len(grades) + r"@{}",
                         [[rf"\texttt{{{g}}}"] + [str(stats["matrix"].get((g, other), 0)) for other in grades]
                          for g in grades]))
    out.append(rf"Admission requires the first cell: \texttt{{strong}} twice. That is why the release is "
               rf"{len(accepted)} rows and not {considered}.\par")

    out.append(r"\clearpage\section{The corpus}")
    for number, row in enumerate(accepted, 1):
        out.append(_card(number, row))

    out.append(r"\end{document}")
    return "\n\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml1_release.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("data/release/eml1_rejected.jsonl"))
    parser.add_argument("--run1", type=Path, default=Path("data/release/rejudged.json"))
    parser.add_argument("--run2", type=Path, default=Path("data/release/rejudged_run2.json"))
    parser.add_argument("--output-tex", type=Path, default=Path("docs/eml1_release_report.tex"))
    parser.add_argument("--split-by-benchmark", action="store_true",
                        help="also write one document per source benchmark. The two "
                             "read differently -- miniF2F statements are competition "
                             "prose with inline LaTeX, ProofNet's are textbook exercises "
                             "-- and a reader working on one does not want the other "
                             "interleaved.")
    args = parser.parse_args()

    accepted = sorted(load(args.release), key=lambda r: (r["op_type"], r["operator_variant"], r["problem_id"]))
    held = load(args.rejected)
    stats = consistency(args.run1, args.run2)

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(render_tex(accepted, held, stats), encoding="utf-8")
    print(f"  {args.output_tex}  {args.output_tex.stat().st_size/1024:.0f} KB")

    if args.split_by_benchmark:
        families = sorted({r.get("benchmark") or "unknown" for r in accepted})
        for family in families:
            subset = [r for r in accepted if (r.get("benchmark") or "unknown") == family]
            # The held-back rows are filtered the same way, so each document's
            # discard pile is the one that belongs to it.
            subset_held = [r for r in held if (r.get("benchmark") or "unknown") == family]
            path = args.output_tex.with_name(
                args.output_tex.stem + "_" + family.lower().replace(" ", "_") + args.output_tex.suffix)
            path.write_text(render_tex(subset, subset_held, stats, family=family), encoding="utf-8")
            print(f"  {path}  {path.stat().st_size/1024:.0f} KB  ({len(subset)} cards)")
    print(f"\n{len(accepted)} cards · {sum(len(r['review']['passes']) for r in accepted)} reasoning texts "
          f"· {len(held)} held back")
    print("  compile with: lualatex (Unicode Lean; pdflatex cannot set it)")


if __name__ == "__main__":
    main()
