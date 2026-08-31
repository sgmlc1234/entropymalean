#!/usr/bin/env python3
"""Generate the paper's pastel workflow figures as deterministic SVG assets."""

from __future__ import annotations

import pathlib

import html
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Output lands in this repository's figure directory. The generator is the
# source; docs/figures/*.svg are its output, and editing those by hand is how
# a regeneration silently reverts them.
FIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "figures"


PALETTE = {
    "bg": "#FBFBF7",
    "ink": "#20323D",
    "muted": "#697783",
    "line": "#8EA1AE",
    "blue": "#D9EAF8",
    "blue2": "#BFD8EE",
    "mint": "#D7EDDF",
    "mint2": "#BFE2CF",
    "lav": "#E7DDF6",
    "lav2": "#D5C3EF",
    "peach": "#F8DBCB",
    "peach2": "#F0C4AC",
    "rose": "#F4D8E0",
    "yellow": "#F8EDC8",
    "slate": "#E8EEF2",
    "white": "#FFFFFF",
    "green": "#DDEED0",
    "paper": "#FFFEFC",
    "paper2": "#F6F3EE",
    "rule": "#66727C",
}


def esc(text: str) -> str:
    return html.escape(text, quote=True)


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["line"]}"/>',
            "</marker>",
            '<filter id="shadow" x="-8%" y="-8%" width="116%" height="116%">',
            '<feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#8EA1AE" flood-opacity="0.18"/>',
            "</filter>",
            "</defs>",
            f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>',
        ]

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None,
             rx: float = 22, opacity: float = 1.0, shadow: bool = True, dash: bool = False) -> None:
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
            f'rx="{rx}"', f'fill="{fill}"', f'opacity="{opacity}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}"')
            attrs.append('stroke-width="2"')
        if dash:
            attrs.append('stroke-dasharray="9 7"')
        if shadow:
            attrs.append('filter="url(#shadow)"')
        self.parts.append("<rect " + " ".join(attrs) + "/>")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str | None = None,
             width: float = 3, arrow: bool = True, dash: bool = False) -> None:
        attrs = [
            f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
            f'stroke="{color or PALETTE["line"]}"', f'stroke-width="{width}"',
            'stroke-linecap="round"', 'fill="none"',
        ]
        if arrow:
            attrs.append('marker-end="url(#arrow)"')
        if dash:
            attrs.append('stroke-dasharray="8 8"')
        self.parts.append("<line " + " ".join(attrs) + "/>")

    def path(self, d: str, color: str | None = None, width: float = 3,
             arrow: bool = True, dash: bool = False) -> None:
        attrs = [
            f'd="{d}"', f'stroke="{color or PALETTE["line"]}"',
            f'stroke-width="{width}"', 'stroke-linecap="round"',
            'stroke-linejoin="round"', 'fill="none"',
        ]
        if arrow:
            attrs.append('marker-end="url(#arrow)"')
        if dash:
            attrs.append('stroke-dasharray="8 8"')
        self.parts.append("<path " + " ".join(attrs) + "/>")

    def text(self, x: float, y: float, text: str, width: float, size: int = 24,
             color: str | None = None, weight: int = 400, line_height: float = 1.25,
             anchor: str = "start") -> float:
        color = color or PALETTE["ink"]
        chars = max(8, int(width / (size * 0.55)))
        lines: list[str] = []
        for paragraph in text.split("\n"):
            if paragraph.strip():
                lines.extend(textwrap.wrap(paragraph, width=chars, break_long_words=False))
            else:
                lines.append("")
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        )
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * line_height
            self.parts.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.parts.append("</text>")
        return y + max(1, len(lines)) * size * line_height

    def label(self, x: float, y: float, text: str, fill: str, color: str | None = None) -> None:
        pad_x, pad_y = 14, 8
        w = max(76, len(text) * 12 + 2 * pad_x)
        h = 34
        self.rect(x, y, w, h, fill, stroke=None, rx=17, shadow=False)
        self.text(x + w / 2, y + 23, text, w - 2 * pad_x, size=17,
                  color=color or PALETTE["ink"], weight=700, anchor="middle")

    def card(self, x: float, y: float, w: float, h: float, fill: str, title: str,
             body: str, accent: str | None = None, title_size: int = 26,
             body_size: int = 21) -> None:
        self.rect(x, y, w, h, fill, stroke=accent or PALETTE["line"], rx=24)
        if accent:
            self.rect(x, y, 12, h, accent, stroke=None, rx=6, shadow=False)
        title_end = self.text(x + 28, y + 42, title, w - 56, size=title_size, weight=800)
        self.text(x + 28, title_end + 14, body, w - 56, size=body_size, color=PALETTE["muted"])

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def architecture() -> None:
    s = SVG(1700, 1120)
    s.parts[-1] = f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>'

    def raw_text(x: float, y: float, value: str, size: int = 22,
                 color: str = "#20323D", weight: int = 400,
                 family: str = "Times New Roman, Times, serif",
                 anchor: str = "start", letter_spacing: float = 0) -> None:
        s.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'letter-spacing="{letter_spacing}">{esc(value)}</text>'
        )

    def lines(x: float, y: float, values: list[str], size: int = 20,
              color: str = "#4D5962", weight: int = 400,
              family: str = "Times New Roman, Times, serif",
              leading: float = 1.18, anchor: str = "start") -> None:
        for i, value in enumerate(values):
            raw_text(x, y + i * size * leading, value, size, color, weight,
                     family, anchor)

    portrait_count = 0

    def agent_portrait(x: float, y: float, agent: str, icon: str,
                       fill: str, stroke: str, size: float = 56) -> None:
        """Bottts portrait with a small, role-specific Lucide prop."""
        nonlocal portrait_count
        portrait_count += 1
        radius = size / 2
        clip_id = f"agent-clip-{portrait_count}"
        s.parts.append(
            f'<clipPath id="{clip_id}"><rect x="{x - radius}" y="{y - radius}" '
            f'width="{size}" height="{size}" rx="{size * 0.23}"/></clipPath>'
        )
        s.parts.append(
            f'<rect x="{x - radius}" y="{y - radius}" width="{size}" height="{size}" '
            f'rx="{size * 0.23}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        s.parts.append(
            f'<image href="assets/agents/{agent}.svg" x="{x - radius}" y="{y - radius}" '
            f'width="{size}" height="{size}" preserveAspectRatio="xMidYMid slice" '
            f'clip-path="url(#{clip_id})"/>'
        )
        badge = max(15, size * 0.31)
        bx = x + radius * 0.92
        by = y + radius * 0.92
        s.parts.append(
            f'<circle cx="{bx}" cy="{by}" r="{badge / 2}" fill="#FFFFFF" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
        )
        icon_size = badge * 0.68
        s.parts.append(
            f'<image href="assets/icons/{icon}.svg" x="{bx - icon_size / 2}" '
            f'y="{by - icon_size / 2}" width="{icon_size}" height="{icon_size}"/>'
        )

    def section_label(x: float, y: float, tag: str, title: str) -> None:
        s.parts.append(
            f'<circle cx="{x}" cy="{y - 7}" r="17" fill="#315E77"/>'
        )
        raw_text(x, y, tag, 18, "#FFFFFF", 700,
                 "Arial, Helvetica, sans-serif", "middle")
        raw_text(x + 30, y, title, 20, "#4E606B", 700,
                 "Arial, Helvetica, sans-serif", letter_spacing=1.0)

    def pill(x: float, y: float, w: float, value: str, fill: str,
             stroke: str, color: str, size: int = 16) -> None:
        s.rect(x, y, w, 31, fill, stroke=stroke, rx=15, shadow=False)
        raw_text(x + w / 2, y + 21, value, size, color, 700,
                 "Arial, Helvetica, sans-serif", "middle", 0.25)

    def check_row(x: float, y: float, title: str, note: str) -> None:
        s.rect(x, y, 360, 48, "#FFFFFF", stroke="#B9CBC1", rx=7, shadow=False)
        s.parts.append(
            f'<circle cx="{x + 23}" cy="{y + 24}" r="12" fill="#D9EEE0" '
            'stroke="#5E8B6C" stroke-width="1.5"/>'
        )
        s.path(f"M{x + 17} {y + 24} L{x + 22} {y + 29} L{x + 30} {y + 18}",
               color="#477356", width=2.4, arrow=False)
        raw_text(x + 44, y + 21, title, 18, "#263842", 700,
                 "Arial, Helvetica, sans-serif")
        raw_text(x + 44, y + 40, note, 14, "#6A787F", 400,
                 "Arial, Helvetica, sans-serif")

    # Two clearly separated bands: the production loop and what lineage buys.
    s.rect(38, 30, 1624, 650, PALETTE["paper"], stroke="#D4D8D5",
           rx=15, shadow=False)
    s.rect(38, 715, 1624, 365, "#F7F9FA", stroke="#CDD5DA",
           rx=15, shadow=False)
    section_label(72, 70, "A", "CERTIFIED EVOLUTION LOOP")
    section_label(72, 755, "B", "LINEAGE BECOMES A MEASUREMENT")

    # The accepted child loops back into the parent pool.
    s.path("M1510 140 C1510 78 710 76 650 128",
           color="#5E7180", width=3, arrow=True)
    pill(925, 55, 334, "ACCEPTED CHILD BECOMES A PARENT",
         "#FFFFFF", "#AAB8C1", "#526772", 15)

    # Seed bank.
    s.rect(70, 125, 270, 440, "#FFF9E8", stroke="#D8C777", rx=13, shadow=False)
    raw_text(96, 165, "Seed bank", 29, "#283A44", 700)
    raw_text(96, 194, "100 controls · 50 per benchmark", 17, "#68757C")

    s.rect(92, 225, 226, 115, "#FFFFFF", stroke="#D7C982", rx=10, shadow=False)
    raw_text(112, 258, "miniF2F", 22, "#3A4B54", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(112, 288, "statement only", 18, "#68757C")
    pill(112, 300, 150, "GEN-0 PROVE", "#FFF4CE", "#D2BD63", "#76621B", 14)

    s.rect(92, 365, 226, 115, "#FFFFFF", stroke="#C9BADB", rx=10, shadow=False)
    raw_text(112, 398, "ProofNet", 22, "#3A4B54", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(112, 428, "statement + proof", 18, "#68757C")
    pill(112, 440, 165, "REPLAY VERIFIED", "#F1ECFA", "#C4B2DA", "#67547B", 14)
    lines(96, 525, ["Every admitted seed", "is Lean-certified."], 16,
          "#5F6E76", 700, leading=1.18)

    # One evolutionary slot, with an actual compact crossover example.
    s.rect(390, 125, 380, 440, "#FBF3EF", stroke="#D8BBAA", rx=13, shadow=False)
    agent_portrait(436, 170, "orchestrator", "workflow", "#EEE7F7", "#70598A", 58)
    raw_text(480, 165, "Orchestrator", 25, "#283A44", 700)
    raw_text(480, 194, "parents · operator · retry", 16, "#68757C")

    s.rect(416, 220, 148, 75, "#FFFFFF", stroke="#C9CED2", rx=8, shadow=False)
    raw_text(433, 249, "P1", 16, "#7B858A", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(433, 279, "gcd = 12", 20, "#273A44", 700,
             "Courier New, Courier, monospace")
    s.rect(594, 220, 160, 75, "#FFFFFF", stroke="#C9CED2", rx=8, shadow=False)
    raw_text(611, 249, "P2", 16, "#7B858A", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(611, 279, "divs(198)", 20, "#273A44", 700,
             "Courier New, Courier, monospace")

    s.path("M490 295 C490 310 490 315 490 329", color="#8A9AA4", width=2.2)
    s.path("M490 295 C520 307 635 305 660 329", color="#8A9AA4", width=2.2)
    s.path("M674 295 L674 329", color="#8A9AA4", width=2.2)

    s.rect(416, 330, 157, 82, "#F1ECFA", stroke="#C4B2DA", rx=9, shadow=False)
    agent_portrait(440, 371, "mutation_judge", "gavel", "#E8DDF5", "#70598A", 42)
    raw_text(469, 360, "Mutation judge", 13, "#5F526B", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(469, 390, "variant contract", 12, "#5F526B", 700,
             "Arial, Helvetica, sans-serif")

    s.rect(587, 330, 167, 82, "#FBE9DF", stroke="#D7AA91", rx=9, shadow=False)
    agent_portrait(611, 371, "crossover_judge", "gavel", "#F6DDCF", "#8D5B42", 42)
    raw_text(640, 360, "Crossover judge", 13, "#684E40", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(640, 390, "both parents used", 12, "#684E40", 700,
             "Arial, Helvetica, sans-serif")

    s.path("M495 412 C495 428 552 428 566 439", color="#8A9AA4", width=2.2)
    s.path("M670 412 C670 428 610 428 596 439", color="#8A9AA4", width=2.2)
    s.rect(445, 442, 282, 68, "#FBE9DF", stroke="#D7AA91", rx=10, shadow=False)
    agent_portrait(471, 476, "generator", "pencil-line", "#F9DCCC", "#8D5B42", 42)
    raw_text(505, 466, "Generator · candidate", 14, "#7A6559", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(505, 494, "sum = 90; gcd = 6", 17, "#3B342F", 700,
             "Courier New, Courier, monospace")
    pill(470, 522, 232, "ONE LOCAL RETRY", "#FFFFFF", "#D8BBAA", "#765B4C", 14)

    # Formal acceptance boundary. The rows match the four faculties in the text,
    # with deterministic family/evaluator checks made explicit up front.
    s.rect(850, 105, 430, 490, "#F1F8F3", stroke="#93B59E", rx=15, shadow=False)
    agent_portrait(897, 150, "validator", "clipboard-check", "#DDEEE2", "#4E7A5B", 58)
    raw_text(941, 145, "Validation crew", 25, "#274237", 700)
    raw_text(941, 174, "shared boundary · all checks required", 16, "#61746A")
    check_row(875, 200, "Schema + evaluator", "supported template; answer round-trip")
    check_row(875, 260, "Hash + anti-stub", "identity; uniqueness")
    check_row(875, 320, "Selected judge", "operator-specific independent pass")
    check_row(875, 380, "Lean proof", "vacuity; load-bearing hypotheses")
    check_row(875, 440, "Kernel replay + 2-platform export", "exported proof term")
    pill(916, 515, 277, "EARNS A NAMED CERTIFICATE", "#E3F2E7",
         "#85AA91", "#3E694B", 14)

    # Released ledger, reusing the small fruit/check seal from Figure 2.
    s.rect(1320, 125, 320, 345, "#EEF5FA", stroke="#91AFC2", rx=13, shadow=False)
    raw_text(1345, 165, "Certified ledger", 25, "#283A44", 700)
    raw_text(1345, 235, "535", 52, "#315E77", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(1451, 232, "released rows", 20, "#536771", 700)
    raw_text(1345, 280, "248 miniF2F · 287 ProofNet", 18, "#536771")
    raw_text(1345, 318, "parents · content hashes", 18, "#536771")
    raw_text(1345, 348, "5 operators · gen. 1–10", 18, "#536771")
    pill(1345, 378, 243, "REPRODUCIBLE", "#FFFFFF", "#82A6BC", "#315E77", 14)
    s.parts.append('<circle cx="1577" cy="185" r="27" fill="#C96E62" stroke="#8F4D45" stroke-width="2"/>')
    s.parts.append('<path d="M1574 158 C1574 148 1582 142 1592 142" fill="none" stroke="#557A61" stroke-width="3" stroke-linecap="round"/>')
    s.parts.append('<path d="M1588 147 C1601 137 1615 142 1617 154 C1604 158 1595 157 1588 147 Z" fill="#9DBFA7" stroke="#557A61" stroke-width="1.5"/>')
    s.path("M1564 185 L1574 195 L1591 174", color="#FFFFFF", width=4, arrow=False)
    raw_text(1345, 445, "Lineage recorded at every generation.", 15,
             "#586B75", 700)

    # Main flow and the connected failure route.
    s.line(340, 280, 390, 280, color="#687C88", width=3)
    s.path("M770 476 C796 476 824 475 850 475", color="#687C88", width=3)
    s.line(1280, 350, 1320, 350, color="#4F7A61", width=3.5)
    raw_text(810, 458, "candidate", 14, "#6B777D", 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(1300, 333, "PASS", 14, "#477356", 700,
             "Arial, Helvetica, sans-serif", "middle")

    s.rect(846, 620, 536, 43, "#FBF0EE", stroke="#C78C83", rx=10, shadow=False)
    raw_text(866, 648, "FAIL", 15, "#97564D", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(929, 648, "quarantine  →  repair/backfill  →  rerun slot",
             17, "#704F4A", 700)
    s.path("M1088 595 L1088 620", color="#B06D63", width=2.5)
    s.path("M846 642 C795 642 760 600 744 565", color="#B06D63",
           width=2.5, arrow=True, dash=True)

    # Measurement band: show the condition before the statistic.
    raw_text(88, 800, "Condition on capability", 23, "#334852", 700)
    raw_text(88, 832, "test child only after every root is solved", 17, "#68757C")

    s.parts.append('<circle cx="146" cy="940" r="31" fill="#E3F2E7" stroke="#699077" stroke-width="2"/>')
    s.parts.append('<circle cx="262" cy="940" r="31" fill="#E3F2E7" stroke="#699077" stroke-width="2"/>')
    s.parts.append('<circle cx="398" cy="940" r="36" fill="#FBF0EE" stroke="#B7776E" stroke-width="2"/>')
    s.path("M134 940 L143 949 L160 927", color="#477356", width=3.5, arrow=False)
    s.path("M250 940 L259 949 L276 927", color="#477356", width=3.5, arrow=False)
    s.path("M384 926 L412 954 M412 926 L384 954", color="#9B5C54", width=3.5, arrow=False)
    s.path("M177 940 C210 880 330 880 368 932", color="#6B8595", width=2.5)
    s.path("M293 940 C320 940 344 940 362 940", color="#6B8595", width=2.5)
    raw_text(146, 994, "root A", 16, "#56676F", 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(262, 994, "root B", 16, "#56676F", 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(398, 994, "child", 16, "#8D514A", 700,
             "Arial, Helvetica, sans-serif", "middle")

    s.line(452, 940, 510, 940, color="#6B7E89", width=3)
    s.rect(520, 860, 390, 162, "#FFFFFF", stroke="#A9BAC4", rx=12, shadow=False)
    raw_text(715, 900, "LINEAGE PROOF GAP", 18, "#315E77", 700,
             "Arial, Helvetica, sans-serif", "middle", 0.7)
    raw_text(715, 953, "all roots solved  +  child missed", 22, "#2F424C", 700,
             "Times New Roman, Times, serif", "middle")
    raw_text(715, 989, "count over eligible treatment lineages", 16, "#68757C", 400,
             "Arial, Helvetica, sans-serif", "middle")

    s.line(910, 940, 968, 940, color="#6B7E89", width=3)
    s.rect(978, 850, 300, 180, "#FBF0EE", stroke="#C78C83", rx=12, shadow=False)
    raw_text(1002, 890, "LEAN PROVERS", 16, "#8A524A", 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.5)
    raw_text(1128, 961, "35–89%", 43, "#9B5C54", 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(1128, 999, "children missed", 16, "#735A56", 700,
             "Arial, Helvetica, sans-serif", "middle")

    s.rect(1294, 850, 320, 180, "#EDF6F0", stroke="#8DAF97", rx=12, shadow=False)
    raw_text(1318, 890, "FRONTIER REFERENCE", 16, "#4E7359", 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.5)
    raw_text(1454, 961, "11–32%", 43, "#477356", 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(1454, 999, "children missed", 16, "#566F5D", 700,
             "Arial, Helvetica, sans-serif", "middle")

    raw_text(850, 1060, "Same Lean verifier · K=3 per row · control-adjusted Pass@3",
             15, "#6A777E", 700, "Arial, Helvetica, sans-serif", "middle")
    s.save(FIG_DIR / "architecture_workflow.svg")


def certificate_ladder() -> None:
    """Draw the cumulative certificate ladder and its release threshold."""
    s = SVG(1700, 760)
    s.parts[-1] = f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>'
    defs_end = s.parts.index("</defs>")
    s.parts[defs_end:defs_end] = [
        '<linearGradient id="woodRail" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0" stop-color="#9C6638"/>',
        '<stop offset="0.42" stop-color="#D19A5C"/>',
        '<stop offset="0.72" stop-color="#BC7D43"/>',
        '<stop offset="1" stop-color="#83532F"/>',
        '</linearGradient>',
        '<linearGradient id="woodRung" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0" stop-color="#D5A268"/>',
        '<stop offset="0.55" stop-color="#B8793F"/>',
        '<stop offset="1" stop-color="#8E5B34"/>',
        '</linearGradient>',
    ]

    def raw_text(x: float, y: float, value: str, size: int = 24,
                 color: str = "#20323D", weight: int = 400,
                 family: str = "Times New Roman, Times, serif",
                 anchor: str = "start", letter_spacing: float = 0) -> None:
        s.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'letter-spacing="{letter_spacing}">{esc(value)}</text>'
        )

    def pill(x: float, y: float, w: float, value: str, fill: str,
             stroke: str, color: str, prefix: str = "") -> None:
        s.rect(x, y, w, 34, fill, stroke=stroke, rx=17, shadow=False)
        raw_text(x + w / 2, y + 23, f"{prefix}{value}", 17, color, 700,
                 "Arial, Helvetica, sans-serif", "middle", 0.35)

    # The rails taper slightly toward the top, and each rung sits behind them,
    # following the clean joinery of a simple orchard ladder.
    raw_text(143, 42, "STRONGER CLAIM", 17, "#5D6C76", 700,
             "Arial, Helvetica, sans-serif", "middle", 1.2)
    s.path("M143 76 L143 51 M143 51 L136 61 M143 51 L150 61",
           color="#5F7482", width=2.4, arrow=False)

    levels = [
        (132, "4", "#AFCBE4"),
        (253, "3", "#8FB8D7"),
        (407, "2", "#B9D1E6"),
        (558, "1", "#D8E6F2"),
        (665, "0", "#E5E7E8"),
    ]
    for cy, number, color in levels:
        s.parts.append(
            f'<rect x="87" y="{cy - 8}" width="112" height="16" rx="5" '
            'fill="url(#woodRung)" stroke="#784C2B" stroke-width="2"/>'
        )

    s.parts.append(
        '<path d="M70 708 L88 92 Q89 86 95 86 L113 86 L103 708 Z" '
        'fill="url(#woodRail)" stroke="#754927" stroke-width="2"/>'
    )
    s.parts.append(
        '<path d="M181 86 L199 86 Q205 86 206 92 L225 708 L192 708 Z" '
        'fill="url(#woodRail)" stroke="#754927" stroke-width="2"/>'
    )
    s.parts.append(
        '<path d="M85 690 C91 560 87 405 98 112 M203 690 C197 520 202 365 194 108" '
        'fill="none" stroke="#F1C78F" stroke-width="2.2" stroke-opacity="0.42" '
        'stroke-linecap="round"/>'
    )
    s.parts.append(
        '<path d="M76 652 C83 532 79 322 91 146 M217 662 C208 514 213 304 201 132" '
        'fill="none" stroke="#6F4527" stroke-width="1.5" stroke-opacity="0.28" '
        'stroke-linecap="round"/>'
    )

    for cy, number, color in levels:
        s.parts.append(
            f'<circle cx="143" cy="{cy}" r="19" fill="{color}" '
            'stroke="#754927" stroke-width="2"/>'
        )
        raw_text(143, cy + 7, number, 19, "#263842", 700,
                 "Arial, Helvetica, sans-serif", "middle")

    # Tier 4: the strongest claim remains above (and is not required to release).
    s.rect(258, 90, 1088, 84, "#EAF3FA", stroke="#7FA8C7", rx=12, shadow=False)
    raw_text(286, 124, "reproducible", 25, "#20323D", 700,
             "Courier New, Courier, monospace")
    raw_text(650, 124, "Byte-identical export on a second platform", 24, "#334852")
    pill(1157, 114, 162, "STRONGEST", "#F7FAFC", "#8EABC0", "#476377")
    raw_text(650, 152, "for environments that match the pinned toolchain", 18, "#667680")

    # Tier 3: the release threshold branches into a small fruit/check seal.
    s.rect(258, 204, 1088, 98, "#D7E9F6", stroke="#578BAF", rx=12, shadow=False)
    raw_text(286, 244, "kernel_replayed", 25, "#183A50", 700,
             "Courier New, Courier, monospace")
    raw_text(650, 241, "Independent kernel accepts the proof term", 24, "#263D49")
    raw_text(650, 272, "independent of the elaborator's own verdict", 18, "#5D707B")
    pill(1090, 252, 229, "RELEASE THRESHOLD", "#FFFFFF", "#4E819F", "#315E77")

    # The badge sits 130px above the arrow's own row so it clears the tier-1
    # band; kept as a transform rather than baked-in coordinates so the arrow
    # and the badge stay defined against the same grid as everything else.
    s.parts.append('<g transform="translate(0,-130)">')
    s.path("M1346 253 C1405 253 1404 253 1444 253",
           color="#4E819F", width=4, arrow=False)
    s.parts.append('<circle cx="1502" cy="254" r="51" fill="#C96E62" stroke="#8F4D45" stroke-width="3"/>')
    s.parts.append('<path d="M1498 203 C1497 184 1509 175 1523 174" fill="none" stroke="#557A61" stroke-width="5" stroke-linecap="round"/>')
    s.parts.append('<path d="M1515 183 C1538 162 1562 170 1567 193 C1542 199 1526 195 1515 183 Z" fill="#9DBFA7" stroke="#557A61" stroke-width="2"/>')
    s.path("M1479 254 L1496 272 L1527 234", color="#FFFFFF", width=7, arrow=False)
    raw_text(1503, 330, "EML-1 RELEASE", 21, "#7D403A", 700,
             "Arial, Helvetica, sans-serif", "middle", 0.6)
    raw_text(1503, 356, "535 / 535 rows", 19, "#5D6C76", 700,
             "Arial, Helvetica, sans-serif", "middle")
    s.parts.append('</g>')

    # Tier 2: use explicit labels and symbols so the semantics survive grayscale.
    s.rect(258, 326, 1088, 162, "#E7F0F7", stroke="#83A9C3", rx=12, shadow=False)
    raw_text(286, 367, "proof_checked", 25, "#20323D", 700,
             "Courier New, Courier, monospace")
    raw_text(650, 363, "Proof compiles; the obligation cannot be deferred", 24, "#2B414C")
    raw_text(650, 397, "Axiom closure", 18, "#5D707B", 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.45)
    pill(650, 409, 134, "ALLOWED", "#FFFFFF", "#6F94AE", "#3D6178", "+  ")
    pill(796, 409, 114, "propext", "#FFFFFF", "#8AA8BC", "#334852")
    pill(922, 409, 144, "Quot.sound", "#FFFFFF", "#8AA8BC", "#334852")
    pill(1078, 409, 219, "Classical.choice", "#FFFFFF", "#8AA8BC", "#334852")
    pill(650, 448, 144, "EXCLUDED", "#FBF0EE", "#B7776E", "#8D514A", "x  ")
    pill(806, 448, 112, "sorryAx", "#FBF0EE", "#B7776E", "#8D514A")
    pill(930, 448, 172, "native_decide", "#FBF0EE", "#B7776E", "#8D514A")

    # Tier 1 and the rejected baseline.
    s.rect(258, 510, 1088, 96, "#F1F6FA", stroke="#A7BCCB", rx=12, shadow=False)
    raw_text(286, 550, "statement_checked", 25, "#2C414D", 700,
             "Courier New, Courier, monospace")
    raw_text(650, 547, "Elaborates under the pinned toolchain", 24, "#334852")
    raw_text(650, 578, "anti-stub guard clear; no proof claim yet", 18, "#667680")

    s.rect(258, 624, 1088, 82, "#F5F5F3", stroke="#9C9C98", rx=12,
           shadow=False, dash=True)
    raw_text(286, 674, "none", 25, "#777A7B", 700,
             "Courier New, Courier, monospace")
    raw_text(650, 672, "Not accepted for EML-1", 24, "#777A7B")
    pill(1139, 648, 180, "OUTSIDE RELEASE", "#FFFFFF", "#A5A5A1", "#6F7273")

    s.save(FIG_DIR / "certificate_ladder.svg")


def episode_interfaces() -> None:
    """Contrast tactic search and whole-proof revision under one token cap."""
    s = SVG(1400, 720)
    s.parts[-1] = f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>'

    def raw_text(x: float, y: float, value: str, size: int = 22,
                 color: str = "#20323D", weight: int = 400,
                 family: str = "Times New Roman, Times, serif",
                 anchor: str = "start", letter_spacing: float = 0) -> None:
        s.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'letter-spacing="{letter_spacing}">{esc(value)}</text>'
        )

    def pill(x: float, y: float, w: float, value: str, fill: str,
             stroke: str, color: str, size: int = 17) -> None:
        s.rect(x, y, w, 36, fill, stroke=stroke, rx=18, shadow=False)
        raw_text(x + w / 2, y + 24, value, size, color, 700,
                 "Arial, Helvetica, sans-serif", "middle", 0.15)

    portrait_count = 0

    def bot_portrait(x: float, y: float, agent: str, icon: str,
                     fill: str, stroke: str, size: float = 66) -> None:
        nonlocal portrait_count
        portrait_count += 1
        radius = size / 2
        clip_id = f"episode-agent-clip-{portrait_count}"
        s.parts.append(
            f'<clipPath id="{clip_id}"><rect x="{x - radius}" y="{y - radius}" '
            f'width="{size}" height="{size}" rx="{size * 0.23}"/></clipPath>'
        )
        s.parts.append(
            f'<rect x="{x - radius}" y="{y - radius}" width="{size}" height="{size}" '
            f'rx="{size * 0.23}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        s.parts.append(
            f'<image href="assets/agents/{agent}.svg" x="{x - radius}" y="{y - radius}" '
            f'width="{size}" height="{size}" preserveAspectRatio="xMidYMid slice" '
            f'clip-path="url(#{clip_id})"/>'
        )
        badge = 20
        bx = x + radius * 0.9
        by = y + radius * 0.9
        s.parts.append(
            f'<circle cx="{bx}" cy="{by}" r="{badge / 2}" fill="#FFFFFF" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
        )
        s.parts.append(
            f'<image href="assets/icons/{icon}.svg" x="{bx - 7}" y="{by - 7}" '
            'width="14" height="14"/>'
        )

    blue = "#4A73A6"
    blue_bg = "#EFF4FA"
    blue_soft = "#DDEAF7"
    amber = "#B5762F"
    amber_bg = "#FCF4EA"
    amber_soft = "#F7E3C8"
    ink = "#263842"
    muted = "#68757C"

    # Figure-local markers keep arrowheads fully visible and color-matched.
    s.parts.append(
        '<defs>'
        '<marker id="episode-arrow-blue" viewBox="0 0 12 12" refX="11" refY="6" '
        'markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M1 1 L11 6 L1 11 Z" fill="{blue}"/></marker>'
        '<marker id="episode-arrow-muted" viewBox="0 0 12 12" refX="11" refY="6" '
        'markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto">'
        '<path d="M1 1 L11 6 L1 11 Z" fill="#8DA3B3"/></marker>'
        '<marker id="episode-arrow-amber" viewBox="0 0 12 12" refX="11" refY="6" '
        'markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M1 1 L11 6 L1 11 Z" fill="{amber}"/></marker>'
        '</defs>'
    )

    def flow_path(d: str, color: str, marker: str, width: float = 3.0,
                  dash: bool = False) -> None:
        dash_attr = ' stroke-dasharray="8 8"' if dash else ""
        s.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            'stroke-linecap="round" stroke-linejoin="round" '
            f'marker-end="url(#{marker})"{dash_attr}/>'
        )

    # Two interfaces use the same card geometry, header rhythm, and footer badge.
    s.rect(35, 45, 640, 520, blue_bg, stroke="#83A5C9", rx=17, shadow=False)
    s.rect(725, 45, 640, 520, amber_bg, stroke="#D4A064", rx=17, shadow=False)

    bot_portrait(85, 100, "tactic_prover", "git-branch", "#E5EFF9", blue, 66)
    raw_text(135, 92, "Tactic prover", 29, ink, 700)
    raw_text(135, 122, "best-first search over Lean states", 18, muted, 400,
             "Arial, Helvetica, sans-serif")

    bot_portrait(775, 100, "whole_proof_prover", "file-pen-line",
                 "#FAE9D7", amber, 66)
    raw_text(825, 92, "Whole-proof prover", 29, ink, 700)
    raw_text(825, 122, "revision from Lean diagnostics", 18, muted, 400,
             "Arial, Helvetica, sans-serif")

    # Left: make the search tree visible instead of describing it only in boxes.
    s.rect(90, 145, 530, 66, "#FFFFFF", stroke="#9BB7D4", rx=10, shadow=False)
    raw_text(112, 171, "GOAL STATE  G_k", 16, blue, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.45)
    raw_text(112, 199, "|- theorem after k tactics", 21, ink, 700,
             "Courier New, Courier, monospace")
    for cx, color, marker, width in (
        (170, "#8DA3B3", "episode-arrow-muted", 2.8),
        (355, blue, "episode-arrow-blue", 3.4),
        (540, "#8DA3B3", "episode-arrow-muted", 2.8),
    ):
        flow_path(f"M355 211 C355 241 {cx} 247 {cx} 276",
                  color, marker, width)
    # The label masks the shared stem while leaving complete arrowheads below it.
    s.rect(220, 226, 270, 34, "#FFFFFF", stroke="#B7CADB", rx=17,
           shadow=False)
    raw_text(355, 249, "sample n tactic lines", 17, muted, 700,
             "Arial, Helvetica, sans-serif", "middle")

    tactic_specs = [
        (100, "candidate 1", "linarith", "#FFFFFF", "#AFC4D8"),
        (285, "SELECTED", "omega", blue_soft, blue),
        (470, "candidate n", "ring_nf", "#FFFFFF", "#AFC4D8"),
    ]
    for x, tag, tactic, fill, stroke in tactic_specs:
        s.rect(x, 284, 140, 50, fill, stroke=stroke, rx=9, shadow=False)
        tag_color = blue if tag == "SELECTED" else "#6F8290"
        raw_text(x + 70, 303, tag, 12, tag_color, 700,
                 "Arial, Helvetica, sans-serif", "middle", 0.25)
        raw_text(x + 70, 326, tactic, 18, ink, 700,
                 "Courier New, Courier, monospace", "middle")

    flow_path("M170 334 C170 346 250 346 275 350",
              "#A6B5BF", "episode-arrow-muted", 2.6)
    flow_path("M355 334 L355 350", blue, "episode-arrow-blue", 3.4)
    flow_path("M540 334 C540 346 460 346 435 350",
              "#A6B5BF", "episode-arrow-muted", 2.6)
    s.rect(175, 358, 360, 58, "#FFFFFF", stroke="#9BB7D4", rx=10, shadow=False)
    raw_text(355, 382, "BEST-FIRST QUEUE", 16, blue, 700,
             "Arial, Helvetica, sans-serif", "middle", 0.45)
    raw_text(355, 406, "rank by cumulative log-probability", 18, ink, 700,
             "Arial, Helvetica, sans-serif", "middle")

    flow_path("M355 416 L355 432", blue, "episode-arrow-blue", 3.4)
    s.rect(145, 440, 420, 56, blue_soft, stroke=blue, rx=10, shadow=False)
    raw_text(355, 464, "apply selected tactic; extend the tree", 19, ink, 700,
             "Arial, Helvetica, sans-serif", "middle")
    raw_text(355, 487, "Lean returns the next goal state", 16, muted, 400,
             "Arial, Helvetica, sans-serif", "middle")
    flow_path("M145 468 C83 468 58 424 58 350 L58 178 L82 178",
              "#7898B0", "episode-arrow-muted", 2.8, dash=True)
    # Keep the loop and its label disjoint; a short leader associates them.
    s.line(58, 385, 76, 385, color="#7898B0", width=2.0, arrow=False)
    s.rect(78, 370, 84, 30, "#FFFFFF", stroke="#B7CADB", rx=15,
           shadow=False)
    raw_text(120, 391, "repeat", 14, blue, 700,
             "Arial, Helvetica, sans-serif", "middle")
    pill(180, 516, 350, "ONE ACTION = ONE TACTIC LINE",
         "#FFFFFF", "#89A9C9", blue, 16)

    # Right: a whole proof is emitted, checked, and revised from diagnostics.
    s.rect(765, 160, 245, 86, "#FFFFFF", stroke="#D6AB79", rx=10, shadow=False)
    raw_text(785, 185, "LAST ATTEMPT", 15, amber, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.4)
    raw_text(785, 214, "by\n  exact ...", 18, ink, 700,
             "Courier New, Courier, monospace")
    raw_text(1039, 208, "+", 29, amber, 700,
             "Arial, Helvetica, sans-serif", "middle")
    s.rect(1065, 160, 250, 86, "#FFFFFF", stroke="#D6AB79", rx=10, shadow=False)
    raw_text(1085, 185, "LEAN COMPLAINT", 15, amber, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.4)
    raw_text(1085, 216, "unsolved goals", 18, "#8A524A", 700,
             "Courier New, Courier, monospace")

    flow_path("M1040 246 L1040 276", amber, "episode-arrow-amber", 3.4)
    s.rect(825, 285, 430, 90, amber_soft, stroke=amber, rx=10, shadow=False)
    raw_text(1040, 313, "COMPLETE CANDIDATE PROOF", 16, amber, 700,
             "Arial, Helvetica, sans-serif", "middle", 0.45)
    raw_text(1040, 343, "by  omega", 21, ink, 700,
             "Courier New, Courier, monospace", "middle")
    raw_text(1040, 365, "one generation emits the entire proof body", 15, muted, 400,
             "Arial, Helvetica, sans-serif", "middle")

    flow_path("M1040 375 L1040 412", amber, "episode-arrow-amber", 3.4)
    s.rect(870, 420, 340, 70, "#FFFFFF", stroke="#D6AB79", rx=10, shadow=False)
    s.parts.append('<circle cx="910" cy="455" r="18" fill="#E4F2E8" stroke="#5E8B6C" stroke-width="2"/>')
    s.path("M900 455 L907 462 L920 445", color="#477356", width=3.0, arrow=False)
    raw_text(945, 449, "LEAN VERIFIER", 17, amber, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.4)
    raw_text(945, 474, "close or return diagnostics", 17, ink, 700,
             "Arial, Helvetica, sans-serif")
    flow_path("M1210 455 L1335 455 L1335 203 L1323 203",
              "#C18A4E", "episode-arrow-amber", 2.8, dash=True)
    s.rect(1172, 384, 158, 29, "#FFFFFF", stroke="#D6AB79", rx=14.5,
           shadow=False)
    raw_text(1251, 404, "Lean feedback", 14, amber, 700,
             "Arial, Helvetica, sans-serif", "middle")
    pill(870, 516, 350, "ONE ACTION = ONE COMPLETE PROOF",
         "#FFFFFF", "#D0A16C", amber, 16)

    # The shared currency is the visual conclusion, not a footnote.
    s.line(355, 565, 355, 598, color="#8A9AA4", width=2.2, arrow=False)
    s.line(1040, 565, 1040, 598, color="#8A9AA4", width=2.2, arrow=False)
    s.rect(35, 600, 1330, 92, "#FFFFFF", stroke="#A8B3BA", rx=15, shadow=False)
    for i, fill in enumerate(("#F2C45D", "#E5AD3C", "#D99A27")):
        s.parts.append(
            f'<circle cx="{78 + i * 22}" cy="646" r="13" fill="{fill}" '
            'stroke="#9B7228" stroke-width="1.5"/>'
        )
    raw_text(160, 632, "SHARED CURRENCY", 16, "#566872", 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.6)
    raw_text(160, 669, "8,192 generated tokens / episode", 28, ink, 700)
    raw_text(770, 636, "fair across unequal action sizes", 18, "#566872", 700,
             "Arial, Helvetica, sans-serif")
    raw_text(770, 668, "not actions  ·  not attempts", 18, "#8A524A", 700,
             "Arial, Helvetica, sans-serif")
    pill(1115, 627, 205, "K = 3 PER CELL", "#F4F6F7", "#A8B3BA", "#455862", 16)

    s.save(FIG_DIR / "episode_interfaces.svg")


def _crossover_card_draft() -> None:
    """Render the shortest fully reviewed crossover in the released gallery."""
    s = SVG(1700, 900)
    s.parts[-1] = f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>'

    def raw_text(x: float, y: float, value: str, size: int = 22,
                 color: str = "#20323D", weight: int = 400,
                 family: str = "Times New Roman, Times, serif",
                 anchor: str = "start", letter_spacing: float = 0) -> None:
        s.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'letter-spacing="{letter_spacing}">{esc(value)}</text>'
        )

    def lines(x: float, y: float, values: list[str], size: int = 20,
              color: str = "#4D5962", weight: int = 400,
              family: str = "Arial, Helvetica, sans-serif",
              leading: float = 1.35, anchor: str = "start") -> None:
        for i, value in enumerate(values):
            raw_text(x, y + i * size * leading, value, size, color, weight,
                     family, anchor)

    def pill(x: float, y: float, w: float, value: str, fill: str,
             stroke: str, color: str, size: int = 16) -> None:
        s.rect(x, y, w, 36, fill, stroke=stroke, rx=18, shadow=False)
        raw_text(x + w / 2, y + 24, value, size, color, 700,
                 "Arial, Helvetica, sans-serif", "middle", 0.2)

    ink = "#263842"
    muted = "#68757C"
    blue = "#4A73A6"
    green = "#4F8062"
    amber = "#B5762F"

    s.rect(40, 38, 1620, 824, "#FFFEFC", stroke="#C8CFD3", rx=16,
           shadow=False)
    pill(326, 58, 1048,
         "ACTUAL RELEASE ROW  ·  mathd_algebra_251 × mathd_algebra_114  ·  e0f06d2f",
         "#F4F6F7", "#B3BDC3", "#52656F", 16)

    for cx, title in ((330, "Parent facts"), (850, "Witness exchange"),
                      (1370, "Released child")):
        raw_text(cx, 133, title, 24, ink, 700,
                 "Times New Roman, Times, serif", "middle")
        s.line(cx - 120, 145, cx + 120, 145, color="#B5BEC3", width=1.6,
               arrow=False)

    # Parent A creates the witness x = 2.
    s.rect(90, 170, 480, 190, "#EFF4FA", stroke="#95B2CF", rx=12,
           shadow=False)
    raw_text(116, 207, "Parent A", 27, ink, 700)
    raw_text(116, 237, "mathd_algebra_251", 15, blue, 700,
             "Arial, Helvetica, sans-serif")
    s.rect(116, 255, 428, 55, "#FFFFFF", stroke="#B8CADB", rx=8,
           shadow=False)
    raw_text(330, 290, "3 + 1/x = 7/x", 23, ink, 700,
             "Courier New, Courier, monospace", "middle")
    pill(180, 316, 300, "PRODUCES   x = 2", "#FFFFFF", "#95B2CF", blue, 17)

    # Parent B exposes the input that the crossover must supply.
    s.rect(90, 390, 480, 205, "#FFF7E9", stroke="#D8B27F", rx=12,
           shadow=False)
    raw_text(116, 427, "Parent B", 27, ink, 700)
    raw_text(116, 457, "mathd_algebra_114", 15, amber, 700,
             "Arial, Helvetica, sans-serif")
    raw_text(116, 492, "If a = 8, then", 19, muted, 700,
             "Arial, Helvetica, sans-serif")
    s.rect(116, 507, 428, 52, "#FFFFFF", stroke="#DEC49F", rx=8,
           shadow=False)
    raw_text(330, 539, "(16 * (a^2)^(1/3))^(1/3) = 4", 17, ink, 700,
             "Courier New, Courier, monospace", "middle")
    pill(180, 555, 300, "REQUIRES   a = 8", "#FFFFFF", "#D8B27F", amber, 17)

    # The crossover is a serial witness handoff, not a conjunction.
    s.rect(650, 170, 400, 425, "#F1F8F3", stroke="#9DBEAA", rx=12,
           shadow=False)
    raw_text(850, 211, "witness_exchange", 20, green, 700,
             "Courier New, Courier, monospace", "middle")
    steps = [
        (238, "1", "solve Parent A", "x = 2"),
        (332, "2", "apply child coupling", "a = x^3"),
        (426, "3", "transport witness", "a = 8"),
    ]
    for y, number, title, result in steps:
        s.parts.append(
            f'<circle cx="700" cy="{y + 31}" r="21" fill="#FFFFFF" '
            'stroke="#75A086" stroke-width="2"/>'
        )
        raw_text(700, y + 38, number, 17, green, 700,
                 "Arial, Helvetica, sans-serif", "middle")
        s.rect(735, y, 270, 63, "#FFFFFF", stroke="#B7CDBF", rx=9,
               shadow=False)
        raw_text(755, y + 25, title, 16, muted, 700,
                 "Arial, Helvetica, sans-serif")
        raw_text(755, y + 50, result, 20, ink, 700,
                 "Courier New, Courier, monospace")
        if y < 426:
            s.line(850, y + 64, 850, y + 87, color="#75A086", width=2.8)
    pill(704, 522, 292, "SEQUENTIAL, NOT PARALLEL", "#FFFFFF",
         "#9DBEAA", green, 15)

    # The child carries both hypotheses and Parent B's conclusion.
    s.rect(1130, 170, 480, 425, "#FBF1EC", stroke="#D5AE9B", rx=12,
           shadow=False)
    raw_text(1158, 207, "latent_parameter_radical", 18, amber, 700,
             "Courier New, Courier, monospace")
    s.rect(1158, 229, 424, 126, "#FFFFFF", stroke="#D8C2B8", rx=9,
           shadow=False)
    raw_text(1180, 260, "ASSUMPTIONS", 14, amber, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.5)
    raw_text(1180, 290, "3 + 1/x = 7/x ; a = x^3", 18, ink, 700,
             "Courier New, Courier, monospace")
    raw_text(1180, 326, "GOAL", 14, amber, 700,
             "Arial, Helvetica, sans-serif", letter_spacing=0.5)
    raw_text(1180, 348, "(16 * (a^2)^(1/3))^(1/3) = 4", 16, ink, 700,
             "Courier New, Courier, monospace")

    for y, name, result in (
        (378, "solve", "x = 2"),
        (444, "transport", "a = x^3 = 8"),
        (510, "close", "radical value = 4"),
    ):
        s.rect(1158, y, 424, 52, "#FFFFFF", stroke="#D8C2B8", rx=8,
               shadow=False)
        raw_text(1178, y + 32, name.upper(), 14, amber, 700,
                 "Arial, Helvetica, sans-serif")
        raw_text(1325, y + 33, result, 18, ink, 700,
                 "Courier New, Courier, monospace")

    # Complete arrowheads terminate in whitespace rather than under card borders.
    s.path("M570 265 C600 265 610 286 642 286", color="#718795", width=3.0)
    s.path("M570 493 C600 493 610 475 642 475", color="#718795", width=3.0)
    s.path("M1050 382 L1122 382", color="#718795", width=3.2)

    raw_text(850, 646, "Lean proof checkpoints", 21, muted, 700,
             "Arial, Helvetica, sans-serif", "middle", 0.35)
    checkpoints = [
        (90, "1  nonzero", "x != 0 from h1"),
        (485, "2  solve", "field_simp + linarith"),
        (880, "3  substitute", "a = x^3 = 8"),
        (1275, "4  evaluate", "64^(1/3) = 4"),
    ]
    for i, (x, title, body) in enumerate(checkpoints):
        s.rect(x, 670, 335, 76, "#F5F8FA" if i % 2 == 0 else "#FAFAF8",
               stroke="#C4CDD2", rx=9, shadow=False)
        raw_text(x + 18, 699, title.upper(), 14, "#536771", 700,
                 "Arial, Helvetica, sans-serif", letter_spacing=0.25)
        raw_text(x + 18, 728, body, 16, ink, 700,
                 "Courier New, Courier, monospace")

    s.rect(90, 778, 1520, 52, "#EAF5EE", stroke="#B8D2C0", rx=10,
           shadow=False)
    raw_text(850, 811,
             "KERNEL_REPLAYED  ·  BOTH HYPOTHESES LOAD-BEARING  ·  NO PARENT PROVES THE CHILD ALONE",
             16, green, 700, "Arial, Helvetica, sans-serif", "middle", 0.25)

    s.save(FIG_DIR / "crossover_pipeline.svg")


def crossover() -> None:
    """Compile the LaTeX-typeset release example and export it as SVG."""
    import subprocess

    source = FIG_DIR / "crossover_pipeline_source.tex"
    build_dir = ROOT / "tmp" / "figure_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
            "-output-directory", str(build_dir), str(source),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source_pdf = build_dir / "crossover_pipeline_source.pdf"
    subprocess.run(
        ["pdftocairo", "-svg", str(source_pdf),
         str(FIG_DIR / "crossover_pipeline.svg")],
        cwd=ROOT,
        check=True,
    )


def mutation() -> None:
    """Compile the LaTeX-typeset mutation example and export it as SVG."""
    import subprocess

    source = FIG_DIR / "mutation_pipeline_source.tex"
    build_dir = ROOT / "tmp" / "figure_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
            "-output-directory", str(build_dir), str(source),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source_pdf = build_dir / "mutation_pipeline_source.pdf"
    subprocess.run(
        ["pdftocairo", "-svg", str(source_pdf),
         str(FIG_DIR / "mutation_pipeline.svg")],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    certificate_ladder()
    episode_interfaces()
    crossover()
    mutation()
    for name in ("architecture_workflow", "certificate_ladder", "episode_interfaces",
                 "crossover_pipeline", "mutation_pipeline"):
        print(FIG_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
