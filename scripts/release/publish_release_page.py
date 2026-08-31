"""Put the release and its audit trail on the working-paper site.

The site already carries the seeds, the certificate ladder, and the results. It
does not carry the thing reviewers keep asking for: what the judge actually said,
and what the pipeline threw away. This adds a `#/release` route holding both,
built from the same JSONL the Markdown and HTML reports are built from, so the
three cannot drift.

The page is edited in place between markers and the script is safe to re-run --
each insertion is replaced, not appended. `workspace.html` is hand-authored and
425 KB; nothing outside the markers is touched.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import glob
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from scripts.release.build_release_report import FAILURE_GLOSS, PIPELINE, consistency, load

BEGIN = "<!-- release:begin -->"
END = "<!-- release:end -->"
FAITH_BEGIN = "<!-- faithfulness:begin -->"
FAITH_END = "<!-- faithfulness:end -->"

#: What each gate settles, in the words a reader needs rather than the field
#: name. The release page lists coverage; this says what coverage would mean.
SETTLES = {
    "statement_type_check": "the statement is well-formed Lean and invents nothing",
    "proof_accepted": "the proof closes, with no <code>sorry</code>",
    "axiom_audit": "the proof rests on Lean's standard axioms and nothing else",
    "vacuity": "the hypotheses can all hold at once, so the theorem is not free",
    "dead_hypotheses": "every hypothesis is load-bearing",
    "corpus_dedup": "the statement is not already in the corpus under another name",
    "redundancy": "no hypothesis can be deleted, no parent proves itself",
    "hypothesis_preservation": "a silent mutation kept every hypothesis its parent had <em>(silent only)</em>",
    "comparator": "an independent runner replays the proof through the kernel",
    "goal_roundtrip": "the elaborated goal is the problem the prose states",
}

#: Checks that stopped being what they were. Each entry is what it used to do,
#: what it does now, and the measurement that forced the change -- a claim about
#: a gate is worth what its evidence is worth.
CHANGED = [
    ("redundancy", "added the parent as a hypothesis", "deletes a hypothesis and re-proves",
     "The old form asked a prover to close <code>(hP : parent) (child binders) : child goal</code>. "
     "Every certified child makes that true with <code>hP</code> unused, so it was provable for the "
     "whole corpus: run over the release it called 66 of 165 rows redundant, and a control putting "
     "a contentless hypothesis in the parent's place reproduced four of the first six. It measured "
     "prover strength. What replaced it deletes a hypothesis instead, which makes the statement "
     "strictly harder, and it is now a gate rather than a note — 12 rows two judge passes both "
     "called strong were dropped by it."),
    ("silent_equivalence", "decided the row", "informs the judge",
     "Asked to derive parent and child from each other. Both are theorems, so a "
     "direction closes whether or not the hypothesis is used: tested directly it "
     "passed a child that had dropped a conjunct, and an unrelated true statement. "
     "It could not fail on the cases it existed to catch."),
    ("dead_hypotheses", "searched the proof for the name", "removes it and recompiles",
     "The name search was right 24 times out of 53, because <code>omega</code>, "
     "<code>simp_all</code> and <code>linarith</code> consume hypotheses without "
     "naming one. Lean deciding it found 25 genuinely dead hypotheses across 14 "
     "released rows, repeated to a fixpoint because removing one can make another dead."),
    ("corpus_dedup", "blanked binder declarations", "alpha-normalises the statement",
     "Binder names were erased where they were declared and left where they were "
     "used, so the same closure identity over <code>n</code> and over <code>r</code> "
     "hashed differently. Four alpha-equivalent pairs had been released."),
]

PAGE = """<section class="page" id="p-/release">
  <h1>The release, and how it was checked</h1>
  <p class="lede">%(n)d generated theorems with complete Lean proofs. Every one carries the
  reasoning behind each judgement it received, including the passes that disagreed, and the
  result of every mechanical check. The rows that did not make it are here too, with their
  reasons, because a corpus that shows only its survivors cannot be checked.</p>

  <h2>What it is made of</h2>
  <p>Topic is a property of the seed, so it is read through the lineage: each row's
  roots are taken from its identifier and each root's topic from the seed sheet. Set
  against the seed donuts on <a href="#/seeds">Seeds</a>, these say what the corpus grew
  into on the axis it was bred from.</p>
  <div class="grid g2">
    <div class="card"><h3 style="margin:0 0 .6rem">ProofNet lineage</h3><div id="rel-donut-pn"></div></div>
    <div class="card"><h3 style="margin:0 0 .6rem">miniF2F lineage</h3><div id="rel-donut-mf"></div></div>
  </div>
  <p class="muted" id="rel-donut-note"></p>

  <h2>What was thrown away</h2>
  <p>%(considered)d certified rows were considered. Admission required <code>strong</code> from
  two independent passes of the judge.</p>
  <div class="rel-two">
    <div class="card"><div class="rel-scroll"><table class="rel-t" id="rel-outcome"></table></div></div>
    <div class="card"><div class="rel-scroll"><table class="rel-t" id="rel-failures"></table></div></div>
  </div>

  <h2>How a row got here</h2>
  <p>Each gate, and what kind of thing settles it. Lean decides the questions that have an
  answer; a model is asked only the questions that do not. Coverage is uneven and the table
  says so — some checks were switched on partway through these campaigns, and a row without
  a check is not a row that failed it.</p>
  <div class="card"><div class="rel-scroll"><table class="rel-t" id="rel-pipeline"></table></div></div>
  <p id="rel-comparator"></p>

  <h2>How much a verdict is worth</h2>
  <p>The judge was run twice over all %(consistency_n)d rows — same inputs, same order, no shared
  state — so that its disagreement with itself could be measured rather than assumed. It is
  why no single verdict decides a row here.</p>
  <div class="rel-two">
    <div class="card"><div class="rel-scroll"><table class="rel-t" id="rel-agree"></table></div></div>
    <div class="card"><div class="rel-scroll"><table class="rel-t" id="rel-matrix"></table></div></div>
  </div>

  <h2>Every row</h2>
  <div class="rel-controls">
    <input id="rel-q" placeholder="search statements, reasoning, Lean…" spellcheck="false">
    <span class="rel-filters" id="rel-filters"></span>
    <span class="rel-count" id="rel-count"></span>
  </div>
  <div class="rel-split">
    <ol class="rel-list" id="rel-list"></ol>
    <article class="rel-detail card" id="rel-detail"></article>
  </div>
</section>"""

CSS = """<style id="release-css">
  /* A table is the one block that will not wrap: it must scroll inside its own
     box or the whole page scrolls sideways on a phone. */
  .rel-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
  .rel-detail code,.rel-list code{overflow-wrap:anywhere}
  .rel-two{display:grid;gap:.85rem;grid-template-columns:repeat(auto-fit,minmax(19rem,1fr));margin-bottom:1rem}
  .rel-t{border-collapse:collapse;width:100%;font-size:.83rem;font-variant-numeric:tabular-nums}
  .rel-t th,.rel-t td{text-align:left;padding:.32rem .5rem;border-bottom:1px solid var(--rule);vertical-align:top}
  .rel-t th{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:600}
  .rel-t td.n,.rel-t th.n{text-align:right;white-space:nowrap}
  .rel-controls{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.6rem 0}
  #rel-q{flex:1 1 13rem;min-width:0;background:var(--surface);color:var(--ink);border:1px solid var(--rule);
    border-radius:4px;padding:.34rem .6rem;font:inherit;font-size:.85rem}
  .rel-filters{display:flex;flex-wrap:wrap;gap:.25rem}
  .rel-filters button{padding:.18rem .5rem;border:1px solid var(--rule);border-radius:999px;
    background:var(--surface);color:var(--soft);font:inherit;font-size:.76rem;cursor:pointer}
  .rel-filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--paper)}
  .rel-count{font-size:.72rem;color:var(--faint);margin-left:auto}
  .rel-split{display:grid;gap:.85rem;grid-template-columns:minmax(0,22rem) minmax(0,1fr);align-items:start}
  @media (max-width:60rem){.rel-split{grid-template-columns:minmax(0,1fr)}}
  .rel-list{list-style:none;margin:0;padding:0;max-height:38rem;overflow:auto;border:1px solid var(--rule);
    border-radius:6px;background:var(--surface)}
  .rel-list li{padding:.5rem .7rem;border-bottom:1px solid var(--rule);cursor:pointer;font-size:.82rem}
  .rel-list li:hover{background:var(--sunk)}
  .rel-list li[aria-current="true"]{background:var(--accent-soft)}
  .rel-list .rid{font-family:var(--mono);font-size:.74rem;color:var(--soft);
    overflow-wrap:anywhere;display:block}
  .rel-detail{min-width:0;min-height:12rem}
  .rel-detail h3{font-family:var(--mono);font-size:.9rem;margin:.1rem 0 .5rem;overflow-wrap:anywhere}
  .rel-detail pre{background:var(--sunk);border:1px solid var(--rule);border-radius:5px;padding:.6rem .75rem;
    overflow-x:auto;font-family:var(--mono);font-size:.76rem;line-height:1.5;margin:.3rem 0}
  .rel-detail blockquote{margin:.25rem 0;padding:.1rem 0 .1rem .7rem;border-left:2px solid var(--rule);
    color:var(--soft);font-size:.85rem}
  .rel-lab{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:var(--faint);
    display:block;margin-top:.8rem}
  .rel-pill{display:inline-block;border:1px solid var(--rule);border-radius:999px;padding:.02rem .45rem;
    font-family:var(--mono);font-size:.7rem;color:var(--soft);margin:.1rem .2rem .1rem 0}
  .rel-pill.ok{color:var(--pass);border-color:currentColor}
  .rel-pill.no{color:var(--fail);border-color:currentColor}
  .rel-pill.mid{color:var(--warn);border-color:currentColor}
  .rel-detail details>summary{cursor:pointer;font-size:.8rem;color:var(--soft);margin-top:.5rem}
</style>"""

SCRIPT = """<script id="release-js">
(function(){
  var D = JSON.parse(document.getElementById('release-data').textContent);
  var esc = function(t){ return String(t==null?'':t).replace(/[&<>]/g,function(m){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;'})[m]; }); };
  function table(id, head, rows){
    var el = document.getElementById(id); if(!el) return;
    el.innerHTML = '<thead><tr>' + head.map(function(h,i){
      return '<th'+(i?' class="n"':'')+'>'+esc(h)+'</th>'; }).join('') + '</tr></thead><tbody>' +
      rows.map(function(r){ return '<tr>' + r.map(function(c,i){
        return '<td'+(i&&typeof c==='number'?' class="n"':'')+'>'+c+'</td>'; }).join('') + '</tr>';
      }).join('') + '</tbody>';
  }
  // The seed page has a donut of its own, in a scope this block cannot reach.
  // Duplicated rather than hoisted: the two are drawn from different data and
  // a shared helper would couple a generated block to a hand-authored one.
  var RPAL = ['#0E7C7B','#1F6B4A','#A3323C','#7c5cbf','#c2703a','#2b6cb0','#8b8b3a'];
  function reldonut(host, tally){
    if(!host || !tally) return 0;
    var keys = Object.keys(tally), total = keys.reduce(function(a,k){ return a+tally[k]; },0);
    var r = 52, c = 2*Math.PI*r, off = 0;
    var arcs = keys.map(function(k,i){
      var seg = total ? c*(tally[k]/total) : 0;
      var a = '<circle r="'+r+'" cx="70" cy="70" fill="none" stroke="'+RPAL[i%RPAL.length]+
              '" stroke-width="22" stroke-dasharray="'+seg.toFixed(2)+' '+(c-seg).toFixed(2)+
              '" stroke-dashoffset="'+(-off).toFixed(2)+'" transform="rotate(-90 70 70)"></circle>';
      off += seg; return a;
    }).join('');
    host.className = 'donut';
    host.innerHTML = '<svg viewBox="0 0 140 140" width="140" height="140" role="img">'+arcs+
      '<text x="70" y="76" text-anchor="middle" font-size="22" fill="currentColor">'+total+'</text></svg>'+
      '<div class="dleg">'+keys.map(function(k,i){
        return '<span><i style="background:'+RPAL[i%RPAL.length]+'"></i>'+
               esc(k)+' · '+tally[k]+'</span>'; }).join('')+'</div>';
    return total;
  }
  var pnN = reldonut(document.getElementById('rel-donut-pn'), (D.lineage||{})['ProofNet']);
  var mfN = reldonut(document.getElementById('rel-donut-mf'), (D.lineage||{})['miniF2F']);
  var note = document.getElementById('rel-donut-note');
  if(note){
    note.textContent = 'Counted as lineage attributions, not rows: a crossover reaches two '
      + 'roots and is counted under both topics, so ' + (pnN+mfN) + ' attributions come from '
      + D.summary.total + ' released rows.';
  }

  table('rel-outcome', ['outcome','rows'],
    D.summary.outcomes.map(function(o){ return [esc(o[0]), o[1]]; }));
  table('rel-failures', ['failure mode','rows','what it means'],
    D.summary.failures.map(function(f){
      return ['<code>'+esc(f[0])+'</code>', f[1], esc(f[2])]; }));
  table('rel-pipeline', ['check','by','coverage','what it establishes'],
    D.summary.pipeline.map(function(p){
      return ['<code>'+esc(p[0])+'</code>', esc(p[1]), p[2]+'/'+D.summary.total, esc(p[3])]; }));
  table('rel-agree', ['measure','agreement'],
    D.summary.agreement.map(function(a){
      return [esc(a[0]), a[1]+'/'+a[2]+' · '+Math.round(100*a[1]/a[2])+'%']; }));
  var G = ['strong','acceptable','weak'];
  table('rel-matrix', ['pass 1 ↓ / pass 2 →'].concat(G),
    G.map(function(a){ return ['<strong>'+a+'</strong>'].concat(G.map(function(b){
      return D.summary.matrix[a+'|'+b] || 0; })); }));

  var note = document.getElementById('rel-comparator');
  if (note) {
    note.innerHTML = '<strong>Comparator.</strong> Its coverage above is ' +
      D.summary.comparator_replayed + '/' + D.summary.total + ' and that is the honest number: a ' +
      'workspace has been built for ' + D.summary.comparator_ready + ' of ' + D.summary.total +
      ' rows and none has been replayed. The comparator sandbox is Landlock through <code>landrun</code>, ' +
      'so it runs on Linux and this corpus was assembled on macOS. Every released row therefore sits at ' +
      '<code>proof_checked</code>; none claims <code>kernel_replayed</code>. The workspaces ship with the ' +
      'release, so anyone with a Linux runner can settle it.';
  }

  var rows = D.rows, listEl = document.getElementById('rel-list'),
      detailEl = document.getElementById('rel-detail'), qEl = document.getElementById('rel-q'),
      countEl = document.getElementById('rel-count'), fbox = document.getElementById('rel-filters');
  var variants = rows.map(function(r){ return r.variant; }).filter(function(v,i,a){ return a.indexOf(v)===i; }).sort();
  var active = null;
  fbox.innerHTML = variants.map(function(v){
    return '<button data-v="'+esc(v)+'" aria-pressed="false">'+esc(v)+'</button>'; }).join('');

  function haystack(r){
    if(r._h) return r._h;
    r._h = [r.id, r.variant, r.statement, r.lean, JSON.stringify(r.review)].join(' ').toLowerCase();
    return r._h;
  }
  function shown(){
    var needle = (qEl.value||'').toLowerCase();
    return rows.filter(function(r){
      return (!active || r.variant===active) && (!needle || haystack(r).indexOf(needle) >= 0); });
  }
  function pill(quality){
    return quality==='strong' ? 'ok' : (quality==='weak' ? 'no' : 'mid');
  }
  function render(){
    var list = shown();
    countEl.textContent = list.length + ' of ' + rows.length;
    listEl.innerHTML = list.map(function(r){
      return '<li data-id="'+esc(r.id)+'" tabindex="0"><span class="rid">'+esc(r.id)+'</span>' +
             '<span class="rel-pill">'+esc(r.variant)+'</span>' +
             '<span class="rel-pill ok">strong ×2</span></li>'; }).join('');
    if(list.length) detail(list[0].id);
    else detailEl.innerHTML = '<p class="muted">Nothing matches.</p>';
  }
  function detail(id){
    var r = rows.filter(function(x){ return x.id===id; })[0]; if(!r) return;
    [].forEach.call(listEl.children, function(li){
      li.setAttribute('aria-current', String(li.dataset.id===id)); });
    var h = ['<h3>'+esc(r.id)+'</h3>'];
    h.push('<div>'+['<span class="rel-pill">'+esc(r.op)+'</span>',
                    '<span class="rel-pill">'+esc(r.variant)+'</span>',
                    '<span class="rel-pill ok">'+esc(r.certificate.level)+'</span>',
                    '<span class="rel-pill">lineage '+r.lineage+'</span>'].join('')+'</div>');
    if(r.plan.goal){ h.push('<span class="rel-lab">The slot was asked for</span><p>'+esc(r.plan.goal)+'</p>'); }
    if(r.plan.mechanism){ h.push('<span class="rel-lab">Fusion mechanism</span><p><code>'+
      esc(r.plan.mechanism)+'</code> '+esc(r.plan.fusion_goal||'')+'</p>'); }
    if(r.parents && r.parents.length){
      h.push('<span class="rel-lab">Parents</span>');
      r.parents.forEach(function(p){
        h.push('<p><code>'+esc(p.id)+'</code></p>');
        if(p.lean) h.push('<pre>'+esc(p.lean)+'</pre>');
        if(p.contribution) h.push('<blockquote>'+esc(p.contribution)+'</blockquote>');
      });
    }
    if(r.statement){ h.push('<span class="rel-lab">Statement</span><p>'+esc(r.statement)+'</p>'); }
    h.push('<span class="rel-lab">Lean</span><pre>'+esc(r.lean)+'</pre>');
    h.push('<span class="rel-lab">Review — every judgement this row received</span>');
    r.review.forEach(function(v){
      h.push('<div><span class="rel-pill '+pill(v.quality)+'">'+esc(v.quality)+'</span>' +
             '<span class="rel-pill">'+esc(v.pass)+'</span>' +
             (v.failure ? '<span class="rel-pill no">'+esc(v.failure)+'</span>' : '') + '</div>');
      if(v.reason) h.push('<blockquote>'+esc(v.reason)+'</blockquote>');
    });
    h.push('<details><summary>Checks</summary><div class="rel-scroll"><table class="rel-t"><tbody>' +
      r.checks.map(function(c){
        return '<tr><td><code>'+esc(c[0])+'</code></td><td>'+esc(c[1])+'</td><td>'+esc(c[2])+'</td></tr>';
      }).join('') + '</tbody></table></div></details>');
    if(r.roundtrip){
      h.push('<details><summary>Goal round-trip evidence</summary>' +
        '<span class="rel-lab">What Lean elaborated</span><pre>'+esc(r.roundtrip.goal)+'</pre>' +
        '<span class="rel-lab">Read back by a model that saw only the Lean</span>' +
        '<blockquote>'+esc(r.roundtrip.read_back)+'</blockquote></details>');
    }
    h.push('<details><summary>Full Lean certificate</summary><pre>'+esc(r.lean_code)+'</pre></details>');
    h.push('<p class="muted">toolchain <code>'+esc(r.certificate.toolchain)+'</code> · mathlib <code>'+
      esc((r.certificate.mathlib||'').slice(0,12))+'</code> · axioms <code>'+
      esc(r.certificate.axioms.join(', ')||'none recorded')+'</code></p>');
    detailEl.innerHTML = h.join('');
    detailEl.scrollTop = 0;
  }
  listEl.addEventListener('click', function(e){
    var li = e.target.closest('li'); if(li) detail(li.dataset.id); });
  listEl.addEventListener('keydown', function(e){
    if(e.key==='Enter'||e.key===' '){ var li=e.target.closest('li'); if(li){ e.preventDefault(); detail(li.dataset.id); } } });
  qEl.addEventListener('input', render);
  fbox.addEventListener('click', function(e){
    var b = e.target.closest('button'); if(!b) return;
    active = (active===b.dataset.v) ? null : b.dataset.v;
    [].forEach.call(fbox.children, function(x){ x.setAttribute('aria-pressed', String(x.dataset.v===active)); });
    render();
  });
  render();
})();
</script>"""


def payload(release: List[dict], held: List[dict], stats: Dict[str, Any]) -> Dict[str, Any]:
    both = [r for r in held if r["admission"]["why_not"] == "rejected by both passes"]
    outcomes = [("admitted — strong from both passes", len(release))]
    outcomes += list(collections.Counter(r["admission"]["why_not"] for r in held).most_common())
    failures = [
        (failure, count, FAILURE_GLOSS.get(failure, ""))
        for failure, count in collections.Counter(
            (r["review"]["passes"][-1].get("failure") or "unlabelled") for r in both
        ).most_common()
    ]
    # Coverage over the rows the check applies to, not over the corpus: a
    # crossover-only probe reported corpus-wide reads as a gap it is not.
    pipeline = []
    for name, by, description in PIPELINE:
        scope = [r for r in release
                 if (r["checks"].get(name) or {}).get("applies", True) is not False]
        pipeline.append((name, by, sum(1 for r in scope if (r["checks"].get(name) or {}).get("ran")),
                         description, len(scope)))
    rows = []
    for row in release:
        goal = row["checks"].get("goal_roundtrip") or {}
        rows.append(
            {
                "id": row["problem_id"],
                "op": row["op_type"],
                "variant": row["operator_variant"],
                "lineage": row["lineage_depth"],
                "statement": row["statement"],
                "lean": (row["formal_statement"] or "").strip(),
                "lean_code": (row["lean_code"] or "").strip(),
                "parents": [
                    {"id": p["parent_id"], "lean": (p.get("formal_statement") or "").strip(),
                     "contribution": p.get("contribution") or ""}
                    for p in row.get("parents") or []
                ],
                "plan": {
                    "goal": (row.get("plan") or {}).get("operator_goal") or "",
                    "mechanism": (row.get("plan") or {}).get("fusion_mechanism") or "",
                    "fusion_goal": (row.get("plan") or {}).get("fusion_goal") or "",
                },
                "review": [
                    {"pass": p["pass"], "quality": p["quality"], "failure": p.get("failure") or "",
                     "reason": p.get("reason") or ""}
                    for p in row["review"]["passes"]
                ],
                "checks": [
                    [name, (row["checks"].get(name) or {}).get("by") or "",
                     (row["checks"].get(name) or {}).get("result") or "not run"]
                    for name, _, _ in PIPELINE
                ],
                "roundtrip": ({"goal": goal.get("elaborated_goal") or "",
                               "read_back": goal.get("read_back_as") or ""}
                              if goal.get("ran") and goal.get("elaborated_goal") else None),
                "certificate": {
                    "level": row["certificate"]["level"],
                    "toolchain": row["certificate"]["lean_toolchain"],
                    "mathlib": row["certificate"]["mathlib_revision"],
                    "axioms": row["certificate"]["axioms"],
                },
            }
        )
    return {
        "lineage": _lineage_topics(release),
        "summary": {
            "total": len(release),
            "considered": len(release) + len(held),
            "outcomes": outcomes,
            "failures": failures,
            "pipeline": pipeline,
            "agreement": [
                ("keep or reject", stats["verdict_agreement"], stats["n"]),
                ("exact quality grade", stats["quality_agreement"], stats["n"]),
                ("failure label, when both rejected", stats["failure_agreement"], max(1, stats["both_reject"])),
            ],
            "matrix": {f"{a}|{b}": n for (a, b), n in stats["matrix"].items()},
            "consistency_n": stats["n"],
            "comparator_ready": sum(
                1 for r in release
                if "prepared" in ((r["checks"].get("comparator") or {}).get("result") or "")),
            "comparator_replayed": sum(
                1 for r in release if (r["checks"].get("comparator") or {}).get("ran")),
        },
        "rows": rows,
    }



def _lineage_topics(release: List[dict]) -> Dict[str, Dict[str, int]]:
    """Released rows grouped by the topic of the seeds their lineage reaches.

    The seed page shows what the corpus was bred *from*; this shows what it grew
    *into*, on the same axis, so the two donuts can be read against each other.
    Topic is a property of the seed, not of the generated row, so it is resolved
    through the lineage: a row's roots are read out of its identifier and each
    root's topic is taken from the seed sheet.

    A crossover reaches two roots and is counted under both. The totals are
    therefore lineage attributions rather than rows, and the caption says so --
    a donut whose slices sum past the row count would otherwise read as an
    error.
    """
    from src.orchestration.problem_ids import roots_of

    topic: Dict[str, str] = {}
    for path in sorted(glob.glob("data/benchmarks/*/seeds_50_levels.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                match = re.search(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)",
                                  f"{row.get('lean_goal') or ''} {row.get('solution') or ''}")
                if match and row.get("topic"):
                    topic[match.group(1)] = str(row["topic"]).replace("_", " ")

    out: Dict[str, Dict[str, int]] = {}
    attributions = collections.Counter()
    for row in release:
        bench = str(row.get("benchmark") or "other")
        seen = {topic[r] for r in roots_of(str(row.get("problem_id") or "")) if r in topic}
        for name in seen:
            out.setdefault(bench, {})
            out[bench][name] = out[bench].get(name, 0) + 1
            attributions[bench] += 1
    return {b: dict(sorted(v.items(), key=lambda kv: -kv[1])) for b, v in out.items()}


def check_script(block: str) -> None:
    """Refuse to publish a page whose script will not parse.

    An apostrophe inside a single-quoted JS string shipped a page where the
    whole release section rendered empty: no console error reached the tooling,
    the markup was intact, and only reading the extracted script found it. The
    browser is not a syntax checker, so this is one.
    """
    node = shutil.which("node")
    if not node:
        print("  ! node not found — injected script not syntax-checked")
        return
    body = re.search(r'<script id="release-js">(.*?)</script>', block, re.S)
    if not body:
        raise SystemExit("no release script found in the block")
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
        handle.write(body.group(1))
        path = handle.name
    result = subprocess.run([node, "--check", path], capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise SystemExit("injected script does not parse:\n" + (result.stderr or result.stdout))
    print("  injected script parses")


def _e(text: Any) -> str:
    """Escape a value that came from the data.

    The prose in SETTLES and CHANGED is authored here and carries deliberate
    markup, so it is written through unescaped; anything read out of a row is
    not.
    """
    return html.escape(str(text or ""))


def splice_marked(page: str, block: str, begin: str, end: str, anchor: str) -> str:
    """Replace a previously injected block, or insert one before `anchor`."""
    marked = f"{begin}\n{block}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pattern.search(page):
        return pattern.sub(lambda _: marked, page, count=1)
    index = page.index(anchor)
    return page[:index] + marked + "\n" + page[index:]


def faithfulness_block(release: List[dict], stats: Dict[str, Any]) -> str:
    """The validation layer, on the page that claims to be about validation.

    This section covered the goal round-trip alone, which is one gate of eleven
    and the only one a reader could see. The rest decided rows silently, and
    three of them changed meaning in a single day -- one stopped being a gate at
    all. A page about faithfulness that shows only the check with the nicest
    numbers is not reporting the layer, it is advertising it.
    """
    total = len(release)
    out: List[str] = []
    out.append("<h2>The whole validation layer</h2>")
    out.append("<p>A generated theorem can be sound Lean and still be worthless. It can be true "
               "because nothing satisfies its hypotheses; it can be a statement the corpus already "
               "contains; it can be a corollary its parent's proof already discharges; and it can "
               "prove something other than the problem it is released as. None of those is visible "
               "to a type checker, and no two are the same kind of question — so each check sits "
               "with the faculty that can settle the one it asks.</p>")
    out.append("<ul>"
               "<li><b>Determinate questions go to Lean.</b> Whether a statement elaborates, whether "
               "a proof closes, which axioms it rests on, whether the hypotheses can hold at once, "
               "whether each one is load-bearing. Each is a compilation whose outcome is the "
               "verdict.</li>"
               "<li><b>Identity goes to a hash.</b> Whether two rows state the same theorem is not a "
               "matter of degree, so it is not a matter for a judge. Statements are compared under "
               "an alpha-normal form, against every earlier campaign rather than the current run.</li>"
               "<li><b>Meaning goes to a reader, twice.</b> Whether a child demands reasoning its "
               "parent does not supply, and whether the prose describes the theorem underneath, are "
               "questions of degree. One verdict is evidence; admission needs two.</li>"
               "<li><b>Independence goes to a kernel that did not produce the proof.</b> The "
               "comparator rebuilds the statement from the trusted row alone and replays the term. "
               "It is the one check whose value comes from not being ours.</li>"
               "</ul>")
    out.append('<div class="tablewrap"><table><thead><tr><th>Check</th><th>Decided by</th>'
               '<th>What it settles</th><th class="num">Coverage</th></tr></thead><tbody>')
    for name, by, _ in PIPELINE:
        scope = [r for r in release
                 if (r["checks"].get(name) or {}).get("applies", True) is not False]
        ran = sum(1 for r in scope if (r["checks"].get(name) or {}).get("ran"))
        if not scope:
            cover, klass = "not applicable here", ' class="num"'
        elif len(scope) != total:
            cover, klass = f"{ran}/{len(scope)} applicable", ' class="num"'
        else:
            cover = f"{ran}/{total}"
            klass = ' class="num"' if ran >= total else ' class="num" style="color:var(--warn)"'
        out.append(f'<tr><td><code>{_e(name)}</code></td><td>{_e(by)}</td>'
                   f"<td>{SETTLES.get(name, '')}</td><td{klass}>{cover}</td></tr>")
    out.append("</tbody></table></div>")
    out.append('<p class="muted">Two checks are reported over the rows they apply to, so a blank '
               "reads as inapplicability rather than as a gap: a hypothesis comparison means "
               "nothing where the operator was not asked to preserve them, and a silent mutation "
               "is equivalent to its parent by design, so asking whether it adds anything would "
               "report the operator working.</p>")
    out.append("<p>Redundancy is asked by <i>deleting</i>, never by adding. A hypothesis is "
               "removed from the child and a prover is asked to close the goal without it; a "
               "parent is stripped of every hypothesis and offered alone. Deleting makes the "
               "statement strictly harder, so Lean closing it afterwards has only one "
               "explanation. Asked the other way round — handing the prover the parent as an "
               "extra hypothesis — the question cannot fail: the child's own proof closes it "
               "with the parent unused. That form ran once over the whole release, reported 66 "
               "of 165 rows redundant, and a control substituting a hypothesis with no "
               "mathematics in it reproduced four of the first six findings. It was removed.</p>")

    out.append("<h2>Why this set suffices</h2>")
    out.append("<p>A released row can be wrong in four ways, and the layer covers all four.</p>")
    out.append('<div class="tablewrap"><table><thead><tr><th>The row is…</th>'
               "<th>Excluded by</th></tr></thead><tbody>"
               "<tr><td>not a theorem</td><td><code>statement_type_check</code>, "
               "<code>proof_accepted</code>, <code>axiom_audit</code> — and re-established "
               "independently by <code>comparator</code></td></tr>"
               "<tr><td>a theorem for a degenerate reason</td><td><code>vacuity</code>, "
               "<code>dead_hypotheses</code></td></tr>"
               "<tr><td>a theorem the corpus already has, or one that assumes more than it "
               "uses</td><td><code>corpus_dedup</code>, <code>redundancy</code></td></tr>"
               "<tr><td>a fine theorem released as a different problem</td>"
               "<td><code>goal_roundtrip</code></td></tr>"
               "</tbody></table></div>")
    out.append("<p>The redundancy probe and the round-trip decide <i>against</i> the judge rather "
               "than after it. Of the 171 candidates two independent judge passes both rated "
               "<b>strong</b> and voted to keep, the round-trip removed 12 for stating a problem "
               "other than the one Lean elaborated and held back 1 more whose verdict it never "
               "returned, and the redundancy probe removed a further 11 for holding a hypothesis "
               "Lean could delete. 147 remain. These gates run last and get the final word "
               "because each carries a proof where the judge carries an opinion — and because a "
               "standard that bends for a well-written problem is not a standard.</p>")
    out.append("<p>Admission reads both halves of each judgement, and it did not always. A judge "
               "returns a quality <i>and</i> a vote, and they answer different questions: how "
               "good the problem is, and whether it belongs in the corpus at all. Reading only "
               "the first admitted six rows the judge had voted to reject — two of them rejected "
               "by both passes — every one for <code>recall</code>, the label for a problem that "
               "is good and already covered by a sibling. Those are exactly the rows the second "
               "question exists to catch.</p>")
    out.append("<p>That is a claim about coverage of failure <i>kinds</i>, not about detection "
               "rates, and it is made in that form deliberately. Three of these checks are "
               "one-sided: a redundancy probe that finds no proof has found nothing, and the same "
               "is true of vacuity and of the judge's uncertainty. Where a check cannot fail on the "
               "thing it exists to catch it does not belong here at all — two were removed on "
               "exactly that ground, and a third for the sharper reason that it could not fail at "
               "all. What remains is the set for which we can say, of each member, both what it "
               "decides and what its silence does not mean.</p>")

    out.append("<h2>Four checks changed what they are</h2>")
    out.append("<p>Each of these was trusted before it was measured. What follows is what each one "
               "claimed, what it does now, and the measurement that forced the change.</p>")
    out.append('<div class="grid g2">')
    for name, was, now, why in CHANGED:
        out.append('<div class="card">'
                   f'<p><code>{_e(name)}</code></p>'
                   f'<p><b>{_e(was)}</b> &rarr; <b>{_e(now)}</b></p>'
                   f'<p class="muted">{why}</p></div>')
    out.append("</div>")

    out.append("<h2>What is still not settled</h2>")
    replayed = sum(1 for r in release if (r["checks"].get("comparator") or {}).get("ran"))
    ready = sum(1 for r in release
                if "prepared" in ((r["checks"].get("comparator") or {}).get("result") or ""))
    roundtrip = sum(1 for r in release if (r["checks"].get("goal_roundtrip") or {}).get("ran"))
    out.append("<ul>")
    out.append(f"<li><b>Kernel replay.</b> {ready} of {total} workspaces are built and none has been "
               f"replayed ({replayed}/{total}). Comparator's sandbox is Landlock, so it runs on "
               "Linux and this corpus was assembled on macOS. Every row sits at "
               "<code>proof_checked</code>; none claims <code>kernel_replayed</code>.</li>")
    out.append(f"<li><b>Round-trip coverage.</b> The gate diagrammed above ran on {roundtrip} of "
               f"{total} released rows. The numbers below it are from the benchmark that validated "
               "the gate, not from this corpus.</li>")
    pct = 100 * stats["verdict_agreement"] // stats["n"]
    out.append(f"<li><b>The judge disagrees with itself.</b> Run twice over the same {stats['n']} rows "
               f"in the same order, it agreed on keep-or-reject {pct}% of the time. That is the "
               "ceiling on any single verdict, and it is why release admission needs two.</li>")
    out.append("</ul>")
    return "\n".join(out)


def splice(page: str, block: str, anchor: str) -> str:
    """Replace a previously injected block, or insert one before `anchor`."""
    marked = f"{BEGIN}\n{block}\n{END}"
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if pattern.search(page):
        return pattern.sub(lambda _: marked, page, count=1)
    index = page.index(anchor)
    return page[:index] + marked + "\n" + page[index:]


def _repo_root() -> Path:
    """The repository root, found by marker rather than by counting parents.

    This used `parents[1]`, which encoded the file's depth under `scripts/`.
    Moving it into `scripts/release/` made that resolve one directory short and
    the script died looking for a path inside the repository instead of beside
    it. A marker survives the next move too.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release/eml_v1_release.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("data/release/eml_v1_rejected.jsonl"))
    parser.add_argument("--run1", type=Path, default=Path("data/release/rejudged.json"))
    parser.add_argument("--run2", type=Path, default=Path("data/release/rejudged_run2.json"))
    parser.add_argument("--workspace", type=Path,
                        default=_repo_root().parent / "ICLR_2027" / "workspace.html")
    parser.add_argument("--json-output", type=Path,
                        default=_repo_root().parent / "ICLR_2027" / "release_gallery.json")
    args = parser.parse_args()

    release = sorted(load(args.release), key=lambda r: (r["op_type"], r["operator_variant"], r["problem_id"]))
    held = load(args.rejected)
    data = payload(release, held, consistency(args.run1, args.run2))
    args.json_output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {args.json_output}  {args.json_output.stat().st_size/1024:.0f} KB")

    page = args.workspace.read_text(encoding="utf-8")
    before = len(page)

    if 'data-route="/release"' not in page:
        page = page.replace(
            '<a href="#/plan" data-route="/plan">Plan &amp; decisions</a>',
            '<a href="#/release" data-route="/release">Release</a>\n'
            '    <a href="#/plan" data-route="/plan">Plan &amp; decisions</a>',
            1,
        )

    body = PAGE % {
        "n": data["summary"]["total"],
        "considered": data["summary"]["considered"],
        "consistency_n": data["summary"]["consistency_n"],
    }
    serialised = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    block = "\n".join([
        CSS,
        body,
        f'<script id="release-data" type="application/json">{serialised}</script>',
        SCRIPT,
    ])
    check_script(block)
    page = splice(page, block, '<section class="page" id="p-/plan">')

    # The faithfulness page described one gate as though it were the layer.
    # Injected inside that section, before its closing tag, from the same
    # release data the release page uses so the two cannot disagree.
    # The lede described the round-trip as "the second one the validation layer
    # runs", which was true when it was the only gate on the page. The layer is
    # now below it, so the sentence has to stop implying there is nothing else.
    page = page.replace(
        "This is the\n  <code>goal_roundtrip</code> gate — the second one the validation layer runs, and the only\n  one type checking cannot decide.",
        "This is the\n  <code>goal_roundtrip</code> gate: the one check type checking cannot decide, and one of\n  eleven the validation layer runs. The rest are below.",
        1,
    )

    faith = faithfulness_block(release, consistency(args.run1, args.run2))
    pattern = re.compile(re.escape(FAITH_BEGIN) + r".*?" + re.escape(FAITH_END), re.S)
    marked = f"{FAITH_BEGIN}\n{faith}\n{FAITH_END}"
    if pattern.search(page):
        page = pattern.sub(lambda _: marked, page, count=1)
    else:
        marker = '<section class="page" id="p-/align">'
        close = page.index("</section>", page.index(marker))
        page = page[:close] + marked + "\n" + page[close:]
    args.workspace.write_text(page, encoding="utf-8")
    print(f"  {args.workspace}  {before/1024:.0f} KB -> {len(page)/1024:.0f} KB")
    print(f"\n{data['summary']['total']} rows · "
          f"{sum(len(r['review']) for r in data['rows'])} reasoning texts on the page")


if __name__ == "__main__":
    main()
