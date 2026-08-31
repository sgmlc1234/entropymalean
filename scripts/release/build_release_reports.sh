#!/bin/bash
# Compile the three release-report PDFs.
#
# Lived in /tmp until a reboot deleted it mid-task, which cost a rebuild that
# reported success while nothing ran: nohup wrote "No such file or directory"
# into the log and the stale PDFs stayed on disk looking finished. It lives in
# the repo now.
#
# lualatex, not pdflatex: the cards set Lean's Unicode. Twice per document, for
# the table of contents.
set -uo pipefail
cd "$(dirname "$0")/../docs"
for f in eml_v1_release_report_proofnet eml_v1_release_report_minif2f eml_v1_release_report; do
  for i in 1 2; do
    lualatex -interaction=nonstopmode "$f.tex" > "/tmp/lua_${f}_$i.log" 2>&1
  done
  pages=$(pdfinfo "$f.pdf" 2>/dev/null | awk '/Pages/{print $2}')
  miss=$(grep -ci "missing character" "/tmp/lua_${f}_2.log")
  echo "DONE $f pages=${pages:-0} missing_glyphs=$miss bytes=$(stat -f %z "$f.pdf" 2>/dev/null)"
done
echo "ALLDONE"
