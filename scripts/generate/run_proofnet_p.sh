#!/bin/bash
# Resume wrapper for the ProofNet p campaign.
#
# This lived in /tmp until the 2026-08-19 reboot deleted it mid-campaign, which
# cost the time it took to work out what arguments the run had been started
# with. It lives in the repo now so the answer is readable rather than
# reconstructed.
#
# The driver skips any group that already has a .jsonl, so re-running this after
# an interruption resumes at the first unwritten group rather than starting
# over. A group cut off mid-run leaves a partial .jsonl and would be skipped as
# though finished -- delete it before re-running. The 2026-08-19 interruption
# happened to land between groups, so there was nothing to delete.
set -uo pipefail
# Repository root, found by walking up to the marker rather than by
# counting directories: this file has moved once already and `..` was
# wrong the moment it did.
cd "$(cd "$(dirname "$0")" && until [ -f pyproject.toml ] || [ "$PWD" = / ]; do cd ..; done; pwd)"

OUT=data/certified/run-e
LOCK="$OUT/.driver.lock"

# One driver at a time. Two once ran together after I judged a live campaign
# dead from a `pgrep` that returned nothing during a generation gap; they took
# turns SIGTERMing each other. A stale lock from a killed or rebooted driver is
# reclaimed rather than treated as a live one.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "driver $(cat "$LOCK") is already running" >&2; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

EML_OUT="$OUT" EML_GENERATIONS=10 EML_ORDER=reverse \
  bash scripts/generate/run_eml1_main.sh
echo "WRAPPER: exited with $? at $(date '+%F %T')" | tee -a "$OUT/driver.log"
