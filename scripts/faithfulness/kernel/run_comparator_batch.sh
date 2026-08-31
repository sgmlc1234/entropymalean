#!/bin/bash
# Run comparator over every prepared workspace. Linux only.
#
# `scripts/faithfulness/kernel/prepare_comparator_batch.py` builds the workspaces anywhere; this
# runs them, and it cannot run on macOS: comparator's sandbox is Landlock
# through `landrun`, so `validate_comparator_runtime()` refuses on any other
# platform. That is the gap between `proof_checked`, which every released row
# already has, and `kernel_replayed`, which none of them do.
#
# What comparator settles that `lake env lean` does not: the Solution proves the
# *same statement* as the Challenge (compared by lean4export, not by trusting the
# file), it uses only the permitted axioms, and the Lean kernel accepts it.
#
#   COMPARATOR_WORKSPACES  directory of prepared workspaces
#   COMPARATOR_OUT         where the per-row verdicts land
#   COMPARATOR_TIMEOUT     seconds per workspace
set -uo pipefail
# Repository root, found by walking up to the marker rather than by
# counting directories: this file has moved once already and `..` was
# wrong the moment it did.
cd "$(cd "$(dirname "$0")" && until [ -f pyproject.toml ] || [ "$PWD" = / ]; do cd ..; done; pwd)"

WORKSPACES="${COMPARATOR_WORKSPACES:-data/release/comparator}"
OUT="${COMPARATOR_OUT:-data/release/comparator_results.json}"
TIMEOUT="${COMPARATOR_TIMEOUT:-900}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "comparator requires Linux (landrun/Landlock); this is $(uname -s)." >&2
  echo "Workspaces under $WORKSPACES are portable — run this on a Linux runner." >&2
  exit 2
fi
for binary in comparator landrun lean4export lake systemd-run; do
  command -v "$binary" >/dev/null || { echo "missing: $binary" >&2; exit 2; }
done

# Each workspace pins mathlib by absolute path, resolved against the machine
# that built the batch. That was macOS; this is not. Rewriting on arrival is
# what keeps the bundle portable.
if [ -z "${COMPARATOR_MATHLIB:-}" ]; then
  echo "Set COMPARATOR_MATHLIB to this machine's mathlib package directory, e.g." >&2
  echo "  COMPARATOR_MATHLIB=\$HOME/mathlib4/.lake/packages/mathlib $0" >&2
  exit 2
fi
python3 "$(dirname "$0")/comparator_repath.py" "$WORKSPACES" "$COMPARATOR_MATHLIB" || exit 2

total=0; passed=0
echo "[" > "$OUT"
first=1
for workspace in "$WORKSPACES"/*/; do
  [ -f "$workspace/config.json" ] || continue
  total=$((total + 1))
  name=$(basename "$workspace")
  log=$(cd "$workspace" && timeout "$TIMEOUT" lake env comparator config.json 2>&1)
  code=$?
  [ $code -eq 0 ] && passed=$((passed + 1))
  [ $first -eq 0 ] && echo "," >> "$OUT"
  first=0
  python3 - "$name" "$code" "$log" >> "$OUT" <<'PY'
import json, sys
print(json.dumps({"problem_id": sys.argv[1], "returncode": int(sys.argv[2]),
                  "kernel_replayed": int(sys.argv[2]) == 0,
                  "log": sys.argv[3][-2000:]}, ensure_ascii=False), end="")
PY
  printf '  %-4s %s\n' "$([ $code -eq 0 ] && echo ok || echo FAIL)" "$name"
done
echo "]" >> "$OUT"

echo
echo "kernel_replayed $passed/$total"
echo "written: $OUT"
echo "Feed it back with: python3 scripts/release/export_release.py --comparator $OUT"
