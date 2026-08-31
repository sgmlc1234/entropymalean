#!/bin/bash
# Does this machine have what comparator needs? Run before the batch.
#
# comparator's sandbox is Landlock through `landrun`, so the whole gate is Linux
# only; `validate_comparator_runtime()` refuses anywhere else. This reports what
# is present and what is missing, and exits non-zero if the batch cannot run.
set -uo pipefail
missing=0
printf '  %-14s %s\n' "platform" "$(uname -s) $(uname -r)"
[ "$(uname -s)" = "Linux" ] || { echo "  ! not Linux — comparator cannot run here"; missing=1; }
for b in lake lean comparator landrun lean4export systemd-run; do
  path=$(command -v "$b" 2>/dev/null)
  if [ -n "$path" ]; then printf '  %-14s %s\n' "$b" "$path"
  else printf '  %-14s MISSING\n' "$b"; missing=$((missing+1)); fi
done
printf '  %-14s %s\n' "elan" "$(command -v elan 2>/dev/null || echo 'MISSING')"
printf '  %-14s %s\n' "kernel LSM" "$(cat /sys/kernel/security/lsm 2>/dev/null || echo 'unreadable')"
if [ -d "${1:-comparator}" ]; then
  printf '  %-14s %s workspaces\n' "batch" "$(find "${1:-comparator}" -name config.json | wc -l)"
  want_tc=$(cat "$(find "${1:-comparator}" -name lean-toolchain | head -1)" 2>/dev/null)
  printf '  %-14s %s\n' "batch needs" "${want_tc:-unknown}"
  have_tc=$(lake --version 2>/dev/null | head -1)
  [ -n "$have_tc" ] && printf '  %-14s %s\n' "installed" "$have_tc"
fi
# A mathlib at the wrong revision fails proofs for reasons that have nothing to
# do with the rows, and the failure looks identical to a real rejection.
if [ -n "${COMPARATOR_MATHLIB:-}" ] && [ -d "$COMPARATOR_MATHLIB" ]; then
  rev=$(git -C "$COMPARATOR_MATHLIB" rev-parse HEAD 2>/dev/null || echo unknown)
  printf '  %-14s %s\n' "mathlib rev" "$rev"
  printf '  %-14s %s\n' "batch built on" "0fb2045029635862ffb234635a111c80a55e2a87"
  [ "$rev" = "0fb2045029635862ffb234635a111c80a55e2a87" ] || \
    echo "  ! mathlib revision differs — proof failures may be environmental, not real"
fi
echo
if [ "$missing" -gt 0 ]; then
  echo "$missing prerequisite(s) missing. Install notes:"
  echo "  comparator   https://github.com/leanprover/comparator  (lake build, then put the binary on PATH)"
  echo "  landrun      https://github.com/Zouuup/landrun         (needs Linux 5.13+ with Landlock enabled)"
  echo "  lean4export  https://github.com/leanprover/lean4export"
  echo "  lake/lean    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"
  exit 2
fi
echo "ready — run ./run_comparator_batch.sh"
