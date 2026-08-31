#!/usr/bin/env bash
set -euo pipefail

# Lean 4.30.0-rc2-compatible reference checkouts for this repository.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference_root="${repo_root}/references"

comparator_rev="71b52ec29e06d4b7d882726553b1ceb99a2499e0"
lean4export_rev="12581a6b680d8478175596338eb2d53383a323e3"
lean_eval_rev="2a12548d8b599c9c2e7c3ce8edfea0a8b97b5ab7"

fetch_checkout() {
  local url="$1"
  local destination="$2"
  local revision="$3"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
  fi
  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    echo "refusing to replace dirty reference checkout: ${destination}" >&2
    exit 1
  fi
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
}

mkdir -p "${reference_root}"
fetch_checkout \
  "https://github.com/leanprover/comparator.git" \
  "${reference_root}/lean-comparator" \
  "${comparator_rev}"
fetch_checkout \
  "https://github.com/leanprover/lean4export.git" \
  "${reference_root}/lean4export" \
  "${lean4export_rev}"
fetch_checkout \
  "https://github.com/leanprover/lean-eval.git" \
  "${reference_root}/lean-eval-official" \
  "${lean_eval_rev}"

echo "Fetched pinned comparator references under ${reference_root}"
