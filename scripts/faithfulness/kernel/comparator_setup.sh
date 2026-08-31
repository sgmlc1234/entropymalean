#!/bin/bash
# Install everything comparator needs on a bare Ubuntu box, then verify.
#
# The target had none of it: no elan, no lake, no lean, no comparator, no
# landrun, no lean4export. It did have `systemd-run` and `landlock` in
# /sys/kernel/security/lsm, which are the two things that cannot be installed.
#
# Re-runnable: every step checks for its own output first, so a failed run can
# be repeated without redoing the parts that succeeded. The expensive step is
# mathlib, and it is fetched from cache rather than built -- building it from
# source on one machine takes hours.
#
#   TOOLCHAIN   Lean version the batch was checked against
#   MATHLIB_REV Mathlib revision the batch was checked against
#   PREFIX      where sources and binaries go (default ~/comparator-toolchain)
set -uo pipefail

TOOLCHAIN="${TOOLCHAIN:-leanprover/lean4:v4.30.0-rc2}"
MATHLIB_REV="${MATHLIB_REV:-0fb2045029635862ffb234635a111c80a55e2a87}"
PREFIX="${PREFIX:-$HOME/comparator-toolchain}"
BIN="$PREFIX/bin"
mkdir -p "$PREFIX" "$BIN"

say() { printf '\n=== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
die() { echo "FAILED: $1" >&2; exit 1; }

say "0. prerequisites"
# Nothing here uses apt. The target has git, curl, gcc, make and python3 already
# but no passwordless sudo, so Go is installed from the official tarball into
# $PREFIX instead of from the archive. Everything else already builds in
# userspace, which means the whole setup needs no privileges at all.
for b in git curl gcc make python3; do
  have "$b" || die "$b is missing and cannot be installed without sudo"
done
if ! have go && [ ! -x "$PREFIX/go/bin/go" ]; then
  GO_VER="${GO_VER:-1.23.4}"
  echo "  installing go ${GO_VER} into $PREFIX (no sudo)"
  curl -sSfL "https://go.dev/dl/go${GO_VER}.linux-amd64.tar.gz" -o "$PREFIX/go.tgz" || die "download go"
  tar -C "$PREFIX" -xzf "$PREFIX/go.tgz" || die "extract go"
  rm -f "$PREFIX/go.tgz"
fi
export PATH="$PREFIX/go/bin:$PATH"
export GOPATH="${GOPATH:-$PREFIX/gopath}"
have go || die "go still not on PATH"
echo "  git $(git --version | awk '{print $3}') · go $(go version | awk '{print $3}')"

say "1. elan (Lean toolchain manager)"
if ! have elan && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain "$TOOLCHAIN" \
    || die "elan install"
fi
export PATH="$HOME/.elan/bin:$BIN:$PATH"
have lake || die "lake still not on PATH after elan install"
elan toolchain install "$TOOLCHAIN" >/dev/null 2>&1
echo "  lake $(lake --version 2>/dev/null | head -1)"

say "2. mathlib @ ${MATHLIB_REV:0:12} (from cache, not built)"
MATHLIB_ROOT="$PREFIX/mathlib4"
if [ ! -d "$MATHLIB_ROOT/.git" ]; then
  git clone -q https://github.com/leanprover-community/mathlib4.git "$MATHLIB_ROOT" || die "clone mathlib4"
fi
cd "$MATHLIB_ROOT" || die "cd mathlib4"
git fetch -q origin || die "fetch mathlib4"
git checkout -q "$MATHLIB_REV" || die "checkout $MATHLIB_REV — is the revision right for this clone?"
echo "$TOOLCHAIN" > lean-toolchain
lake exe cache get >/dev/null 2>&1 || echo "  ! cache get failed; the build below will be slow"
lake build >/dev/null 2>&1 || die "mathlib build"
MATHLIB_PKG="$MATHLIB_ROOT/.lake/packages/mathlib"
[ -d "$MATHLIB_PKG" ] || MATHLIB_PKG="$MATHLIB_ROOT"
echo "  mathlib package: $MATHLIB_PKG"

say "3. lean4export"
if [ ! -x "$BIN/lean4export" ]; then
  [ -d "$PREFIX/lean4export/.git" ] || \
    git clone -q https://github.com/leanprover/lean4export.git "$PREFIX/lean4export" || die "clone lean4export"
  cd "$PREFIX/lean4export" || die "cd lean4export"
  echo "$TOOLCHAIN" > lean-toolchain
  lake build >/dev/null 2>&1 || die "lean4export build"
  cp .lake/build/bin/lean4export "$BIN/" || die "install lean4export"
fi
echo "  $BIN/lean4export"

say "4. comparator"
if [ ! -x "$BIN/comparator" ]; then
  [ -d "$PREFIX/comparator/.git" ] || \
    git clone -q https://github.com/leanprover/comparator.git "$PREFIX/comparator" || die "clone comparator"
  cd "$PREFIX/comparator" || die "cd comparator"
  lake build >/dev/null 2>&1 || die "comparator build"
  cp .lake/build/bin/comparator "$BIN/" 2>/dev/null || cp .lake/build/bin/* "$BIN/" || die "install comparator"
fi
echo "  $BIN/comparator"

say "5. landrun (Landlock sandbox)"
# The release ships a static linux-amd64 binary, so this needs no Go at all;
# `go install` is kept only as a fallback. The first attempt used the module
# path `github.com/Zouuup/...` and failed: Go module paths are case-sensitive
# and the module is `github.com/zouuup/landrun/cmd/landrun`, even though the
# repository is spelled with a capital Z.
LANDRUN_VER="${LANDRUN_VER:-v0.1.17}"
if [ ! -x "$BIN/landrun" ]; then
  arch=$(uname -m)
  case "$arch" in
    x86_64|amd64) asset="landrun-linux-amd64" ;;
    aarch64|arm64) asset="landrun-linux-arm64" ;;
    *) asset="" ;;
  esac
  if [ -n "$asset" ] && curl -sSfL \
      "https://github.com/Zouuup/landrun/releases/download/${LANDRUN_VER}/${asset}" \
      -o "$BIN/landrun"; then
    chmod +x "$BIN/landrun"
    echo "  installed ${LANDRUN_VER} ${asset}"
  elif GOBIN="$BIN" go install "github.com/zouuup/landrun/cmd/landrun@latest" 2>/dev/null; then
    echo "  built from source via go install"
  else
    git clone -q https://github.com/Zouuup/landrun.git "$PREFIX/landrun" 2>/dev/null || true
    ( cd "$PREFIX/landrun" && go build -o "$BIN/landrun" cmd/landrun/main.go ) \
      || die "landrun: release download, go install and source build all failed"
    echo "  built from a source checkout"
  fi
fi
"$BIN/landrun" --help >/dev/null 2>&1 || "$BIN/landrun" --version >/dev/null 2>&1 \
  || echo "  ! landrun installed but does not run — check kernel Landlock support"
echo "  $BIN/landrun"

say "done"
cat <<EOF
Add this to ~/.bashrc so the batch can find everything:

  export PATH="\$HOME/.elan/bin:$BIN:\$PATH"

Then run the batch:

  export PATH="\$HOME/.elan/bin:$BIN:\$PATH"
  cd ~/comparator_bundle
  ./comparator_preflight.sh comparator
  COMPARATOR_MATHLIB=$MATHLIB_PKG ./run_comparator_batch.sh
EOF
