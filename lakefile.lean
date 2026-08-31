import Lake
open Lake DSL

package EntropyMaLean where

-- No first-party Lean library. Every check in this repository runs
-- `lake env lean <file>` over a generated file or a comparator workspace, so the
-- package exists to pin and provide Mathlib, not to build sources of its own.
-- A `lean_lib` target was declared here previously against an empty `lean/`
-- directory, which git does not track: the target could not build, and
-- `lake build` -- the setup step this repository asks reviewers to run -- failed
-- on it rather than on anything real.

-- Pinned in source as well as in `lake-manifest.json`. The manifest is what
-- `lake build` honours; the literal here is so a `lake update` cannot quietly
-- move the corpus off the revision it was certified against.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "0fb2045029635862ffb234635a111c80a55e2a87"

require repl from git
  "https://github.com/leanprover-community/repl" @ "f0a88bfca1fa6ac75e2a33d1678a3b67c90fb492"
