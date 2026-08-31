import Lake
open Lake DSL

package EntropyMaG where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib EntropyMaG where
  srcDir := "lean"

require repl from git
  "https://github.com/leanprover-community/repl" @ "master"
