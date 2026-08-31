import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false
open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators

theorem conjugate_generator_kernel_criterion {G H : Type*} [Group G] [Group H]
    (a b : G)
    (hclosure :
      Subgroup.closure ({a, b} : Set G) =
        Subgroup.closure ({b * a * b ^ 2, b * a * b ^ 3} : Set G)) :
    ∀ φ : G →* H,
      φ (b * a * b ^ 2) = 1 →
      φ (b * a * b ^ 3) = 1 →
      φ a = 1 ∧ φ b = 1 := by
  sorry
