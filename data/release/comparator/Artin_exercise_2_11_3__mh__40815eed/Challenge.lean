import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false
open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators

theorem even_finite_group_has_fixed_point_free_left_involution {G : Type*} [Group G] [Fintype G]
  (hG : Even (Fintype.card G)) :
  ∃ x : G, orderOf x = 2 ∧ ∃ f : G → G,
    f = (fun y => x * y) ∧ Function.Involutive f ∧ ∀ y : G, f y ≠ y := by
  sorry
