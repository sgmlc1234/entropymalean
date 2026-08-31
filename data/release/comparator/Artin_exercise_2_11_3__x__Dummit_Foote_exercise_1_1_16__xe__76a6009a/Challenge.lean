import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false

theorem even_finite_group_has_self_inverse_involution {G : Type*} [Group G] [Fintype G]
  (hG : Even (Fintype.card G)) : ∃ x : G, x⁻¹ = x ∧ x ^ 2 = 1 := by
  sorry
