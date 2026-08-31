import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false
open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators

theorem exists_subgroup_card_two_of_card_dvd_two {G : Type*} [Group G] [Fintype G]
  (hG : 2 ∣ Fintype.card G) : ∃ H : Subgroup G, Nat.card H = 2 := by
  sorry
