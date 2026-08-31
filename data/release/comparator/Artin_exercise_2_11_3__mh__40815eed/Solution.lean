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
  obtain ⟨k, hk⟩ := hG
  have h_two_dvd : 2 ∣ Fintype.card G := by
    exact ⟨k, by simpa [two_mul] using hk⟩
  let _ : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩
  obtain ⟨x, hx⟩ := exists_prime_orderOf_dvd_card (G := G) (p := 2) h_two_dvd
  have hsq : x * x = 1 := by
    have hp := pow_orderOf_eq_one x
    rw [hx] at hp
    simpa [pow_two] using hp
  have hxne : x ≠ 1 := by
    intro h
    subst x
    simpa using hx
  refine ⟨x, hx, (fun y => x * y), rfl, ?_, ?_⟩
  · intro y
    change x * (x * y) = y
    rw [← mul_assoc, hsq, one_mul]
  · intro y h
    have h' : x * y = 1 * y := by
      simpa using h
    exact hxne (mul_right_cancel h')
