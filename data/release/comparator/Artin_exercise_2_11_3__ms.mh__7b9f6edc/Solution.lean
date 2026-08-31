import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false
open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators

theorem exists_subgroup_card_two_of_card_dvd_two {G : Type*} [Group G] [Fintype G]
  (hG : 2 ∣ Fintype.card G) : ∃ H : Subgroup G, Nat.card H = 2 := by
  let _ : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩
  obtain ⟨x, hx⟩ := exists_prime_orderOf_dvd_card (G := G) (p := 2) hG
  let H : Subgroup G := Subgroup.zpowers x
  have hcard : Nat.card H = orderOf x := by
    dsimp [H]
    rw [Nat.card_eq_fintype_card, Fintype.card_zpowers]
  have hH : Nat.card H = 2 := by
    calc
      Nat.card H = orderOf x := hcard
      _ = 2 := hx
  exact ⟨H, hH⟩
