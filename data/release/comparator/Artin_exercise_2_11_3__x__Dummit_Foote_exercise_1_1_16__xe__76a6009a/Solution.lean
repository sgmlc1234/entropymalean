import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false

theorem even_finite_group_has_self_inverse_involution {G : Type*} [Group G] [Fintype G]
  (hG : Even (Fintype.card G)) : ∃ x : G, x⁻¹ = x ∧ x ^ 2 = 1 := by
  obtain ⟨k, hk⟩ := hG
  have h_two_dvd : 2 ∣ Fintype.card G := by
    exact ⟨k, by simpa [two_mul] using hk⟩
  let _ : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩
  obtain ⟨x, hx⟩ := exists_prime_orderOf_dvd_card (G := G) (p := 2) h_two_dvd
  have hchar : ∀ y : G, y ^ 2 = 1 ↔ (orderOf y = 1 ∨ orderOf y = 2) := by
    intro y
    constructor
    · intro hy
      have hy_dvd : orderOf y ∣ 2 := by
        exact (orderOf_dvd_iff_pow_eq_one (x := y) (n := 2)).mpr (by simpa using hy)
      exact (Nat.dvd_prime Nat.prime_two).1 hy_dvd
    · intro hy
      rcases hy with hy | hy
      · have hpow : y ^ orderOf y = 1 := pow_orderOf_eq_one (x := y)
        have hy_eq_one : y = 1 := by
          have h : y ^ 1 = 1 := by simpa [hy] using hpow
          simpa using h
        simp [hy_eq_one]
      · have hpow : y ^ orderOf y = 1 := pow_orderOf_eq_one (x := y)
        simpa [hy] using hpow
  have hsq : x ^ 2 = 1 := (hchar x).mpr (Or.inr hx)
  have hinv : x⁻¹ = x := by
    have hmul : x * x = 1 := by simpa [pow_two] using hsq
    calc
      x⁻¹ = x⁻¹ * 1 := by simp
      _ = x⁻¹ * (x * x) := by rw [hmul]
      _ = x := by simp [mul_assoc]
  exact ⟨x, hinv, hsq⟩
