import Mathlib
import Aesop
set_option maxHeartbeats 2000000
set_option autoImplicit false

open scoped BigOperators
open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd

theorem finite_field_prod_units_root_characterization {K : Type*} [Field K] [Fintype Kˣ] :
  (∏ x : Kˣ, x) = 1 ∨
    ((∏ x : Kˣ, x) ≠ 1 ∧ ∀ z : Kˣ, z ^ 2 = 1 → z = 1 ∨ z = ∏ x : Kˣ, x) := by
  classical
  have hprod : (∏ x : Kˣ, x) = (-1 : Kˣ) :=
    FiniteField.prod_univ_units_id_eq_neg_one (K := K)
  have hroot : ∀ z : Kˣ, z ^ 2 = 1 → z = 1 ∨ z = (-1 : Kˣ) := by
    intro z hz
    have hzK : (z : K) ^ 2 = 1 := by
      simpa using congrArg (fun u : Kˣ => (u : K)) hz
    have hfac : ((z : K) - 1) * ((z : K) + 1) = 0 := by
      calc
        ((z : K) - 1) * ((z : K) + 1) = (z : K) ^ 2 - 1 := by ring
        _ = 0 := by rw [hzK]; ring
    rcases mul_eq_zero.mp hfac with hzero | hzero
    · left
      apply Units.ext
      have hval : (z : K) = 1 := by linear_combination hzero
      simpa using hval
    · right
      apply Units.ext
      have hval : (z : K) = -1 := by linear_combination hzero
      simpa using hval
  rw [hprod]
  by_cases heq : (-1 : Kˣ) = 1
  · exact Or.inl heq
  · right
    constructor
    · exact heq
    · intro z hz
      exact hroot z hz
