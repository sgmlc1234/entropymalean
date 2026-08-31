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
  intro φ h1 h2
  let K : Subgroup G := φ.ker
  have hgen :
      ({b * a * b ^ 2, b * a * b ^ 3} : Set G) ⊆ K := by
    intro x hx
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
    rcases hx with rfl | rfl
    · change φ (b * a * b ^ 2) = 1
      exact h1
    · change φ (b * a * b ^ 3) = 1
      exact h2
  have hle :
      Subgroup.closure ({b * a * b ^ 2, b * a * b ^ 3} : Set G) ≤ K :=
    (Subgroup.closure_le K).2 hgen
  have ha0 : a ∈ Subgroup.closure ({a, b} : Set G) :=
    Subgroup.subset_closure (by simp)
  have hb0 : b ∈ Subgroup.closure ({a, b} : Set G) :=
    Subgroup.subset_closure (by simp)
  have haR :
      a ∈ Subgroup.closure ({b * a * b ^ 2, b * a * b ^ 3} : Set G) := by
    rw [← hclosure]
    exact ha0
  have hbR :
      b ∈ Subgroup.closure ({b * a * b ^ 2, b * a * b ^ 3} : Set G) := by
    rw [← hclosure]
    exact hb0
  have haK : a ∈ K := hle haR
  have hbK : b ∈ K := hle hbR
  have hfa : φ a = 1 := by
    change φ a = 1 at haK
    exact haK
  have hfb : φ b = 1 := by
    change φ b = 1 at hbK
    exact hbK
  exact ⟨hfa, hfb⟩
