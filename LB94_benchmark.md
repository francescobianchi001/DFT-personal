# LB94 validation against van Leeuwen & Baerends (1994)

Reference: R. van Leeuwen and E. J. Baerends, *"Exchange-correlation potential
with correct asymptotic behavior"*, Phys. Rev. A **49**, 2421 (1994), Table I.

The table reports ionization energies / electron affinities taken as the
negative of the highest-occupied Kohn–Sham eigenvalue, `-ε(HOMO)`, in Hartree.
The **Model** column is the LB94 model potential — the quantity our `--LB94`
flag implements. We reproduce the LDA and LB94 columns from a self-consistent
free-atom solve (exponential grid, r ∈ [1e-6, 15] bohr, 3000 points).

## Closed-shell atoms

| Atom | HOMO | our LDA | ref LDA | **our LB94** | **ref Model** | Expt |
|------|------|--------:|--------:|-------------:|--------------:|-----:|
| He | 1s | 0.570 | 0.571 | 0.851 | 0.851 | 0.903 |
| Be | 2s | 0.206 | 0.206 | 0.320 | 0.321 | 0.343 |
| Ne | 2p | 0.498 | 0.490 | 0.782 | 0.788 | 0.792 |
| Ar | 3p | 0.382 | 0.381 | 0.579 | 0.577 | 0.579 |
| Kr | 4p | 0.346 | 0.346 | 0.529 | 0.529 | 0.517 |
| Xe | 5p | 0.310 | 0.310 | 0.473 | 0.474 | 0.446 |

Agreement with the paper's Model column is 1–6 mHa across the whole He→Xe
series. LB94 lifts `-ε(HOMO)` by ~0.1–0.3 Ha over LDA, toward experiment,
exactly as reported.

## Halide anions (electron affinities) and the H atom

For a negative ion, `-ε(HOMO)` of the anion is the electron affinity of the
neutral. These are closed-shell (noble-gas configurations), so the ρ_σ = ρ/2
assumption still holds. H is the exception: it is open-shell (one electron).

| Case | HOMO | our LDA | ref LDA | **our LB94** | **ref Model** | Expt |
|------|------|--------:|--------:|-------------:|--------------:|-----:|
| H    | 1s | 0.233 | 0.234 | 0.394 | 0.440 | 0.500 |
| F⁻   | 2p | 0.008 | −0.097 | 0.140 | 0.128 | 0.125 |
| Cl⁻  | 3p | 0.009 | −0.022 | 0.144 | 0.140 | 0.133 |
| Br⁻  | 4p | 0.007 | −0.008 | 0.141 | 0.140 | 0.124 |
| I⁻   | 5p | 0.010 | 0.005 | 0.139 | 0.139 | 0.112 |

**LB94 anions reproduce the paper's Model column to 1–12 mHa** (I⁻ exact, Br⁻/Cl⁻
within a few mHa, F⁻ the largest at 12 mHa). This is the key result: LB94's
−1/r tail binds the extra electron, so the anion HOMO is a genuine bound state
our solver brackets cleanly, landing right on the reference.

Two expected mismatches:

- **LDA anion column.** The paper's LDA electron affinities are negative or
  near-zero (F⁻ = −0.097): under LDA the anion HOMO is *unbound* (positive
  eigenvalue). Our free-atom solver only brackets bound states, so instead of a
  positive eigenvalue it returns a spuriously weakly-bound one (F⁻ = +0.008
  rather than −0.097). This is the known free-atom bracketing limitation, not an
  LB94 issue — and it is exactly why LB94 is needed for anions.

- **H atom (0.394 vs 0.440, 46 mHa).** Our LB94 hard-codes the closed-shell
  spin channel ρ_σ = ρ/2, which is wrong for a single electron (the true spin
  density is ρ_↑ = ρ). The paper does H spin-polarized. Closed-shell systems are
  unaffected; a spin-polarized ρ_σ would be needed to match the H row.


## Notes

- Our LB94 uses β = 0.05 and the closed-shell spin channel ρ_σ = ρ/2, matching
  the paper.
- The residual LB94-vs-experiment gap (e.g. Xe 0.473 vs 0.446) is expected: LB94
  is a model potential, not exact, and the paper shows the same residual.
