# Atomic DFT — Kohn-Sham LDA Solver for Atoms

A Python implementation of a radial Kohn-Sham density functional theory (DFT) solver for atoms up to Nickel (Z=28). Uses the local density approximation (LDA) with the Vosko-Wilk-Nusair (VWN) correlation functional and a Numerov method for radial integration.

## Features

- Kohn-Sham self-consistent field (SCF) solver on a uniform radial grid
- Numerov integration with left/right matching at the classical turning point
- VWN correlation + Slater exchange (LDA)
- Aufbau filling with Slater screening rules for starting energies
- Numba JIT-compiled Numerov propagation (~200x speedup over pure Python)
- Vectorized Hartree potential via cumulative trapezoid integration
- 2D and 3D orbital density visualization
- Save/load wavefunctions for fast replotting

## Requirements

```
numpy
scipy
matplotlib
numba
scikit-image   # only for 3D isosurface plots
```

## Files

| File | Description |
|------|-------------|
| `atomDFT.py` | Main solver (optimized with Numba) |
| `atomDFT_original.py` | Reference solver (pure Python, no Numba) |
| `diagnosis.py` | Run DFT, print all energy components, plot wavefunctions |
| `allplotter.py` | 2D/3D orbital density visualizer with energy queries |
| `hydrogenicatom.py` | Hydrogenic atom utilities (standalone reference) |

## Quick Start

```bash
# Run a diagnosis for Neon
python3 diagnosis.py 10

# Run for Iron with no plot
python3 diagnosis.py 26 --no-plot
```

Output:
```
==================================================
  Ne (Z=10)  --  KS-LDA Results
==================================================
  Time:         3.337 s
  N electrons:  10.000000
--------------------------------------------------
  Orbital eigenvalues:
      1s:    -30.301009 Ha   (occ=2)
      2s:     -1.322563 Ha   (occ=2)
      2p:     -0.498057 Ha   (occ=6)
--------------------------------------------------
  E_kinetic:      127.649084 Ha
  E_en (nucl):   -309.956883 Ha
  E_hartree:       65.726428 Ha
  E_xc:           -11.709787 Ha
--------------------------------------------------
  E_total:       -128.291158 Ha
==================================================
```

## Visualization Examples

```bash
# 2D orbital density plots
python3 allplotter.py 26                            # Iron, all orbitals
python3 allplotter.py 10 --orbital 2p               # Neon, only 2p

# 3D isosurface plots
python3 allplotter.py 26 --3d                       # Iron, all orbitals in 3D
python3 allplotter.py 22 --3d --orbital 3d          # Titanium, just the 3d

# Energy queries
python3 allplotter.py 6 --energies                  # Carbon, full energy breakdown
python3 allplotter.py 18 --energy total              # Argon, just E_total
python3 allplotter.py 26 --energy eigenvalues        # Iron, all orbital energies

# Save/load workflow (avoids re-running DFT)
python3 allplotter.py 26 --save Fe_data              # Run once, save results
python3 allplotter.py 26 --load Fe_data.npz          # Instant replot
python3 allplotter.py 26 --load Fe_data.npz --3d     # 3D from saved data

# Use the original (non-optimized) solver
python3 allplotter.py 10 --original

# Higher resolution grid
python3 allplotter.py 10 --grid 5000
```

## Supported Atoms

Hydrogen (Z=1) through Nickel (Z=28). Typical timings on a 3000-point grid:

| Atom | Z | Time |
|------|---|------|
| H    | 1 | 0.05s |
| Ne   | 10 | 4.6s |
| Fe   | 26 | 17s |
| Ni   | 28 | 24s |

## Validation

Total energies compared against the NIST [Atomic Reference Data for Electronic Structure Calculations](https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electro) (LDA, spin-restricted). All values in Hartree.

| Atom | Z  | This code    | NIST LDA     | ΔE (Ha)  |
|------|----|--------------|--------------|----------|
| H    | 1  | -0.445673    | -0.445671    | 2e-6     |
| He   | 2  | -2.834903    | -2.834836    | 7e-5     |
| C    | 6  | -37.432981   | -37.425749   | 7e-3     |
| N    | 7  | -54.038624   | -54.025016   | 1e-2     |
| O    | 8  | -74.496514   | -74.473077   | 2e-2     |
| S    | 16 | -397.093383  | -397.034551  | 6e-2     |

Agreement is at the micro-Hartree level for one-/two-electron atoms; the deviation grows with Z due to the uniform radial grid (NIST uses a logarithmic grid better suited to resolving the core).

## Method

The code solves the radial Kohn-Sham equations:

1. Generate initial orbitals from Slater-type functions with screened energies
2. Build the electron density from occupied orbitals
3. Construct the effective potential: V_eff = V_nuclear + V_Hartree + V_xc
4. Solve for each (n,l) channel using Numerov integration + Brent root finding
5. Iterate until eigenvalues converge (SCF loop with density mixing)

The exchange-correlation functional uses Slater exchange with VWN parametrization for the correlation energy.

## Limitations

- Uniform radial grid limits accuracy for heavy atoms (Z > 28)
- Spin-unpolarized (closed-shell / fractional occupation)
- LDA only (no GGA or hybrid functionals)
