#!/usr/bin/env python3
"""
Diagnosis script for AtomicDFT: runs KS-LDA, prints all energy components,
and plots radial wavefunctions.

Usage:
    python3 diagnosis.py 10        # Neon
    python3 diagnosis.py 26        # Iron
    python3 diagnosis.py 1 --no-plot
"""

import numpy as np
import matplotlib.pyplot as plt
from atomDFT import AtomicDFT
from scipy.integrate import simpson
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser(description='AtomicDFT diagnosis')
parser.add_argument('Z', type=int, help='Atomic number (1-26)')
parser.add_argument('--grid', type=int, default=3000, help='Radial grid points (default 3000)')
parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
args = parser.parse_args()

Z = args.Z

element_names = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
    11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',
    19:'K', 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe'
}
elem = element_names.get(Z, f"Z={Z}")
l_names = 'spdf'

# ============================================================
# Run DFT
# ============================================================
print(f"Running KS-LDA for {elem} (Z={Z})...")
r = np.linspace(1e-6, 15.0, args.grid)
atom = AtomicDFT(r, Z)
atom.CONTROLL = False
WF = atom.GetOrbitals()

start = time.time()
evals, WF_f, rho = atom.GetKohnShamEquation(WF)
elapsed = time.time() - start

# ============================================================
# Print energies
# ============================================================
N_elec = simpson(rho * 4 * np.pi * r**2, x=r)
Exc = atom.getXC_Energy(rho)
Eh = atom.getH_Energy(rho)
Een = atom.getEenuc(rho)
Ekin = atom.getEkin(WF_f)
E_tot = atom.getE_tot(rho, WF_f)

print(f"\n{'='*50}")
print(f"  {elem} (Z={Z})  —  KS-LDA Results")
print(f"{'='*50}")
print(f"  Time:         {elapsed:.3f} s")
print(f"  N electrons:  {N_elec:.6f}")
print(f"{'—'*50}")
print(f"  Orbital eigenvalues:")
for i, shell in enumerate(evals):
    for j, e in enumerate(shell):
        occ = atom.occupied[i][j]
        label = f"  {i+1}{l_names[j]}"
        print(f"    {label}:  {e:12.6f} Ha   (occ={occ})")
print(f"{'—'*50}")
print(f"  E_kinetic:    {Ekin:12.6f} Ha")
print(f"  E_en (nucl):  {Een:12.6f} Ha")
print(f"  E_hartree:    {Eh:12.6f} Ha")
print(f"  E_xc:         {Exc:12.6f} Ha")
print(f"{'—'*50}")
print(f"  E_total:      {E_tot:12.6f} Ha")
print(f"{'='*50}")

# ============================================================
# Plot radial wavefunctions
# ============================================================
if not args.no_plot:
    orbitals = []
    for i, shell in enumerate(WF_f):
        for j, wf in enumerate(shell):
            if isinstance(wf, np.ndarray) and np.any(wf != 0):
                label = f"{i+1}{l_names[j]}"
                orbitals.append((label, wf, evals[i][j]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Plot wavefunctions u(r)
    for label, wf, e in orbitals:
        ax1.plot(r, wf, label=f"{label} ({e:.4f} Ha)")
    ax1.set_ylabel('u(r)')
    ax1.set_title(f'{elem} (Z={Z}) — Radial Wavefunctions')
    ax1.legend(fontsize=8)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set_xlim(0, 8)

    # Plot density
    ax2.plot(r, rho * 4 * np.pi * r**2, 'k-', lw=1.5, label='4πr²ρ(r)')
    ax2.set_xlabel('r (bohr)')
    ax2.set_ylabel('4πr²ρ(r)')
    ax2.set_title('Radial Electron Density')
    ax2.legend()
    ax2.set_xlim(0, 8)

    plt.tight_layout()
    fname = f'{elem}_diagnosis.png'
    #plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved {fname}")
    plt.show()
