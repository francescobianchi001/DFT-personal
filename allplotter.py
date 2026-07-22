#!/usr/bin/env python3
"""
Plot 2D and 3D orbital densities for any atom Z=1 to 28 (Z=1 to 54 with --exp-grid).

Usage:
    python3 plot_orbitals.py 28        # Nickel, 2D
    python3 plot_orbitals.py 11 --3d   # Sodium, 3D isosurfaces
    python3 plot_orbitals.py 26 --3d --orbital 3d  # Iron, 3D of just 3d
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.interpolate import interp1d
import sys, io, argparse
from scipy.integrate import simpson

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser(description='Atomic orbital visualizer')
parser.add_argument('Z', type=int, help='Atomic number (1-28; up to 54 with --exp-grid)')
parser.add_argument('--3d', dest='three_d', action='store_true', help='3D isosurface plot')
parser.add_argument('--orbital', type=str, default=None, help='Plot specific orbital (e.g. 3d, 2p)')
parser.add_argument('--grid', type=int, default=3000, help='Radial grid points (default 3000)')
parser.add_argument('--energies', action='store_true', help='Print all energy components')
parser.add_argument('--energy', type=str, default=None,
                    choices=['total', 'kinetic', 'hartree', 'xc', 'nuclear', 'eigenvalues'],
                    help='Print a specific energy component')
parser.add_argument('--save', type=str, default=None, metavar='FILE',
                    help='Save WF, eigenvalues, density and grid to .npz file')
parser.add_argument('--load', type=str, default=None, metavar='FILE',
                    help='Load WF data from .npz file instead of running DFT')
parser.add_argument('--original', action='store_true',
                    help='Use the original (non-optimized) solver instead of the fast one')
parser.add_argument('--charge', type=int, default=0,
                    help='Ion charge (0=neutral, 1=+1 cation, ...). Must satisfy 0 <= charge < Z.')
parser.add_argument('--pseudoatom', action='store_true',
                    help='Solve the confined pseudo-atom (Vconf = (r/r0)^2, r0 = 2*r_cov)')
parser.add_argument('--exp-grid', dest='exp_grid', action='store_true',
                    help='Use an exponential radial grid (dense near the nucleus, sparse far '
                         'out): far more accurate per grid point. Optimized solver only.')
parser.add_argument('--rmin', type=float, default=1e-6,
                    help='Inner radius of the radial grid in bohr (default 1e-6; '
                         'try 1e-5 for robust heavy-atom exp-grid runs)')
parser.add_argument('--rmax', type=float, default=15.0,
                    help='Outer radius of the radial grid in bohr (default 15.0)')
parser.add_argument('--save_V', nargs='?', const='__AUTO__', default=None, metavar='FILE',
                    help='Save the converged Veff (includes Vconf) and Vconf. Bare flag '
                         'writes <elem>_potentials.npz; pass a path to choose the file. '
                         'Requires --pseudoatom.')
parser.add_argument('--VO', type=int, default=None, metavar='N',
                    help='Also solve N virtual (unoccupied) shells above the highest '
                         'occupied one. They carry occ=0 in the saved "occupied" list but '
                         'get real eigenvalues/orbitals, to enrich the downstream DFTB basis.')
parser.add_argument('--LB94', action='store_true',
                    help='Add the LB94 gradient correction to Vxc (restores the -1/r tail '
                         'so free-atom virtual orbitals become bound and brackettable).')
parser.add_argument('--r0', dest='r0', type=float, default=None, metavar='R0',
                    help='Override the valence confinement radius (bohr). Default is '
                         '2*r_cov; use this to tune the DFTB off-diagonal (bonding).')
parser.add_argument('--r0-VO', dest='r0_VO', type=float, default=None, metavar='R0',
                    help='Weaker confinement radius (bohr) applied only to the virtual '
                         '(occ=0) shells from --VO; occupied shells keep r0=2*r_cov. Runs '
                         'a second confined solve and combines the two: valence WFs from '
                         'r0, virtual WFs from r0_VO. Both Vconf are saved with --save_V.')
parser.add_argument('--VO-keep', dest='VO_keep', type=str, default=None, metavar='LS',
                    help="Keep only the virtual (--VO) shells with these l when saving: "
                         "l-letters ('d', 'sd') or numbers ('2', '0,2'); the rest are "
                         "zeroed. Use to keep only virtuals with no valence partner "
                         "(e.g. d), avoiding on-site non-orthogonality in DFTB.")

args = parser.parse_args()

Z = args.Z
zmax = 54 if args.exp_grid else 28
assert 1 <= Z <= zmax, f"Z must be between 1 and {zmax}" + ("" if args.exp_grid else " (use --exp-grid for heavier atoms)")
charge = args.charge
assert 0 <= charge < Z, f"charge must satisfy 0 <= charge < Z={Z}"
if args.exp_grid and args.original:
    parser.error("--exp-grid is only supported by the optimized solver (not --original)")
if args.save_V and not args.pseudoatom:
    parser.error("--save_V requires --pseudoatom (Vconf only exists for a confined atom)")

element_names = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
    11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',
    19:'K', 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni',
    29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr',
    37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru',
    45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I', 54:'Xe'
}
elem = element_names.get(Z, f"Z={Z}")

# ============================================================
# Run DFT or load from file
# ============================================================
if args.original:
    from atomDFT_original import AtomicDFT
    print("Using original (non-optimized) solver")
else:
    from atomDFT import AtomicDFT

l_names = 'spdfghi'

if args.load:
    print(f"Loading data from {args.load}...")
    data = np.load(args.load, allow_pickle=True)
    r = data['grid']
    rho = data['rho']
    evals = data['eigenvalues'].tolist()
    occupied = data['occupied'].tolist()
    WF_final = data['wavefunctions'].tolist()
    saved_charge = int(data['charge']) if 'charge' in data.files else 0
    # Reconstruct dft object for energy calculations
    dft = AtomicDFT(r, Z, charge=saved_charge)
    dft.CONTROLL = False
    dft.GetOrbitals()
    dft.occupied = occupied
else:
    label = elem if charge == 0 else f"{elem}{charge:+d}"
    print(f"Running KS-LDA for {label} (Z={Z}, charge={charge})...")
    r = np.linspace(args.rmin, args.rmax, args.grid)
    dft = AtomicDFT(r, Z, charge=charge, exp_grid=args.exp_grid,
                    rmin=args.rmin, rmax=args.rmax, VO=args.VO, LB94=args.LB94)
    r = dft.radial_grid   # canonical grid the WFs live on (exponential if requested)
    dft.CONTROLL = False
    WF = dft.GetOrbitals()

    old_stdout = sys.stdout; sys.stdout = io.StringIO()
    evals, WF_final, rho = dft.GetKohnShamEquation(WF)
    sys.stdout = old_stdout

# Covalent radii (Cordero 2008), in bohr (Å × 1.8897)
R_COV = {
    1: 0.586,  2: 0.529,  3: 2.419,  4: 1.814,  5: 1.588,  6: 1.436,
    7: 1.342,  8: 1.247,  9: 1.077, 10: 1.096, 11: 3.137, 12: 2.665,
    13: 2.287, 14: 2.098, 15: 2.022, 16: 1.984, 17: 1.928, 18: 2.003,
    19: 3.836, 20: 3.326, 21: 3.213, 22: 3.024, 23: 2.891, 24: 2.627,
    25: 2.627, 26: 2.494, 27: 2.381, 28: 2.343,
    29: 2.494, 30: 2.305, 31: 2.305, 32: 2.268, 33: 2.249, 34: 2.268,
    35: 2.268, 36: 2.192, 37: 4.157, 38: 3.685, 39: 3.590, 40: 3.307,
    41: 3.099, 42: 2.910, 43: 2.778, 44: 2.759, 45: 2.683, 46: 2.627,
    47: 2.740, 48: 2.721, 49: 2.683, 50: 2.627, 51: 2.627, 52: 2.608,
    53: 2.627, 54: 2.646,
}

if args.pseudoatom:
    if args.r0 is not None:
        r0 = args.r0                       # explicit override (r_c tuning)
    else:
        if Z not in R_COV:
            parser.error(f"--pseudoatom needs a covalent radius for Z={Z}; R_COV only covers Z<=28")
        r0 = 2 * R_COV[Z]
    split = args.r0_VO is not None
    if split and not args.VO:
        parser.error("--r0-VO requires --VO N (no virtual shells to re-confine otherwise)")
    print(f"Solving confined pseudo-atom: r0 = 2*r_cov = {r0:.3f} bohr"
          + (f"; virtuals re-confined at r0_VO = {args.r0_VO:.3f} bohr" if split else ""))

    def _confined(r0_use, need_V):
        d = AtomicDFT(r, Z, charge=charge, r0=r0_use, exp_grid=args.exp_grid,
                      rmin=args.rmin, rmax=args.rmax, save_V=need_V, VO=args.VO,
                      LB94=args.LB94)
        d.CONTROLL = False
        wf = d.GetOrbitals()
        so = sys.stdout; sys.stdout = io.StringIO()
        res = d.GetKohnShamEquation(wf)
        sys.stdout = so
        return d, res

    need_V = bool(args.save_V) or split
    dft, res = _confined(r0, need_V)
    r = dft.radial_grid   # canonical grid the WFs live on (exponential if requested)
    if need_V:
        evals, WF_final, rho, Veff, Vconf = res
    else:
        evals, WF_final, rho = res

    Vconf_VO = None
    vo_shells = []
    if split:
        # Second solve at the weaker wall; swap its orbitals into the virtual
        # (occ=0) slots. Occupied shells, rho and Veff stay from the r0 solve.
        # vo_shells records which (n,l) got the VO wall, for the DFTB off-diagonal.
        _, res_vo = _confined(args.r0_VO, True)
        evals_vo, WF_vo, _, _, Vconf_VO = res_vo
        for ni in range(len(dft.occupied)):
            for l in range(len(dft.occupied[ni])):
                w = WF_vo[ni][l] if ni < len(WF_vo) and l < len(WF_vo[ni]) else None
                if dft.occupied[ni][l] == 0 and isinstance(w, np.ndarray) and np.any(w != 0):
                    WF_final[ni][l] = w
                    evals[ni][l] = evals_vo[ni][l]
                    vo_shells.append([ni, l])
        print(f"  combined: occupied shells from r0={r0:.3f}, "
              f"virtual shells from r0_VO={args.r0_VO:.3f}")

    if args.save_V:
        # Converged Veff (includes Vconf) and the confinement term(s) alone, on
        # the canonical radial grid, for the downstream DFTB off-diagonal.
        vfname = f'{elem}_potentials' if args.save_V == '__AUTO__' else args.save_V
        vfname = vfname if vfname.endswith('.npz') else vfname + '.npz'
        Vdict = dict(grid=r, Veff=Veff, Vconf=Vconf, Z=Z, r0=r0, charge=dft.charge)
        if split:
            Vdict.update(Vconf_VO=Vconf_VO, r0_VO=args.r0_VO,
                         vo_shells=np.array(vo_shells, dtype=int).reshape(-1, 2))
        np.savez(vfname, **Vdict)
        print(f"Saved Veff and Vconf to {vfname}" + (" (+ Vconf_VO)" if split else ""))

if args.VO is not None and args.VO_keep is not None:
    # Keep only selected-l virtual shells; zero the rest so the downstream
    # basis skips them (occupied shells are never touched).
    _lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    keep_l = set()
    for tok in args.VO_keep.replace(',', ' ').split():
        tok = tok.strip().lower()
        if tok.isdigit():
            keep_l.add(int(tok))
        elif tok and all(c in _lmap for c in tok):
            keep_l.update(_lmap[c] for c in tok)
        else:
            parser.error(f"--VO-keep: cannot parse '{tok}' (use l-letters spdf or numbers)")
    dropped = []
    for ni in range(len(dft.occupied)):
        for l in range(len(dft.occupied[ni])):
            if dft.occupied[ni][l] == 0 and l not in keep_l:
                w = WF_final[ni][l]
                if isinstance(w, np.ndarray) and np.any(w != 0):
                    WF_final[ni][l] = np.zeros_like(r)
                    evals[ni][l] = 0.0
                    dropped.append(f"{ni + 1}{l_names[l]}")
    if dropped:
        print(f"VO-keep: zeroed virtual shells {', '.join(dropped)} (kept l={sorted(keep_l)})")

if args.save:
    fname = args.save if args.save.endswith('.npz') else args.save + '.npz'
    save_kw = dict(grid=r, rho=rho,
                   eigenvalues=np.array(evals, dtype=object),
                   occupied=np.array(dft.occupied, dtype=object),
                   wavefunctions=np.array(WF_final, dtype=object),
                   Z=Z, charge=dft.charge)
    if not args.pseudoatom:
        # plain-LDA neutral effective potential (no LB94, no Vconf) = Vneutral for
        # the DFTB density/wavefunction split. LB94 stays a diagonal-only device.
        save_kw['Veff'] = (dft.getVXC(rho) + dft.getHartreePotential(rho)
                           + dft.getNuclearPotential())
    np.savez(fname, **save_kw)
    print(f"Saved data to {fname}")

# Build orbital dictionary
wf_dict = {}
eval_dict = {}
occ_dict = {}
for i, shell in enumerate(WF_final):
    for j, wf in enumerate(shell):
        n = i + 1
        l = j
        label = f"{n}{l_names[l]}"
        if isinstance(wf, np.ndarray) and np.any(wf != 0):
            wf_dict[label] = (n, l, wf)
            eval_dict[label] = evals[i][j]
            occ_dict[label] = dft.occupied[n - 1][l]

print(f"Converged orbitals for {elem}:")
for label in wf_dict:
    n, l, _ = wf_dict[label]
    occ = dft.occupied[n - 1][l]
    tag = '' if occ > 0 else '   (virtual, occ=0)'
    print(f"  {label}: {eval_dict[label]:.6f} Ha{tag}")

# ============================================================
# Energy output
# ============================================================
if args.energies or args.energy:
    from scipy.integrate import simpson
    Ekin = dft.getEkin(WF_final)
    Een = dft.getEenuc(rho)
    Eh = dft.getH_Energy(rho)
    Exc = dft.getXC_Energy(rho)
    E_tot = dft.getE_tot(rho, WF_final)

    energies = {
        'kinetic':     ('E_kinetic',  Ekin),
        'nuclear':     ('E_nuclear',  Een),
        'hartree':     ('E_hartree',  Eh),
        'xc':          ('E_xc',       Exc),
        'total':       ('E_total',    E_tot),
    }

    if args.energy == 'eigenvalues':
        print(f"\nEigenvalues for {elem} (Z={Z}):")
        for label in wf_dict:
            i = int(label[0]) - 1
            j = l_names.index(label[1])
            occ = dft.occupied[i][j]
            print(f"  {label}: {eval_dict[label]:12.6f} Ha  (occ={occ})")
    elif args.energy:
        name, val = energies[args.energy]
        print(f"\n{name} = {val:.6f} Ha")
    else:
        print(f"\n{'='*40}")
        print(f"  Energy components for {elem} (Z={Z})")
        print(f"{'='*40}")
        for key, (name, val) in energies.items():
            print(f"  {name:12s}  {val:12.6f} Ha")
        print(f"{'='*40}")
        print(f"\n  Eigenvalues:")
        for label in wf_dict:
            i = int(label[0]) - 1
            j = l_names.index(label[1])
            occ = dft.occupied[i][j]
            print(f"    {label}: {eval_dict[label]:12.6f} Ha  (occ={occ})")

# Filter if specific orbital requested
if args.orbital:
    if args.orbital in wf_dict:
        wf_dict = {args.orbital: wf_dict[args.orbital]}
        eval_dict = {args.orbital: eval_dict[args.orbital]}
        occ_dict = {args.orbital: occ_dict[args.orbital]}
    else:
        print(f"Orbital '{args.orbital}' not found. Available: {list(wf_dict.keys())}")
        sys.exit(1)
# ============================================================
# Spherical harmonics
# ============================================================
def real_Ylm(l, m, theta, phi):
    """Real spherical harmonics."""
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    if l == 0:
        return 0.5 * np.sqrt(1.0 / np.pi) * np.ones_like(theta)
    elif l == 1:
        if m == 0:  return 0.5 * np.sqrt(3.0/np.pi) * ct
        if m == 1:  return -0.5 * np.sqrt(3.0/np.pi) * st * cp
        if m == -1: return -0.5 * np.sqrt(3.0/np.pi) * st * sp
    elif l == 2:
        if m == 0:  return 0.25 * np.sqrt(5.0/np.pi) * (3*ct**2 - 1)
        if m == 1:  return -0.5 * np.sqrt(15.0/np.pi) * st * ct * cp
        if m == -1: return -0.5 * np.sqrt(15.0/np.pi) * st * ct * sp
        if m == 2:  return 0.25 * np.sqrt(15.0/np.pi) * st**2 * np.cos(2*phi)
        if m == -2: return 0.25 * np.sqrt(15.0/np.pi) * st**2 * np.sin(2*phi)
    return np.ones_like(theta) * 0.28

# m value to show for each l (most visually interesting in xz plane)
m_for_l = {0: 0, 1: 0, 2: 0, 3: 0}

# ============================================================
# 2D plotting
# ============================================================
def get_rmax(u_r):
    threshold = 0.01 * np.max(np.abs(u_r))
    significant = np.where(np.abs(u_r) > threshold)[0]
    if len(significant) > 0:
        rmax = r[significant[-1]] * 1.3
    else:
        rmax = r[-1]
    return min(max(rmax, 0.5), 12.0)

def plot_2d(wf_dict, eval_dict):
    orbitals = list(wf_dict.keys())
    n_orbs = len(orbitals)
    ncols = min(4, n_orbs)
    nrows = (n_orbs + ncols - 1) // ncols
    grid_size = 300

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), facecolor='black')
    if n_orbs == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, label in enumerate(orbitals):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        n, l, u_r = wf_dict[label]
        m = m_for_l[l]
        rmax = get_rmax(u_r)

        x = np.linspace(-rmax, rmax, grid_size)
        z = np.linspace(-rmax, rmax, grid_size)
        X, Z_grid = np.meshgrid(x, z)
        R = np.sqrt(X**2 + Z_grid**2)
        THETA = np.arctan2(np.abs(X), Z_grid)
        PHI = np.where(X >= 0, 0.0, np.pi)

        u_interp = interp1d(r, u_r, kind='cubic', bounds_error=False, fill_value=0.0)
        U = u_interp(R)
        R_safe = np.where(R > 1e-10, R, 1e-10)
        Y = real_Ylm(l, m, THETA, PHI)
        PSI = U / R_safe * Y
        density = PSI**2

        vmax = np.max(density)
        if vmax > 0:
            ax.pcolormesh(X, Z_grid, density, cmap='inferno',
                         norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax), shading='auto')
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        e_str = f"{eval_dict[label]:.4f}" if label in eval_dict else ""
        vtag = "  [virtual]" if occ_dict.get(label, 1) == 0 else ""
        ax.set_title(f"{label}  ({e_str} Ha){vtag}", color='#88aaff', fontsize=13, fontweight='bold')
        ax.set_xlabel('x (bohr)', color='#555', fontsize=8)
        ax.set_ylabel('z (bohr)', color='#555', fontsize=8)
        ax.tick_params(colors='#444', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#222')

    for idx in range(n_orbs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f'{elem} (Z={Z}) — KS-LDA Orbital Densities |ψ(x,z)|²',
                 color='#88aaff', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'{elem}_orbitals_2d.png'
    #plt.savefig(fname, dpi=150, facecolor='black', bbox_inches='tight')
    #print(f"Saved {fname}")
    plt.show()

# ============================================================
# 3D plotting
# ============================================================
def plot_3d(wf_dict, eval_dict):
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure

    orbitals = list(wf_dict.keys())
    n_orbs = len(orbitals)
    ncols = min(4, n_orbs)
    nrows = (n_orbs + ncols - 1) // ncols

    fig = plt.figure(figsize=(5*ncols, 5*nrows), facecolor='#0a0a14')

    for idx, label in enumerate(orbitals):
        n, l, u_r = wf_dict[label]
        m = m_for_l[l]
        rmax = get_rmax(u_r)
        grid_size = 60  # 3D needs fewer points

        coords = np.linspace(-rmax, rmax, grid_size)
        X3, Y3, Z3 = np.meshgrid(coords, coords, coords, indexing='ij')
        R3 = np.sqrt(X3**2 + Y3**2 + Z3**2)
        THETA3 = np.arccos(np.clip(Z3 / np.maximum(R3, 1e-10), -1, 1))
        PHI3 = np.arctan2(Y3, X3)

        u_interp = interp1d(r, u_r, kind='cubic', bounds_error=False, fill_value=0.0)
        U3 = u_interp(R3)
        R3_safe = np.where(R3 > 1e-10, R3, 1e-10)
        YLM = real_Ylm(l, m, THETA3, PHI3)
        PSI3 = U3 / R3_safe * YLM
        density3 = PSI3**2

        # Isosurface at some fraction of max
        dmax = np.max(density3)
        if dmax == 0:
            continue
        iso_level = 0.05 * dmax

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d',
                            facecolor='#0a0a14')

        try:
            verts, faces, _, _ = measure.marching_cubes(density3, level=iso_level)
            # Scale vertices to real coordinates
            verts = verts / grid_size * 2 * rmax - rmax

            # Color by z-coordinate for visual depth
            face_z = np.mean(verts[faces], axis=1)[:, 2]
            face_z_norm = (face_z - face_z.min()) / (face_z.max() - face_z.min() + 1e-10)

            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           cmap='coolwarm', alpha=0.7, edgecolor='none',
                           antialiased=True)
        except Exception:
            # If marching cubes fails, show a scatter plot instead
            threshold = 0.1 * dmax
            mask = density3 > threshold
            pts = np.column_stack([X3[mask], Y3[mask], Z3[mask]])
            vals = density3[mask]
            if len(pts) > 2000:
                idx_sample = np.random.choice(len(pts), 2000, replace=False)
                pts = pts[idx_sample]
                vals = vals[idx_sample]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=vals, cmap='inferno',
                      s=1, alpha=0.3)

        e_str = f"{eval_dict[label]:.4f}" if label in eval_dict else ""
        vtag = " [virtual]" if occ_dict.get(label, 1) == 0 else ""
        ax.set_title(f"{label} ({e_str} Ha){vtag}", color='#88aaff', fontsize=12, pad=0)
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)
        ax.set_xlabel('x', color='#444', fontsize=7)
        ax.set_ylabel('y', color='#444', fontsize=7)
        ax.set_zlabel('z', color='#444', fontsize=7)
        ax.tick_params(labelsize=6, colors='#444')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#222')
        ax.yaxis.pane.set_edgecolor('#222')
        ax.zaxis.pane.set_edgecolor('#222')

    fig.suptitle(f'{elem} (Z={Z}) — KS-LDA 3D Isosurfaces |ψ|²',
                 color='#88aaff', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'{elem}_orbitals_3d.png'
    #plt.savefig(fname, dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    #print(f"Saved {fname}")
    plt.show()

# ============================================================
# Run
# ============================================================
if args.three_d:
    plot_3d(wf_dict, eval_dict)
else:
    plot_2d(wf_dict, eval_dict)
