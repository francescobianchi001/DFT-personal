#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import splrep, splev
from numba import njit


@njit
def _numerov_propagate(g, radial_grid, e, l):
    """Numba-compiled Numerov propagation (forward + backward + matching)."""
    N = len(radial_grid)
    h = radial_grid[1] - radial_grid[0]
    c1 = 5.0 * h**2 / 12.0
    c2 = h**2 / 12.0
    sqrt_neg2e = np.sqrt(-2.0 * e)
    y_right = np.zeros(N)
    y_right[N-1] = np.exp(-sqrt_neg2e * radial_grid[N-1])
    y_right[N-2] = np.exp(-sqrt_neg2e * radial_grid[N-2])
    y_left = np.zeros(N)
    y_left[0] = radial_grid[0]**(l + 1)
    y_left[1] = radial_grid[1]**(l + 1)
    # Find turning point
    tp = 0
    last_pos = -1
    for i in range(N):
        if g[i] > 0:
            last_pos = i
    if last_pos >= 0:
        tp = min(last_pos + 1, N - 3)
    else:
        for i in range(N):
            if g[i] <= 0:
                tp = i
                break
    # Forward propagation
    for n in range(2, tp + 2):
        den = 1.0 / (1.0 + c2 * g[n])
        num = 2.0 * y_left[n-1] * (1.0 - c1 * g[n-1]) - y_left[n-2] * (1.0 + c2 * g[n-2])
        y_left[n] = num * den
    # Backward propagation
    for n in range(N - 3, tp - 2, -1):
        den = 1.0 / (1.0 + c2 * g[n])
        num = 2.0 * y_right[n+1] * (1.0 - c1 * g[n+1]) - y_right[n+2] * (1.0 + c2 * g[n+2])
        y_right[n] = num * den
    # Match at turning point
    if y_left[tp] * y_right[tp] < 0:
        for i in range(N):
            y_right[i] = -y_right[i]
    scale = y_right[tp] / y_left[tp]
    for i in range(N):
        y_left[i] = y_left[i] * scale
    # Build full wavefunction
    Y = np.empty(N)
    for i in range(tp):
        Y[i] = y_left[i]
    for i in range(tp, N):
        Y[i] = y_right[i]
    dL = y_left[tp+1] - y_left[tp-1]
    dR = y_right[tp+1] - y_right[tp-1]
    return Y, dL, dR, tp

class AtomicDFT:

    MAX_Z = 26  # Fe; uniform grid cannot resolve deeper core states

    def __init__(self, grid, Z):
        if Z > self.MAX_Z:
            raise ValueError(
                f"Z={Z} exceeds MAX_Z={self.MAX_Z}. "
                f"Uniform grid cannot resolve core orbitals for Z>{self.MAX_Z}."
            )
        self.Z = Z
        self.radial_grid = grid
        self.CONTROLL = False

    def _get_screened_energy(self, n, l):
        """Estimate starting eigenvalue using Slater screening rules."""
        Z = self.Z
        aufbau_fill = [(1,0,2),(2,0,2),(2,1,6),(3,0,2),(3,1,6),(4,0,2),(3,2,10),
                        (4,1,6),(5,0,2),(4,2,10),(5,1,6),(6,0,2),(4,3,14),(5,2,10)]
        filled = {}
        rem = Z
        for (gn, gl, gmax) in aufbau_fill:
            if rem <= 0: break
            filled[(gn,gl)] = min(rem, gmax)
            rem -= filled[(gn,gl)]
        sigma = 0.0
        for (gn, gl), occ in filled.items():
            if gn == n and gl == l:
                sigma += (0.30 if n == 1 else 0.35) * (occ - 1)
            elif gn == n and n >= 2 and l <= 1 and gl <= 1:
                sigma += 0.35 * occ
            elif gn == n:
                sigma += 0.35 * occ
            elif gn == n - 1:
                sigma += (0.85 if l <= 1 else 1.00) * occ
            elif gn < n - 1:
                sigma += 1.00 * occ
        Z_eff = max(Z - sigma, 1.0)
        return -Z_eff**2 / (2.0 * n**2)

    @staticmethod
    def _slater_radial(n, l, r, E):
        """Hydrogenic Slater-type radial function u(r) = r^(l+1) * exp(-alpha*r)."""
        alpha = np.sqrt(-2.0 * E)
        return r**(l + 1) * np.exp(-alpha * r)

    def GetOrbitals(self, exp_grid=False):
        if exp_grid:
            x = np.linspace(np.log(1e-6), np.log(15.0), 2000)
            r = np.exp(x)
        else:
            r = self.radial_grid

        # Aufbau filling order: (n, l) pairs
        aufbau = [
            (1,0),              # 1s
            (2,0), (2,1),       # 2s, 2p
            (3,0), (3,1),       # 3s, 3p
            (4,0),              # 4s  
            (3,2),              # 3d
            (4,1),              # 4p
            (5,0),              # 5s  
            (4,2),              # 4d
            (5,1),              # 5p
            (6,0),              # 6s
            (4,3),              # 4f
            (5,2),              # 5d
            (6,1),              # 6p
            (7,0),              # 7s
            (5,3),              # 5f
            (6,2),              # 6d
            (7,1),              # 7p
        ]

        N = self.Z
        # Figure out which shells are needed
        max_n = 0
        remaining = N
        for (n, l) in aufbau:
            if remaining <= 0:
                break
            occ_max = 2 * (2 * l + 1)
            remaining -= min(remaining, occ_max)
            max_n = max(max_n, n)
        self.nshells = max_n

        # Initialize empty lists for each shell
        self.occupied = [[] for _ in range(self.nshells)]
        self.E_start = [[] for _ in range(self.nshells)]
        self.radial_WF = [[] for _ in range(self.nshells)]

        # Track which l values have been added to each shell
        shell_l_added = [[] for _ in range(self.nshells)]

        # Fill in Aufbau order
        remaining = N
        for (n, l) in aufbau:
            if remaining <= 0:
                break
            occ_max = 2 * (2 * l + 1)
            occ = min(remaining, occ_max)

            # Build the orbital
            E = self._get_screened_energy(n, l)
            radial = self._slater_radial(n, l, r, E)
            if exp_grid:
                radial_spline = splrep(r, radial)
                radial = splev(self.radial_grid, radial_spline)
            norm = simpson(radial**2, x=self.radial_grid)
            radial = radial / np.sqrt(norm)

            idx = n - 1  
            insert_pos = 0
            for existing_l in shell_l_added[idx]:
                if existing_l < l:
                    insert_pos += 1
                else:
                    break
            self.occupied[idx].insert(insert_pos, occ)
            self.E_start[idx].insert(insert_pos, E)
            self.radial_WF[idx].insert(insert_pos, radial)
            shell_l_added[idx].insert(insert_pos, l)

            remaining -= occ

        return self.radial_WF
    
    def getRadialDensity(self, WF):
        rho = [[self.occupied[i][j] * (WF[i][j])**2 for j in range(len(WF[i]))] for i in range(len(WF))]
        rho = 1 / (4 * np.pi * self.radial_grid**2) * np.sum([orb for shell in rho for orb in shell], axis=0)
        return rho

    def getRadialDensity_sorbs(self, WF_s):
        rho = np.zeros_like(self.radial_grid)
        for i, wf in enumerate(WF_s):
            occ = self.occupied[i][0]
            rho += occ * wf**2
        rho = rho / (4 * np.pi * self.radial_grid**2)
        return rho

    def getVXC(self, rho):
        rho = np.maximum(rho, 1e-14)
        rs = (3.0 / (4.0 * np.pi * rho))**(1.0 / 3.0)
        A = 0.0621814; y0 = -0.10498; b = 3.72744; c = 12.93532
        Cx = 0.5 * 3.0 * (9.0 / (32.0 * np.pi**2))**(1 / 3)
        y = np.sqrt(rs); Y = y**2 + b * y + c; Q = np.sqrt(4.0 * c - b**2)
        t1 = np.log(y**2 / Y)
        t2 = (2.0 * b / Q) * np.arctan(Q / (2.0 * y + b))
        t3 = b * y0 / (y0**2 + b * y0 + c)
        t4 = np.log((y - y0)**2 / Y)
        t5 = 2.0 * (b + 2 * y0) / Q
        t6 = np.arctan(Q / (2.0 * y + b))
        ec = 0.5 * A * (t1 + t2 - t3 * (t4 + t5 * t6))
        dec = 0.5 * A * (c * (y - y0) - b * y * y0) / ((y - y0) * Y)
        Vx = -(4.0 / 3.0) * Cx / rs; Vc = ec - (1.0 / 3.0) * dec
        Vxc = Vx + Vc; self.Vxc = Vxc
        return Vxc

    def getHartreePotential(self, rho):
        r = self.radial_grid
        inner = cumulative_trapezoid(rho * r**2, r, initial=0)
        outer_cumul = cumulative_trapezoid(rho * r, r, initial=0)
        outer = outer_cumul[-1] - outer_cumul
        Vh = 4 * np.pi * (inner / r + outer)
        Vh[0] = Vh[1]
        return Vh

    def getNuclearPotential(self):
        Ven = -self.Z / self.radial_grid; self.Ven = Ven
        return Ven

    def getEffectivePotential(self, rho):
        Vxc = self.getVXC(rho); Vh = self.getHartreePotential(rho); Ven = self.getNuclearPotential()
        return Vxc + Vh + Ven

    def getAngularPotential(self, l):
        return l * (l + 1) / (2 * self.radial_grid**2)

    def getNOdes(self, WF):
        return np.sum(WF[:-1] * WF[1:] <= 0)

    def getEigenValue(self, bracket_Eigenvalue, Veff, l):
        def root_f(E):
            _, F = self.getWaveFunction_numerov(E, l, Veff)
            return F
        eigenvalue = brentq(root_f, bracket_Eigenvalue[0], bracket_Eigenvalue[1])
        WF, _ = self.getWaveFunction_numerov(eigenvalue, l, Veff)
        return eigenvalue, WF

    def getBracketEnergy(self, E_start, Veff, n, l):
        delta = max(0.001 * abs(E_start), 0.01)
        target_nodes = n - l - 1
        e1 = E_start
        WF, F1 = self.getWaveFunction_numerov(e1, l, Veff)
        for _ in range(5000):
            e2 = e1 + delta
            if e2 >= 0: break
            WF, F2 = self.getWaveFunction_numerov(e2, l, Veff)
            if np.isnan(F2): break
            nodes = self.getNOdes(WF)
            if F2 * F1 < 0 and nodes == target_nodes: return (e1, e2)
            e1, F1 = e2, F2
        e1 = E_start
        WF, F1 = self.getWaveFunction_numerov(e1, l, Veff)
        for _ in range(5000):
            e2 = e1 - delta
            WF, F2 = self.getWaveFunction_numerov(e2, l, Veff)
            if np.isnan(F2): break
            nodes = self.getNOdes(WF)
            if F2 * F1 < 0 and nodes == target_nodes: return (e2, e1)
            e1, F1 = e2, F2
        return None

    def getWaveFunction_numerov(self, e, l, Veff):
        h = self.radial_grid[1] - self.radial_grid[0]
        g = 2.0 * (e - Veff)
        Y, dL, dR, tp = _numerov_propagate(g, self.radial_grid, e, l)
        norm = simpson(Y**2, x=self.radial_grid)
        y = Y / np.sqrt(norm)
        F = 1.0 / np.sqrt(norm) * (dL - dR) / (2.0 * h)
        if self.CONTROLL:
            print(f"DEBUG: tp={tp}, E_{l}={e}, F={F}")
        self.tp = tp
        return y, F

    def runSCF_sorbs(self, WF=None, max_iter=20, treshold=1e-6):
        Veff = -self.Z / self.radial_grid
        eigenvalues_old_s = [self.E_start[n][0] for n in range(self.nshells)]
        WF_old_s = [self.radial_WF[n][0] for n in range(self.nshells)]
        damp = 0.3
        for iteration in range(max_iter):
            Veff_old = Veff.copy()
            eigenvalues_next_s = []
            WF_next_s = []
            for n in range(1, self.nshells + 1):
                b = self.getBracketEnergy(eigenvalues_old_s[n-1], Veff, n, 0)
                if b is None:
                    # Reuse old value
                    eigenvalues_next_s.append(eigenvalues_old_s[n-1])
                    WF_next_s.append(WF_old_s[n-1])
                else:
                    E_f, WF_f = self.getEigenValue(b, Veff, 0)
                    eigenvalues_next_s.append(E_f)
                    WF_next_s.append(WF_f)
            eigenvalues_next = [[e] for e in eigenvalues_next_s]
            flat_next = np.array(eigenvalues_next_s)
            flat_old = np.array(eigenvalues_old_s)
            delta = np.max(np.abs(flat_next - flat_old))
            eigenvalues_old_s = list(eigenvalues_next_s)
            WF_old_s = list(WF_next_s)
            rho = self.getRadialDensity_sorbs(WF_next_s)
            Veff_new = self.getEffectivePotential(rho)
            Veff = (1 - damp) * Veff_old + damp * Veff_new
            if delta <= treshold: break
        return eigenvalues_next, tuple(WF_next_s)

    def run_SCF(self, eigenvalues, WF, Veff, max_iter=200, treshold=1e-6):
        eigenvalues_old = [[e for e in shell] for shell in eigenvalues]
        Rho_old = self.getRadialDensity(WF)
        Veff_mixed = Veff.copy()
        for iteration in range(max_iter):
            # Adaptive damping: start gentle, increase as we converge
            damp = min(0.1 + 0.01 * iteration, 0.3)
            Rho_new = self.getRadialDensity(WF)
            Rho_mixed = (1 - damp) * Rho_old + damp * Rho_new
            Veff_new = self.getEffectivePotential(Rho_mixed)
            Veff_mixed = (1 - damp) * Veff_mixed + damp * Veff_new
            eigenvalues_next = []; WF_next = []
            for nshell in range(1, self.nshells + 1):
                eigenvalues_next.append([]); WF_next.append([])
                for lshell in range(len(self.occupied[nshell-1])):
                    if self.occupied[nshell-1][lshell] == 0:
                        eigenvalues_next[nshell-1].append(0)
                        WF_next[nshell-1].append(np.zeros_like(self.radial_grid))
                        continue
                    Vr = self.Vr[nshell - 1][lshell]
                    Veff_next = Veff_mixed + Vr
                    E_start = eigenvalues_old[nshell - 1][lshell]
                    Bracket = self.getBracketEnergy(E_start, Veff_next, nshell, lshell)
                    if Bracket is None:
                        if self.CONTROLL:
                            print(f"WARNING: bracket failed n={nshell},l={lshell},E={E_start:.4f}, reusing old")
                        eigenvalues_next[nshell-1].append(E_start)
                        WF_next[nshell-1].append(WF[nshell-1][lshell])
                        continue
                    E_f, WF_f = self.getEigenValue(Bracket, Veff_next, lshell)
                    WF_next[nshell - 1].append(WF_f)
                    eigenvalues_next[nshell - 1].append(E_f)
            self.eigenvalues = eigenvalues_next
            flat_next = np.array([e for shell in eigenvalues_next for e in shell])
            flat_old = np.array([e for shell in eigenvalues_old for e in shell])
            delta = np.max(np.abs(flat_next - flat_old))
            if self.CONTROLL:
                print(f"SCF iter {iteration}: damp={damp:.2f} delta={delta:.2e}, eigenvalues={eigenvalues_next}")
            eigenvalues_old = [[e for e in shell] for shell in eigenvalues_next]
            WF = [[wf.copy() for wf in shell] for shell in WF_next]
            Rho_old = Rho_mixed.copy()
            if delta <= treshold: break
        return eigenvalues_next, WF_next

    def GetKohnShamEquation(self, WF=None):
        self.Vr = [[self.getAngularPotential(l) for l in range(nshell)]
                    for nshell in range(1, self.nshells + 1)]
        Veff = self.getNuclearPotential()
        if WF is not None:
            rho = self.getRadialDensity(WF)
            Veff = self.getEffectivePotential(rho)
        if self.Z > 2:
            eigenvalues_s, WF_s = self.runSCF_sorbs(WF)
            if eigenvalues_s is not None:
                if self.Z <= 25:
                    for n, shell in enumerate(eigenvalues_s):
                        self.E_start[n][0] = shell[0]
                # Build Veff from s-orbital density
                rho_s = self.getRadialDensity_sorbs(list(WF_s))
                # Also get full initial density (all orbitals)
                if WF is not None:
                    rho_full = self.getRadialDensity(WF)
                    # Mix: use mostly full density to support p,d states
                    rho_mix = 0.3 * rho_s + 0.7 * rho_full
                    Veff = self.getEffectivePotential(rho_mix)
                else:
                    Veff = self.getEffectivePotential(rho_s)
        eigenvalues = []; WF_init = []
        for nshell in range(1, self.nshells + 1):
            eigenvalues.append([]); WF_init.append([])
            for lshell in range(len(self.occupied[nshell-1])):
                if self.occupied[nshell-1][lshell] == 0:
                    eigenvalues[nshell-1].append(0)
                    WF_init[nshell-1].append(np.zeros_like(self.radial_grid))
                    continue
                Vr = self.Vr[nshell - 1][lshell]
                Vl = Veff + Vr
                E = self.E_start[nshell - 1][lshell]
                Bracket_Eigenvalue = self.getBracketEnergy(E, Vl, nshell, lshell)
                if Bracket_Eigenvalue is None:
                    # Try scanning from near zero
                    Bracket_Eigenvalue = self.getBracketEnergy(-0.5, Vl, nshell, lshell)
                if Bracket_Eigenvalue is None:
                    # Try scanning from deeper
                    Bracket_Eigenvalue = self.getBracketEnergy(E * 0.5, Vl, nshell, lshell)
                if Bracket_Eigenvalue is None:
                    raise RuntimeError(f"Cannot bracket n={nshell},l={lshell},E_start={E:.4f}")
                if self.CONTROLL:
                    print(f"First pass n={nshell}, l={lshell}: bracket={Bracket_Eigenvalue}, E_start={E}")
                E_f, WF_f = self.getEigenValue(Bracket_Eigenvalue, Vl, lshell)
                WF_init[nshell - 1].append(WF_f)
                eigenvalues[nshell - 1].append(E_f)
        if self.CONTROLL:
            print(f"First pass eigenvalues: {eigenvalues}")
        eigenvalues, WF_final = self.run_SCF(eigenvalues, WF_init, Veff)
        Rho = self.getRadialDensity(WF_final)
        return eigenvalues, WF_final, Rho

    # Get final energies

    def getXC_Energy(self, rho):
        Cx = (3.0/4.0) * (3.0/np.pi)**(1.0/3.0)
        rho_safe = np.maximum(rho, 1e-14)
        eps_x = -Cx * rho_safe**(1.0/3.0)
    
        # VWN correlation energy per particle (eps_c, NOT Vc)
        rs = (3.0 / (4.0 * np.pi * rho_safe))**(1.0/3.0)
        A = 0.0621814
        y0 = -0.10498
        b = 3.72744
        c = 12.93532
        y = np.sqrt(rs)
        Y = y**2 + b*y + c
        Q = np.sqrt(4.0*c - b**2)
        t1 = np.log(y**2 / Y)
        t2 = (2.0*b/Q) * np.arctan(Q/(2.0*y + b))
        t3 = b*y0 / (y0**2 + b*y0 + c)
        t4 = np.log((y - y0)**2 / Y)
        t5 = 2.0*(b + 2*y0)/Q
        t6 = np.arctan(Q/(2.0*y + b))
        eps_c = 0.5 * A * (t1 + t2 - t3*(t4 + t5*t6))
    
        r = self.radial_grid
        Exc = simpson((eps_x + eps_c) * rho * 4*np.pi*r**2, x=r)
        return Exc
    
    def getH_Energy(self,rho):
        r = self.radial_grid
        Vh = self.getHartreePotential(rho)
        Eh = simpson(0.5 * Vh * 4*np.pi*r**2*rho, r)
        return Eh
    
    def getEenuc(self,rho):
        r = self.radial_grid
        Ven = -self.Z*rho/self.radial_grid
        Een = simpson(Ven*4*np.pi*r**2,r)
        return Een

    def getEkin(self,WF):
        r =self.radial_grid
        cs    = [[CubicSpline(r, u) for u in shell] for shell in WF]
        gradd = [[spline.derivative(2) for spline in shell] for shell in cs]
        integrand = np.zeros_like(r)
        for n in range(1,self.nshells+1):
            for l in range(len(self.occupied[n-1])): 
                integrand_kin = cs[n-1][l](r) * gradd[n-1][l](r)
                integrand_ang  = ((cs[n-1][l](r))**2/r**2) * l*(l+1)
                integrand += self.occupied[n-1][l] * (integrand_kin - integrand_ang)
        Ekin = -0.5*simpson(integrand,r)
        return Ekin

    def getE_tot(self,rho,WF):
        Exc = self.getXC_Energy(rho)
        Eh = self.getH_Energy(rho)
        Een = self.getEenuc(rho)
        Ekin = self.getEkin(WF)
        E_tot = Exc+Eh+Een+Ekin
        return E_tot
