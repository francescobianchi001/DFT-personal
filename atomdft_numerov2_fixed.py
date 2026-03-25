#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
import scipy as sy
import matplotlib.pyplot as plt
import math as mt
from starting_energies import STARTatomicE
from hydrogenicatom import HydrogenicAtom
import sys
import copy

class AtomicDFT:

    def __init__(self,grid,Z):
       self.Z = Z
       self.radial_grid=grid
       self.START=True
       self.CONTROLL =True

    def GetstartOrbSD(self,N):
        cap = [2, 2, 6,10]
        n=mt.ceil(N/2)
        fact = n-2
        ite= n-fact
        if fact < 0:
            ite = 1
        else:
            ite= n-fact
        self.occupied = [[mt.ceil(N/n)]+[N-sum(cap[:s+1])]*s for s in range(0,ite) if N<=sum(cap[:ite+1])]
        if self.occupied == []:
            return print("Error: number of electrons exited 10")
        else:
            self.occupied

        self.nshells = ite


    def GetOrbitals(self):

        if self.START==True:
            self.GetstartOrbSD(self.Z)
        self.E_start= STARTatomicE[(str(self.Z))]
        self.radial_WF= []
        for nshell in range(1,self.nshells+1):
            self.radial_WF.append([])
            for lshell in range(nshell):
                E = self.E_start[nshell-1][lshell]
                radial = HydrogenicAtom.getSlaterRadial(self.Z,nshell,lshell,self.radial_grid,E)
                norm = simpson(radial**2, x=self.radial_grid)
                radial = radial/np.sqrt(norm)
                self.radial_WF[nshell-1].append(radial)
        return self.radial_WF

    def getRadialDensity(self,WF):
        rho = []
        rho = [[self.occupied[i][j] * (WF[i][j])**2 for j in range(i+1)] for i in range(len(WF))]
        rho= 1/(4*np.pi*self.radial_grid**2)* np.sum([orb for shell in rho for orb in shell],axis=0)
        return rho

    #POTENTIALS

    def getVXC(self,rho):
        rho = np.maximum(rho, 1e-14)
        # Wigner-Seitz radius
        rs = (3.0 / (4.0 * np.pi * rho))**(1.0/3.0)

        # VWN parameters (paramagnetic)
        A = 0.0621814
        y0 = -0.10498
        b = 3.72744
        c = 12.93532
        Cx = 0.5 * 3.0 * (9.0 / (32.0 * np.pi**2))**(1/3)

        # Intermediate variables
        y = np.sqrt(rs)
        Y = y**2 + b*y + c
        Q = np.sqrt(4.0*c - b**2)

        t1 = np.log(y**2 / Y)
        t2 = (2.0*b/Q) * np.arctan(Q / (2.0*y + b))
        t3 = b * y0 / (y0**2 + b*y0 + c)
        t4 = np.log((y - y0)**2 / Y)
        t5 = 2.0 * (b + 2*y0) / Q
        t6 = np.arctan(Q / (2.0*y + b))
        ec = 0.5 * A * (t1 + t2 - t3*(t4 + t5*t6))
        dec = 0.5 * A * (c*(y - y0) - b*y*y0) / ((y - y0) * Y)

        # Exchange and correlation potentials
        Vx = -(4.0/3.0) * Cx / rs
        Vc = ec - (1.0/3.0) * dec

        Vxc = Vx + Vc
        self.Vxc=Vxc
        return Vxc

    def getHartreePotential(self,rho):
        r = self.radial_grid
        inner = np.zeros_like(r)
        outer = np.zeros_like(r)
        for i in range(len(r)):
                inner[i] = simpson(rho[:i+1] * r[:i+1]**2, r[:i+1])
                outer[i] = simpson(rho[i:] * r[i:], r[i:])

        Vh = 4*np.pi*(inner/r + outer)
        Vh[0] = Vh[1]
        return Vh

    def getNuclearPotential(self):
        Ven= -self.Z/self.radial_grid
        self.Ven=Ven
        return Ven

    def getEffectivePotential(self,rho):
        """
        if self.Z==1:
            Veff=self.getNuclearPotential()
            return Veff"""
        Vxc = self.getVXC(rho)
        Vh = self.getHartreePotential(rho)
        Ven = self.getNuclearPotential()
        Veff = Vxc + Vh + Ven
        return Veff

    def getAngularPotential(self,l):
        Vr= l*(l+1)/(2*self.radial_grid**2)
        return Vr

    # Diagonalisation
    def getNOdes(self,WF):
        nodes = np.sum(WF[:-1]*WF[1:]<=0)
        return nodes

    # FIX 2: corrected argument order from (e,Veff,l,r) to (e,l,Veff)
    # FIX 4: removed stray plt.plot(WF) inside loop
    def getF(self,e1,e2,Veff,l,r):
        E = np.linspace(e1,e2,2000)
        F_tot = []
        F_old = 0
        Tp=[]
        WF_tot=[]
        for e in E:
                WF,F= self.getWaveFunction_numerov(e,l,Veff)
                if F_old*F<0:
                        Tp.append(e)
                F_old = F
                print(F,e)
                F_tot.append(F)
                WF_tot.append(WF)
        return F_tot,Tp

    def getEV(self,bracket_Eigenvalue,Veff,l):
        def func(x):
            WF,F=self.getWaveFunction_numerov(x,l,Veff)
            return F,WF
        def root_f(E):
            F,_ = func(E)
            return F
        eigenvalue= brentq(root_f,bracket_Eigenvalue[0],bracket_Eigenvalue[1])
        _,WF=func(eigenvalue)
        return eigenvalue,WF

    def getEigenValue(self, bracket_Eigenvalue, Veff, l):
        def root_f(E):
            _, F = self.getWaveFunction_numerov(E, l, Veff)
            return F
        eigenvalue = brentq(root_f, bracket_Eigenvalue[0], bracket_Eigenvalue[1])
        F_check = root_f(eigenvalue)
        print(f"F at eigenvalue: {F_check:.2e}")
        WF, _ = self.getWaveFunction_numerov(eigenvalue, l, Veff)
        return eigenvalue, WF

    def getBracketEnergy(self,E_start,Veff,n,l):
        delta = max(0.001*abs(E_start), 0.01)
        target_nodes = n - l - 1
        # Scan upward from E_start
        e1 = E_start
        WF, F1 = self.getWaveFunction_numerov(e1, l, Veff)
        for _ in range(2000):
            e2 = e1 + delta
            if e2 >= 0:
                break
            WF, F2 = self.getWaveFunction_numerov(e2, l, Veff)
            if np.isnan(F2):
                break
            nodes = self.getNOdes(WF)
            if F2 * F1 < 0 and nodes == target_nodes:
                return (e1, e2)
            e1, F1 = e2, F2
        # Scan downward from E_start
        e1 = E_start
        WF, F1 = self.getWaveFunction_numerov(e1, l, Veff)
        for _ in range(2000):
            e2 = e1 - delta
            WF, F2 = self.getWaveFunction_numerov(e2, l, Veff)
            if np.isnan(F2):
                break
            nodes = self.getNOdes(WF)
            if F2 * F1 < 0 and nodes == target_nodes:
                return (e2, e1)
            e1, F1 = e2, F2
        return None

    def getWaveFunction_numerov(self,e,l,Veff):
        h = self.radial_grid[1]-self.radial_grid[0]
        g = 2*(e-Veff)
        c1 = 5 * h**2/12
        c2=h**2/12
        y_right=np.zeros_like(self.radial_grid)
        y_right[-1]= np.exp(-np.sqrt(-2*e)*self.radial_grid[-1])
        y_right[-2]= np.exp(-np.sqrt(-2*e)*self.radial_grid[-2])
        y_left = np.zeros_like(self.radial_grid)
        y_left[0] = self.radial_grid[0]**(l+1)
        y_left[1] = self.radial_grid[1]**(l+1)
        # Find the OUTER classical turning point: the last index where g > 0, i.e. where
        # the classically allowed region ends. For l>0, the centrifugal barrier makes g<=0
        # near r=0, so we must skip the inner forbidden region and find the outer boundary.
        positive_indices = np.where(g > 0)[0]
        if len(positive_indices) > 0:
            tp = min(positive_indices[-1] + 1, len(g) - 3)
        else:
            tp = np.argmax(g <= 0)
        for n in range(2,tp+2):
            den = 1/(1+c2*g[n])
            num = 2*y_left[n-1]*(1-c1*g[n-1])-y_left[n-2]*(1+c2*g[n-2])
            y_left[n] = num * den
        for n in range(len(g)-3,tp-2,-1):
            den = 1/(1+c2*g[n])
            num = 2*y_right[n+1]*(1-c1*g[n+1])-y_right[n+2]*(1+c2*g[n+2])
            y_right[n] = num * den
        if y_left[tp]*y_right[tp]<0:
            y_right = -y_right
        scale = y_right[tp]/y_left[tp]
        y_left = y_left *scale
        Y = np.concatenate((y_left[:tp],y_right[tp:]))
        norm= simpson(Y**2,x=self.radial_grid)
        y = Y/np.sqrt(norm)
        dL = y_left[tp+1]-y_left[tp-1]
        dR = y_right[tp+1]-y_right[tp-1]
        F = 1/np.sqrt(norm)*(dL-dR)/(2*h)
        # FIX 5: removed noisy "one went" debug print
        if self.CONTROLL==True:
            print(f"DEBUG: tp={tp}, E_{l}={e},argmax={len(g)-np.argmax(g[::-1] <= 0)}, negativevalues={np.argmax(g[::-1]<=0)},F={F}")
        self.tp = tp
        return y,F

    #Solve KohnSham eq:

    # FIX 1 (CRITICAL): convergence check now happens BEFORE updating eigenvalues_old
    def run_SCF(self,eigenvalues,WF,Veff,max_iter=20, treshold=1e-6):
        eigenvalues_old = copy.deepcopy(eigenvalues)
        for iteration in range(max_iter):
            V_old = Veff.copy()
            Rho = self.getRadialDensity(WF)
            Veff = self.getEffectivePotential(Rho)
            eigenvalues_next = []
            WF_next=[]
            damp = 0.5
            for nshell in range(1,self.nshells+1):
                eigenvalues_next.append([])
                WF_next.append([])
                for lshell in range(nshell):
                    Vr = self.Vr[nshell-1][lshell]
                    Veff_mixed = (1-damp)*(V_old)+(damp)*(Veff)
                    Veff_next = Veff_mixed + Vr
                    E_start = eigenvalues_old[nshell-1][lshell]
                    Bracket = self.getBracketEnergy(E_start,Veff_next,nshell,lshell)
                    E_f,WF_f= self.getEigenValue(Bracket,Veff_next,lshell)
                    WF_next[nshell-1].append(WF_f)
                    eigenvalues_next[nshell-1].append(E_f)
            self.eigenvalues = eigenvalues_next
            flat_next = np.array([e for shell in eigenvalues_next for e in shell])
            flat_old = np.array([e for shell in eigenvalues_old for e in shell])
            delta = np.max(np.abs(flat_next - flat_old))
            eigenvalues_old = copy.deepcopy(eigenvalues_next)
            WF= copy.deepcopy(WF_next)
            if delta <= treshold:
                break

        return eigenvalues_next,WF_next

    def GetKohnShamEquation(self,WF=None):
        Veff = self.getNuclearPotential()
        if WF != None:
            rho= self.getRadialDensity(self.radial_WF)
            Veff= self.getEffectivePotential(rho)
        self.Vr = [[self.getAngularPotential(l) for l in range(nshell)] for nshell in range(1,self.nshells+1)]
        eigenvalues = []
        WF = []
        for nshell in range(1,self.nshells+1):
            eigenvalues.append([])
            WF.append([])
            for lshell in range(nshell):
                Vr = self.Vr[nshell-1][lshell]
                Vl = Veff + Vr
                E = self.E_start[nshell-1][lshell]
                Bracket_Eigenvalue = self.getBracketEnergy(E,Vl,nshell,lshell)
                E_f,WF_f = self.getEigenValue(Bracket_Eigenvalue,Vl,lshell)
                WF[nshell-1].append(WF_f)
                eigenvalues[nshell-1].append(E_f)

        eigenvalues,WF = self.run_SCF(eigenvalues,WF,Veff)
        Rho= self.getRadialDensity(WF)
        Vxc = self.getVXC(Rho)
        Vh= self.getHartreePotential(Rho)
        return eigenvalues,WF,Rho
