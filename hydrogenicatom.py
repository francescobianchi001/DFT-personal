#!/usr/bin/env python3

from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt 


class HydrogenicAtom:

    def  __init__(self,Z):
        self.Z = Z
        
    def getEnergy(self,n):
        E = -self.Z**2/(2*n**2)
        return E
     
    def exponential_grid_log(r_min, r_max, n_points):
        x = np.linspace(np.log(r_min), np.log(r_max), n_points)
        r = np.exp(x)
        return r

    def getGSRadialFunction(self,r):
        u_s = 2*r*self.Z**(1.5)*np.exp(-self.Z*r)
        return u_s

    def getSlaterRadial(self,n,l,r,E=None):
        if E is not None:
            alpha = np.sqrt(-2*E)
        
        else:
            alpha= self.Z/n
        
        u_l = r**(l+1) * np.exp(-alpha*r)
        return u_l

    def getGShartree(self,r):
        Vh = np.zeros_like(r)
        u_s = self.getGSRadialFunction(r)
        n_s = 1/(4*np.pi) * 1/r**2 * u_s**2

        inner = (1/r)*cumulative_trapezoid(n_s*r**2,r,initial=0)
        outer = np.flip(cumulative_trapezoid(np.flip(n_s*r),np.flip(r),initial=0))

        Vh = 4*np.pi*(inner+outer)
        Vh[0]=Vh[1]

        return Vh,n_s

"""
#testing routine
r=np.linspace(0.0001,15.0,2000)
test=HydrogenicAtom(6)
hartree, rho=test.getGShartree(r)
print(test.getEnergy(1))
s1 = test.getGSRadialFunction(r)
s1_r= test.getSlaterRadial(3,2,r,-0.5)
#plt.plot(r,rho)
#plt.plot(r,hartree)
plt.plot(r,s1_r)
plt.plot(r,s1)
plt.show()
"""
"""
old potential:
        
        for i in range(len(r)):

            Vh[i]= 4*np.pi/r * quad(n_s[:i]*r[:i]**2, dx=dr) + 4*np.pi* quad(n_s[i:]*r[i:]**2, r, dx=dr)
        """
