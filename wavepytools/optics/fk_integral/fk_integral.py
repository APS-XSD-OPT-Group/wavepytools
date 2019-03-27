# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar 17 11:20:58 2015

@author: wcgrizolli
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt

#==============================================================================
#  auxiliar functions
#==============================================================================


def circ(wy, wz, y_vec, z_vec):  # circular

    Y, Z = np.meshgrid(y_vec, z_vec)
    out = Y*0.0
    out[abs((Y/wy)**2 + (Z/wz)**2) < 0.5**2] = 1.0
    out[abs((Y/wy)**2 + (Z/wz)**2) == 0.5**2] = .50
    return out


def paraboloidSurface(w, l, R, rho):
    return w**2/R**2 + l**2/rho**2


def pathLength(r, y, alpha, zs, rp, yp, beta, zp, u, w, l):

    from numpy import sqrt, sin, cos


    PL1 = (sqrt((zs-l)**2+(sin(alpha)*y-u+cos(alpha)*r)**2 +
          (-cos(alpha)*y-w+sin(alpha)*r)**2))

    PL2 = (sqrt((zp-l)**2+(-sin(beta)*yp-u+cos(beta)*rp)**2 +
          (cos(beta)*yp-w+sin(beta)*rp)**2))

    return PL1 + PL2, PL1, PL2



#==============================================================================
#  Some definitions
#==============================================================================

wavelength = 1.239842e-6/1e3
alpha = 88.0*np.pi/180
beta = alpha
r = 10.00
rp = 5.0

ny1, nz1 = 5, 7
ny2, nz2 = 9, 11
nl, nw = 13, 15

wgt.color_print('WG: Total number of points: %.4g' % (ny1*nz1*ny2*nz2*nl*nw))

#==============================================================================
#  U1
#==============================================================================

Ly1, Lz1 = 1e-3, 1e-3

y1_vec = np.mgrid[-Ly1/2:Ly1/2:ny1*1j]
z1_vec = np.mgrid[-Lz1/2:Lz1/2:nz1*1j]

dy1 = y1_vec[1] - y1_vec[0]
dz1 = z1_vec[1] - z1_vec[0]

u1_yz = circ(1e-3, .6e-3, y1_vec, z1_vec)


#==============================================================================
#  U2
#==============================================================================

Ly, Lz = .5e-3, .5e-3

y2_vec = np.mgrid[-Ly/2:Ly/2:ny2*1j]
z2_vec = np.mgrid[-Lz/2:Lz/2:nz2*1j]

#Y2, Z2 = np.meshgrid(y2_vec, z2_vec)
u2_yz = np.zeros((nz2, ny2), dtype=complex)


#==============================================================================
#  optical surface
#==============================================================================

l, w = np.mgrid[-.1:.1:nl*1j, -.5:.5:nw*1j]

R, rho = wgt.curvatureRadiusToroid(r, rp, alpha=alpha)

u = paraboloidSurface(w, l, R=R, rho=rho)

dl = l[1, 0]-l[0, 0]
dw = w[0, 1]-w[0, 0]

#==============================================================================
#  Integration
#==============================================================================

print("WG: Integration...")

for j2 in range(len(y2_vec)):
    print('\n################ ' + str(j2))
    for i2 in range(len(z2_vec)):
        print(i2)

        for j1 in range(len(y1_vec)):
            for i1 in range(len(z1_vec)):

                PL, PL1, PL2 = pathLength(r, y1_vec[j1], alpha, z1_vec[i1],
                                          rp, y2_vec[j2], beta, z2_vec[i2],
                                          u, w, l)
                gFunc = np.sum((np.exp(1j*2*np.pi/wavelength*PL)/(PL1*PL2)*dw*dl))
                u2_yz[i2,j2] += gFunc*u1_yz[i1, j1]*dy1*dz1
#                u2_yz[i2,j2] += j2

Y2, Z2 = np.meshgrid(y2_vec, z2_vec)

#==============================================================================
#  save for external use
#==============================================================================

np.savez('u2_fk_' + wgt.datetimeNowStr(), u2_yz=u2_yz, Y=Y2, Z=Z2)
print("Bye!")


#==============================================================================
#  Plot
#==============================================================================


plt.contourf(Y2*1e3, Z2*1e3, np.abs(u2_yz), 256)

plt.show()







