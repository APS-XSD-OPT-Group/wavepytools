# -*- coding: utf-8 -*-  #
"""
Created on Mon Mar 23 14:28:03 2015

@author: wcgrizolli
"""


from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt
#from myFourierLib import gaussianBeam




def pathLength(r, y, alpha, zs, rp, yp, beta, zp, u, w, l):
    '''
        alpha and beta are angles to the normal
    '''
    from numpy import sqrt, sin, cos


    PL1 = (sqrt((zs-l)**2+(sin(alpha)*y-u+cos(alpha)*r)**2 +
          (-cos(alpha)*y-w+sin(alpha)*r)**2))

    PL2 = (sqrt((zp-l)**2+(-sin(beta)*yp-u+cos(beta)*rp)**2 +
          (cos(beta)*yp-w+sin(beta)*rp)**2))

    return PL1 + PL2, PL1, PL2



def paraboloidSurface(w, l, R, rho):
    return w**2/R + l**2/rho


#==============================================================================
# %% optical surface
#==============================================================================

nl, nw = 51, 51
r = 10.0
rp = 5.0

l, w = np.mgrid[-.001:.001:nl*1j, -.005:.005:nw*1j]

alpha = 88.00*wgt.deg2rad


R, rho = wgt.curvatureRadiusToroid(r, rp, alpha=alpha)

h_lw = paraboloidSurface(w, l, R=R, rho=rho)

dl = l[1, 0]-l[0, 0]
dw = w[0, 1]-w[0, 0]


#==============================================================================
# %% Path length
#==============================================================================


#PL, PL1, PL2 = pathLength(r, 0.0, alpha, 0.0,
#                          rp, 0.0, -alpha, 0.0,
#                          h_lw , w, l)


PL, PL1, PL2 = pathLength(r, -.5e-3, alpha, -.5e-3,
                          rp, -.5e-3, -alpha, -.5e-3,
                          h_lw , w, l)


#==============================================================================
# %% Plot
#==============================================================================

fig = plt.figure(figsize=(10., 7.))
ax = Axes3D(fig)

PLoo = PL[np.logical_and(w == 0.0, l == 0.0)]

surf = ax.plot_surface(l, w, PL - PLoo,
                       linewidth=0.0, rstride=1, cstride=1,
                       cmap=cm.jet, shade=True)


plt.xlabel(r'$l$')
plt.ylabel(r'$w$')

fig.colorbar(surf, shrink=0.5, aspect=8)
#plt.title('Intensity [ph/s/.1%bw/mm^2]', weight='bold')

plt.show()





