# -*- coding: utf-8 -*-  #
"""
Created on Tue May 19 10:26:00 2015

@author: wcgrizolli
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import sys
sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
from myOpticsLib import *

# %%

wavelength = 100e-9
Lx = 5e-3
npoints = 101
zDist = 10.00

Y, X = np.mgrid[-Lx/2:Lx/2:1j*npoints, -Lx/2:Lx/2:1j*npoints]

div_x = 100e-6
div_y = 100e-6

fwhm_x = wavelength/4/np.pi/div_x
fwhm_y = wavelength/4/np.pi/div_y


# %%
gBeamAst = gaussianBeamAst(X, Y, fwhm_x, fwhm_y, z=2, zxo=20, zyo=0, wavelength=wavelength)

intensity = np.abs(gBeamAst)
wf = np.unwrap(np.unwrap(np.angle(gBeamAst), axis=0), axis=1)

# %%

masked_wf = wf*1.0

masked_wf[intensity/np.max(intensity)<0.01] = np.nan

# %%

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X*1e3, Y*1e3, intensity, rstride=1, cstride=1,
                       cmap='jet',
                       linewidth=0, antialiased=False)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X*1e3, Y*1e3, masked_wf, rstride=1, cstride=1,
                       cmap='jet',
                       vmin=np.nanmin(masked_wf), vmax=np.nanmax(masked_wf),
                       linewidth=0, antialiased=False)
plt.show()




# %%
dx = X[0,1] - X[0,0]
dy = Y[1,0] - Y[0,0]

d2z_dx2 = np.diff(wf, 2, 1)/dx**2
d2z_dy2 = np.diff(wf, 2, 0)/dy**2

Rx = 2*np.pi/wavelength/d2z_dx2
Ry = 2*np.pi/wavelength/d2z_dy2

#
#plt.figure()
#
#plt.subplot(121)
#plt.hist(Rx.flatten(), 51)
#
#plt.subplot(122)
#plt.hist(Ry.flatten(), 51)
#
#plt.show()


print('Rx: {:.4f}, sdv: {:.4g}'.format(np.mean(Rx), np.std(Rx)))
print('Ry: {:.4f}, sdv: {:.4g}'.format(np.mean(Ry), np.std(Ry)))













