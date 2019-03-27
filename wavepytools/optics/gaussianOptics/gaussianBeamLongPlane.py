# -*- coding: utf-8 -*-  #
"""
Created on Tue May 19 10:26:00 2015

@author: wcgrizolli
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
from myOpticsLib import *

# %%


Lx = 2e-6
fwhm = 80e-9
wavelength = 100e-9

zDist = 1.00e-6

npointsX = 1001
npointsY = 1
npointsZ = 1001
zvec = np.linspace(-zDist, zDist,npointsZ)

npoints = 51

gBeam = np.ones((npoints, npoints, npointsZ), dtype=complex)*np.nan

# %%

X, Z  = np.mgrid[-Lx/2:Lx/2:1j*npointsX, -zDist:zDist:1j*npointsZ]


# %%

gBeam = gaussianBeam(X, 0, fwhm, Z, wavelength)

gBeam2 = gaussianBeam(X, 0, fwhm, Z, wavelength)/ \
            gaussianBeam(X*0.0, 0, fwhm, Z, wavelength)
#print gBeam
# %%


#
#for zwert in zvec:
#    gBeam[:,:,zvec == zwert] = np.reshape(gaussianBeam(fwhm, wavelength, zwert, Lx, npoints),
#                                (npoints,npoints,1))



# %%


intVec = np.abs(gBeam)

intVec2 = np.abs(gBeam2)


# %%
plt.figure()
plt.contourf(Z*1e6, X*1e6,
                intVec*np.cos(2*np.pi*Z/wavelength)**2, 101)
plt.contour(Z*1e6, X*1e6,
            intVec2, levels=[.5], linewidths=1.5,
            colors=['r'], linestyles=['dashed'])

plt.xlabel(r'$z$ [$\mu$m]', fontsize=18)
plt.ylabel(r'$r$  [$\mu$m]', fontsize=18)

plt.title(str('Gaussian Beam, $\lambda$ = %.1f nm, FWHM = %.1f nm' %
                    (wavelength*1e9, fwhm*1e9)), fontsize=18)

plt.show()

# %%


