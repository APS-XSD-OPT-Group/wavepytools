# -*- coding: utf-8 -*-  #
"""
Created on Mon Sep 12 16:40:44 2016

@author: grizolli
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wavepy.utils as wpu

from wavepy.utils import easyqt

import h5py

# %%
fname_dpc_h5 = easyqt.get_file_names("DPC Files")[0]


fh5 = h5py.File(fname_dpc_h5)

# %%


dpc_x = np.array(fh5['dpcHorizontal'])
dpc_y = np.array(fh5['dpcVertical'])


phase = np.array(fh5['phase'])



# =============================================================================
# %% parameters
# =============================================================================

size = np.shape(dpc_x)
pixelsize = .650*1e-6
xx, yy = wpu.realcoordmatrix(size[1], pixelsize, size[0], pixelsize)



#==============================================================================
# %% Integration
#==============================================================================


from wavepy.surface_from_grad import frankotchellappa, error_integration

result = frankotchellappa(dpc_x, dpc_y, reflec_pad=True)

result = np.real(result)

result *= -1

result -= np.min(result)

np.savetxt(fname_dpc_h5[:-3] + 'FC_integration.dat', result)

#==============================================================================
# %% Plot integrated 2d
#==============================================================================

wpu.plot_profile(xx[::5, ::5]*1e6, yy[::5, ::5]*1e6, result[::-5, ::-5])

# %%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(xx[::5, ::5]*1e6, yy[::5, ::5],
                       -result[::-5, ::-5],
                       rstride=result.shape[0] // 501 + 1,
                       cstride=result.shape[1] // 501 + 1,
                       cmap='viridis', linewidth=0.1)

plt.xlabel(r'$x$ [$\mu m$]')
plt.ylabel(r'$y$ [$\mu m$]')

plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()
plt.show(block=True)


# %%

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(xx[::5, ::5]*1e6, yy[::5, ::5],
                       -phase[::-5, ::-5]*1e6,
                       rstride=result.shape[0] // 501 + 1,
                       cstride=result.shape[1] // 501 + 1,
                       cmap='viridis', linewidth=0.1)

plt.xlabel(r'$x$ [$\mu m$]')
plt.ylabel(r'$y$ [$\mu m$]')

plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()
plt.show(block=True)
