# -*- coding: utf-8 -*-  #
"""
Created on %(date)s

@author: %(username)s
"""

# %%% imports cell
import numpy as np
import matplotlib.pyplot as plt

import xraylib

import wavepy.utils as wpu

wpu._mpl_settings_4_nice_graphs()

# %% Lens
material = 'Be'
density = xraylib.ElementDensity(4)
curv_radius = 100e-6
nlenses = 1
nsurfaces = 1*nlenses

title = 'Berylium' + ' Lens, Curv. Radius ' + \
        ' {:.1f} um, {:d} curved surfaces'.format(curv_radius*1e6, nsurfaces)

phenergy = np.arange(7e3, 9e3, 100)

# %% Obtain delta refractive index

delta = phenergy*0.0

for i in range(delta.shape[0]):

    delta[i] = 1 - xraylib.Refractive_Index_Re("Be", phenergy[i]/1e3, density)


# %%

focal_d = curv_radius/delta/nsurfaces


# %% q_arm vs energy

plt.figure()
plt.plot(phenergy*1e-3, focal_d, '.-')
plt.xlabel('ph Energy [eV]')
plt.ylabel('Focal distance [m]')
plt.title(title)
#wpu.save_figs_with_idx()
plt.show(block=True)


# %% Geometric optics focus

source_dist = 27.12 # p_arm

imag_dist = focal_d*source_dist/(source_dist - focal_d)



# %% q_arm vs energy

plt.figure()
plt.plot(phenergy*1e-3, imag_dist, '.-')
plt.xlabel('ph Energy [eV]')
plt.ylabel('Image distance [m]')
plt.title(title + '\nImage distance [m]. Source at {:.2f}m'.format(source_dist))

#wpu.save_figs_with_idx()
plt.show(block=True)


# %% fixed energy calculation

single_energy = 8000
single_focal_d = focal_d[phenergy//1 == single_energy]

# %% print results

q_arm = 1/(1/single_focal_d - 1/source_dist)

print(title)
print('Results at {:} eV'.format(single_energy))
print('p_arm: {:}'.format(source_dist))
print('q_arm: {:}'.format(q_arm))
print('focal_d: {:}'.format(single_focal_d))
print('Delta at {}eV : {}'.format(single_energy, delta[phenergy//1 == single_energy]))

print('MAGNIFICATION: {:}'.format(q_arm/source_dist))

# %%

source_dist_vec = np.arange(23.0, 40.00, .1)

q_single_energy = source_dist_vec*single_focal_d/(source_dist_vec - single_focal_d)


plt.figure()
plt.plot(source_dist_vec, q_single_energy, '.-k')
plt.xlabel('Source distance [m]')
plt.ylabel('Image distance [m]')
plt.title(title + '\nEph = {:.2f} KeV'.format(single_energy*1e-3))

#wpu.save_figs_with_idx()
plt.show(block=True)

