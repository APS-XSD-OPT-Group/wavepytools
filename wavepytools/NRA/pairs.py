#! /bin/python
# -*- coding: utf-8 -*-  #
"""
Created 20161014

@author: wcgrizolli
"""


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
#import sys

#sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools/')

#import wgTools as wgt


import wavepy.utils as wpu



positions = np.array([-22.5, -7.5, -1.5, 0, 3, 12])


labels = ['33']
rho = [0.0]

for m, n in itertools.combinations(range(np.size(positions)), 2):
    print('mn: {:d}{:d} \t {:.3f}'.format(m, n, positions[m] - positions[n]))

    labels.append('{:d}{:d}'.format(m, n))
    rho.append(positions[m] - positions[n])


# %% convert to array

rho = np.abs(rho)
labels = np.array(labels)


# %%

labels = labels[np.argsort(rho)]
rho.sort()


# %%

print('mn \t j \t rho_j')

for j in range(np.size(rho)):

    print('{:s} \t {:d} \t {:.2f}'.format(labels[j], j, rho[j]))

# %%
sigma = 20e-6
#intensity = np.exp(-((positions-1)*1e-6/sigma)**2)

intensity = 1./(np.pi*sigma*(1.+(positions*1e-6-0.0)**2/sigma**2))

intensity /= np.max(intensity)

# %%

print('mn \t rho_j \t j \t S_j')
S_j = []
#for j in range(np.size(rho)):

for j in range(np.size(rho)):


    m = int(labels[j][0])
    n = int(labels[j][1])

    S_j.append(intensity[m]*intensity[n])


    print('mn: {:d}{:d} \t {:.2f} \t {:d} \t {:.3g}'.format(m, n, rho[j], j, intensity[m]*intensity[n]))

# %%


S_j = np.array(S_j)


c_j_over_c_0 = np.array([1e2, 0.8203, 0.7483, 0.595, 0.48, 0.5857, 0.3745, 0.3221])



max_j = c_j_over_c_0.shape[0]


# %%

plt.figure()
plt.plot(rho[:max_j], S_j[:max_j], '-ro', label='S_j')
plt.plot(rho[:max_j], c_j_over_c_0*1e-3, '-bo', label='c_j')
plt.legend()
plt.show()



# %%

plt.figure()
plt.plot(rho[:max_j], np.log10(c_j_over_c_0[:max_j]/S_j[:max_j]), '-go')
plt.show()











