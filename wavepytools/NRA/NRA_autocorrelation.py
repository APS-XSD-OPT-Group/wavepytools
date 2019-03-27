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


plt.style.use('ggplot')


#sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools/')

#import wgTools as wgt


import wavepy.utils as wpu

# %%

positions = np.array([-22.5, -7.5, -1.5, 0, 3, 12])


#positions = np.array([5, 27])

xvec = np.mgrid[-100:100:100001j]

nra_mask = xvec*0.0

hole_radius = .50

sigma = 20
intensity = np.exp(-(xvec/sigma)**2)

#intensity = 1./(np.pi*sigma*(1.+(positions*1e-6-0.0)**2/sigma**2))

intensity /= np.max(intensity)

for position in positions:

    nra_mask[np.where(np.abs(xvec - position) <= hole_radius)] = 1.0

nra_mask *= intensity

# %%

labels = ['33']
rho = [0.0]

for m, n in itertools.combinations(range(np.size(positions)), 2):
    print('mn: {:d}{:d} \t {:.3f}'.format(m, n, positions[m] - positions[n]))

    labels.append('{:d}{:d}'.format(m, n))
    rho.append(positions[m] - positions[n])


rho = np.abs(rho)
labels = np.array(labels)

# %%

plt.figure()
plt.plot(xvec, nra_mask, '-ro', label='S_j')
plt.legend()
plt.show()


# %%


corr = np.correlate(nra_mask, nra_mask, 'same')/np.sum(nra_mask**2)

#corr = np.correlate(nra_mask, nra_mask[::-1], 'same')

plt.figure()
plt.plot(xvec, corr, '-ro', label='S_j')

for i in range(rho.size):

    plt.annotate(labels[i], (rho[i], corr[np.where(xvec == rho[i])]),
                 fontsize =20)

plt.legend()
plt.show()
#
#
#
## %%
#
#plt.figure()
#plt.plot(rho[:max_j], np.log10(c_j_over_c_0[:max_j]/S_j[:max_j]), '-go')
#plt.show()
#




# %%

bla = np.abs(np.fft.fftshift(np.fft.fft(nra_mask)))





