# -*- coding: utf-8 -*-  #
"""
Created on %(date)s

@author: %(username)s
"""

# %%% imports cell
import numpy as np
import matplotlib.pyplot as plt

import wavepy.utils as wpu

# %%

import pickle

def _load_data_from_pickle(fname):

    fig = pickle.load(open(fname, 'rb'))
    fig.set_size_inches((12, 9), forward=True)

    plt.show(block=True)  # this lines keep the script alive to see the plot

    curves = []

    for i in range(len(fig.axes[0].lines)):

        curves.append(np.asarray(fig.axes[0].lines[i].get_data()))

    return curves

# %%

fname1 = 'CBhalfpi_3p4um_23p7keV_st8mm_step2mm_100ms_5images_01.pickle'

fname2 = 'CBhalfpi_3p4um_23p7keV_st8mm_step2mm_100ms_5images_02.pickle'

results1 = _load_data_from_pickle(fname1)

results2 = _load_data_from_pickle(fname2)

# %%

zvec1 = results1[0][0]*1e-3
contrastV1 = results1[0][1]*1e-2
contrastV1 /= np.max(contrastV1)
contrastH1 = results1[1][1]*1e-2
contrastH1 /= np.max(contrastH1)



zvec2 = results2[0][0]*1e-3
contrastV2 = results2[0][1]*1e-2
contrastV2 /= np.max(contrastV2)
contrastH2 = results2[1][1]*1e-2
contrastH2 /= np.max(contrastH2)

# %%


plt.figure(figsize=(10,6))
plt.subplot(121)

plt.plot(zvec1*1e3, contrastV1, '-b.')
plt.plot(zvec2*1e3, contrastV2, '-g.')



plt.subplot(122)
plt.plot(zvec1*1e3, contrastH1, '-b.')
plt.plot(zvec2*1e3, contrastH2, '-g.')

plt.show()
