# -*- coding: utf-8 -*-  #
"""
Created on Sat Oct 15 17:25:31 2016

@author: grizolli
"""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import cm

from peak_detect import peakdet


xs = np.arange(0, 5*np.pi, 0.05)
data = np.sin(xs) + .1*np.random.rand(xs.shape[0])
peakind = signal.find_peaks_cwt(data, np.arange(9,10))
peakind, xs[peakind], data[peakind]


max_array, min_array = peakdet(data,.1)


# %%
plt.figure()

plt.plot(xs, data, '-.')
plt.plot(xs[peakind], data[peakind], 'or')

plt.plot(xs[np.array(max_array[:,0], dtype=int)], max_array[:,1], 'xg')


plt.show(block=True)


np.array
