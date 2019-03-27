# -*- coding: utf-8 -*-  #
'''
Created on Thu Jun  8 14:00:08 2017

@author: grizolli
'''


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from wavepy.utils import rocking_3d_figure

import glob
import os


def plot_whatever(npoints):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = np.random.rand(npoints)*2-1
    yy = np.random.rand(npoints)*2-1
    zz = np.sinc(-(xx**2/.5**2+yy**2/1**2))
    ax.plot_trisurf(xx, yy, zz, cmap='viridis', linewidth=0.2, alpha = 0.8)

    plt.title('$\sinc$  Function')

    plt.show()

    return ax

# %%
def plot_whatever2(npoints):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = np.random.rand(npoints)*2-1
    yy = np.random.rand(npoints)*2-1

    zz = np.exp(-(xx**2/.5**2+yy**2/1**2))
    ax.plot_trisurf(xx, yy, zz, cmap='viridis', linewidth=0.2, alpha=0.9)

    xx = np.random.rand(npoints//4)*2-1
    yy = np.random.rand(npoints//4)*2-1

    zz = np.exp(-(xx**2/.5**2+yy**2/1**2))
    zz *= 1 + .1*(np.random.rand(npoints//4) - .5)
    ax.scatter(xx,yy,zz, c='g', marker='s', s=20)

    xx = np.random.rand(npoints//4)*2-1
    yy = np.random.rand(npoints//4)*2-1

    zz = np.exp(-(xx**2/.5**2+yy**2/1**2))
    zz *= 1 + .1*(np.random.rand(npoints//4) - .5)
    ax.scatter(xx,yy,zz, c='r', marker='o', s=20)

    plt.show()

    return ax


# %%

ax = plot_whatever(1000)

plt.pause(.5)

# %%

#rocking_3d_figure(ax, 'out2_050.ogv',
#                  elevAmp=45, azimAmpl=45,
#                  elevOffset=0, azimOffset=45,
#                  dpi=50, del_tmp_imgs=True)
#
#rocking_3d_figure(ax, 'out2_080.ogv',
#                  elevAmp=45, azimAmpl=45,
#                  elevOffset=0, azimOffset=45,
#                  dpi=80, del_tmp_imgs=True)
#
#rocking_3d_figure(ax, 'out2_100.ogv',
#                  elevAmp=45, azimAmpl=45,
#                  elevOffset=0, azimOffset=45,
#                  dpi=100, del_tmp_imgs=True)

# %%


rocking_3d_figure(ax, 'out_050.ogv',
                  elevAmp=60, azimAmpl=60,
                  elevOffset=10, azimOffset=45,
                  dpi=50, del_tmp_imgs=False)

rocking_3d_figure(None, 'out2_050.gif',
                  del_tmp_imgs=True)
