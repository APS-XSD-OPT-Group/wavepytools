# -*- coding: utf-8 -*-  #
"""
Created on Thu Jun  8 14:00:08 2017

@author: grizolli
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx=np.random.rand(50)
yy=np.random.rand(50)
zz=np.random.rand(50)
ax.scatter(xx,yy,zz, marker='o', s=20, c="goldenrod", alpha=0.6)

xx=np.random.rand(50)
yy=np.random.rand(50)
zz=np.random.rand(50)
ax.scatter(xx,yy,zz, c='b', marker='d', s=20, alpha=0.6)

ax.legend(['1', '2'])

ii_fname = 0





for ii in 2*np.pi*np.linspace(0, 1, 201):


    text = ax.text2D(0.05, 0.95, str('{:.2}deg'.format(ii)),
                     transform=ax.transAxes)

    ax.view_init(elev= 0 + 40*np.sin(2*ii), azim=90*np.sin(ii))

    plt.savefig("movie{:03d}.png".format(ii_fname))
    ii_fname += 1

    text.remove()

plt.show()
