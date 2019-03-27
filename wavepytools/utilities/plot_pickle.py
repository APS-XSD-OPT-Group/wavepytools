#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
@author: Walan Grizolli

Hint: make the file executable and add to a directory in the PATH

to make executable in linux:

>>> chmod +x FILENAME

"""

import pickle
from wavepy.utils import easyqt
import os
import sys

import wavepy.utils as wpu

wpu._mpl_settings_4_nice_graphs()

#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


if len(sys.argv) != 1:
    fname = sys.argv[1]
else:
    fname = easyqt.get_file_names("Pickle File to Plot")[0]

file = open(fname,'rb')
figx = pickle.load(file)
whatever_plot = plt.show(block=True) # this lines keep the script alive to see the plot

# Done
# Below an example of how to get the data from a graph
# it is only possible to extract the data if you create your graphs with
# plot and imshow
# %%

if figx.axes[0].lines != []:

    curves = []

    for i in range(len(figx.axes[0].lines)):

        curves.append(figx.axes[0].lines[i].get_data())

    curves = np.asarray(curves)

if (figx.axes[0].images != []):

    data = figx.axes[0].images[0].get_array().data
    [xi, xf, yi, yf] = figx.axes[0].images[0].get_extent()

    ax = figx.axes[0].images[0].get_axes()

    title = figx.axes[0].images[0].axes.properties()['title']
    xlabel = figx.axes[0].images[0].axes.properties()['xlabel']
    ylabel = figx.axes[0].images[0].axes.properties()['ylabel']
    cmap = figx.axes[0].images[0].properties()['cmap'].name

    [[vmin, vmax], cmap] = wpu.plot_slide_colorbar(data, title=title,
                                                   xlabel=xlabel,
                                                   ylabel=ylabel,
                                                   extent=[xi, xf, yi, yf])

    # plot surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pixelsize = [(xf-xi)/data.shape[1], (yf-yi)/data.shape[0]]

    wpu.realcoordmatrix()

    xxGrid, yyGrid = np.meshgrid(np.linspace(xi, xf, data.shape[1]),
                                 np.linspace(yi, yf, data.shape[0]),
                                 indexing='xy')

    if np.all(np.isfinite(data)):
        surf = ax.plot_surface(xxGrid, yyGrid,  data,
                               vmin=vmin, vmax=vmax,
                               rstride=data.shape[0]//101+1,
                               cstride=data.shape[1]//101+1,
                               cmap=cmap, linewidth=.3, edgecolors='black')
    else:
        argNotNAN = np.isfinite(data)
        surf = ax.plot_trisurf(xxGrid[argNotNAN].flatten(),
                               yyGrid[argNotNAN].flatten(),
                               data[argNotNAN].flatten(),
                               vmin=vmin, vmax=vmax,
                               cmap=cmap, linewidth=0.02, shade=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title, fontsize=26)
    plt.colorbar(surf, shrink=.8, aspect=20)

    plt.tight_layout()
    #    plt.savefig('0p3_black.pdf')

    plt.show(block=True)

#    wpu.save_sdf_file(data, [1, 1],
#                      extraHeader= {'Title':title,
#                                    'Xlabel':xlabel,
#                                    'Ylabel':ylabel})







