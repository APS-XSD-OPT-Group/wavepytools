#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
"""
Created on Thu Mar 20 16:46:25 2014

@author: wcgrizolli
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
import numpy as np
from matplotlib import cm
from wavepy.utils import easyqt

import wavepy.utils as wpu


def plot_surf_fast(dataz, pixelsize=[1., 1.], style='viridis',
                   ratio_x_y=1.0, scaling_z=1.0, distCamera=3):

#    _ = QtGui.QApplication([])


    pg.mkQApp()

    maxZ = np.nanmax(np.abs(dataz))

    z = dataz/maxZ*scaling_z

    # THE DEFINITIONS OF X AND Y ARE DIFFERENT IN PYQTGRAPH
    # AND THUS i NEET TO TRANSPOSE z

    [pixelSize_j, pixelSize_i] = pixelsize
    npoints_i = z.shape[1]
    npoints_j = z.shape[0]

    x = np.linspace(-.500, .500, npoints_j)
    y = np.linspace(-.500, .500, npoints_i)

    sizeX = pixelSize_i*npoints_j*ratio_x_y
    sizeY = pixelSize_j*npoints_i

    colorMap = _generateColormap(z, style, 1, 1)
    #    z[np.isnan(z)] = 0.0

    # Create a GL View widget to display data

    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('Lx = {:.3e}m, '.format(pixelSize_i*npoints_i) +
                     'Ly = {:.3e}m, '.format(pixelSize_j*npoints_j) +
                     'Max z = {:.3e}'.format(maxZ))
    w.setCameraPosition(distance=3)

    # plot

    p3 = gl.GLSurfacePlotItem(x=x, y=y, z=z, colors=colorMap, shader='shaded')

    p3.scale(1, sizeX/sizeY, 1)
    w.addItem(p3)

    # Add a grid to the view

    gx = gl.GLGridItem()
    gx.rotate(90, 1, 0, 0)
    gx.translate(0, -.5*sizeX/sizeY, .5*scaling_z)
    gx.scale(.05, .05*scaling_z, 1)
    gx.setDepthValue(10)
    w.addItem(gx)

    gy = gl.GLGridItem()
    gy.rotate(90, 0, 1, 0)
    gy.translate(-.5, 0, .5*scaling_z)
    gy.scale(.05*scaling_z, .05*sizeX/sizeY, 1)
    gy.setDepthValue(10)
    w.addItem(gy)

    gz = gl.GLGridItem()
    gz.scale(.05, .05*sizeX/sizeY, 1)
    gz.setDepthValue(10)
    w.addItem(gz)

    QtGui.QApplication.instance().exec_()


def _generateColormap(z, style='viridis', power=1, inverse=1):

    jetcmap = cm.get_cmap(style)  # generate a jet map with 10 values
    nColours = jetcmap.N
    jet_vals = jetcmap(np.arange(nColours))  # extract those values as an array

    zmin = np.nanmin(z)
    zmax = np.nanmax(z)

    colorIndex = np.rint(inverse*((z-zmin)/(zmax-zmin))**power *
                         (nColours-1)).astype(int)

    colorIndex[np.isnan(z)] = 1
    colorMap = jet_vals[colorIndex[:, :]]

    return colorMap


if __name__ == '__main__':
# %%

    dataFilename = easyqt.get_file_names()

    if len(dataFilename) == 1:
        dataFilename = dataFilename[0]
        dataZ, pixelSize, headerdic = wpu.load_sdf_file(dataFilename)
    else:

        y, x = np.mgrid[-1:1:100j, -1:1:100j]
        dataZ = np.sinc(10*x**2)*np.exp(-y**2/.5**2)

        pixelSize = [1/2, 1]


# %%

    plot_surf_fast(dataZ, pixelSize,
                   style='rainbow',
                   ratio_x_y=1.0, scaling_z=1.0)


    # %% OLD
#    with open(dataFilename) as f:
#        header = f.readline().split()
#
#    # % reshape from x,y,z for meshgrid format5.00
#    dataZ = np.loadtxt(dataFilename, comments='#')
#    dataZ -= np.nanmin(dataZ)
#    pixelSize_i = float(header[header.index('i,j') + 2])
#    pixelSize_j = float(header[header.index('i,j') + 4])

#    plot_surf_fast(dataZ, [pixelSize_i, pixelSize_j],
#                   style='rainbow',
#                   ratio_x_y=1.0, scaling_z=1.0)
