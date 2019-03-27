# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar  3 11:18:30 2015

@author: wcgrizolli
"""

import sys


import numpy as np
import matplotlib.pyplot as plt

from myFourierLib import *


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt

sys.path.append('/home/wcgrizolli/pythonWorkspace/srw/wgTools4srw')
from wgTools4srw import *

##=========================================================#
# %% sampling definition
##=========================================================#
wavelength = 1.2398e-9  # 1KeV
[Lx, Ly] = [2.5e-3, 2.5e-3]
# Mx = Lx^2/wavelength/z
[Mx, My] = [1001, 1001]
dx = Lx/Mx
dy = Ly/My

#zz = 1.00  # XXX: dist to propag
#Lx2 = Lx

zz = .00322808  # XXX: dist to propag
Lx2 = Lx/2500.0

print('WG: sampling x=' + str(Mx))
print('WG: sampling y=' + str(My))

# %%
if Mx > 1001 or My > 1001:
    wgt.color_print('WG: Sampling bigger than 1001^2, stoping the program')
#    sys.exit()

##=========================================================#
# %% 2D u1 function
##=========================================================#


def circ(X, Y, wx, wy, Xo=0.0, Yo=0.0):  # circular
    out = X*0.0
    out[abs(((X-Xo)/wx)**2 + ((Y-Yo)/wy)**2) < 0.5**2] = 1.0
    out[abs(((X-Xo)/wx)**2 + ((Y-Yo)/wy)**2) == 0.5**2] = .50
    return out


def tFuncLens(X, Y, wavelength, fx=1e23, fy=1e23):
    return np.exp(-1j*2*np.pi/wavelength/2/fx*(X**2+Y**2))


def tFuncZP(X, Y, wavelength, fx=1e23, fy=1e23):
    return .5*(1.0 + np.sign(np.cos(np.pi/wavelength/fx*(X**2 + Y**2))))


wx = 200e-6
wy = 200e-6
X, Y = np.meshgrid(np.linspace(-Lx/2, Lx/2, Mx), np.linspace(-Ly/2, Ly/2, My))

print('WG: Creating Source Wave u1...')

#u1_xy = circ(X, Y, wx, wy)*tFuncZP(X, Y, wavelength, fx=zz)
u1_xy = circ(X, Y, wx, wy)*tFuncLens(X, Y, wavelength, fx=zz)

#u1_xy = circ(X, Y, wx, wy, 0, 80e-6) + circ(X, Y, wx, wy, 0,-80e-6)  # double slit

print('WG: Creating Source Wave u1: DONE!')

##=========================================================#
# %% Propagation
##=========================================================#

print('WG: Propagation...')


if Lx == Lx2:
    u2_xy = propTForIR(u1_xy, Lx, Ly, wavelength, zz)
    X2, Y2 = X, Y
else:
    u2_xy = prop2step(u1_xy, Lx, Lx2, wavelength, zz)
    X2, Y2 = np.meshgrid(np.linspace(-Lx2/2, Lx2/2, Mx),
                         np.linspace(-Lx2/2, Lx2/2, My))

print('WG: Propagation: DONE!')

##=========================================================#
# %% Plot u1
##=========================================================#




saveFigure = 0

print('WG: Plot u1...')



factorX, unitStrX = wgt.chooseUnit(X)
factorY, unitStrY = wgt.chooseUnit(Y)

unitStrX = unitStrX + ' m'
unitStrY = unitStrY + ' m'

# %% U1



wgt.plotProfile(X*factorX, Y*factorY, np.abs(u1_xy),
                r'$x [' + unitStrX +']$',
                r'$y [' + unitStrY + ']$',
                r'Intensity [a.u.]',
                xo=0.0, yo=0.0,
                unitX=unitStrX, unitY=unitStrY)


# %% U1

#wgt.plotProfile(X*factorX, Y*factorY, np.abs(u1_xy),
#                r'$x [' + unitStrX +']$',
#                r'$y [' + unitStrY + ']$',
#                r'Intensity [a.u.]',
#                xo=0.0, yo=0.0,
#                unitX=unitStrX, unitY=unitStrY)
if saveFigure:
    outputFigureName = wgt.datetimeNowStr() + '_u1.png'
    plt.savefig(outputFigureName)
    print('WG: Figure saved at %s!\n' % (outputFigureName))
    plt.close()
else:
    plt.show(block=True)

print('WG: Plot u1: DONE!')

##=========================================================#
# %% Plot u2
##=========================================================#

print('WG: Plot u2...')

factorX2, unitStrX2 = wgt.chooseUnit(X2)
factorY2, unitStrY2 = wgt.chooseUnit(Y2)

unitStrX2 = unitStrX2 + ' m'
unitStrY2 = unitStrY2 + ' m'


## U1

wgt.plotProfile(X2*factorX2, Y2*factorY2, np.abs(u2_xy),
                r'$x [' + unitStrX2 + ']$',
                r'$y [' + unitStrY2 + ']$',
                r'Intensity [a.u.]',
                unitX=unitStrX2, unitY=unitStrY2)

if saveFigure:
    outputFigureName = wgt.datetimeNowStr() + '_u2.png'
    plt.savefig(outputFigureName)
    print('WG: Figure saved at %s!\n' % (outputFigureName))
    plt.close()
else:
    plt.show(block=True)

print('WG: Plot u2: DONE!')
# %%
