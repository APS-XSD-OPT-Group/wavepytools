# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar  3 11:18:30 2015

@author: wcgrizolli
"""



import sys
import numpy as np
import matplotlib.pyplot as plt

import wavepy.utils as wpu
from myFourierLib import *
from myOpticsLib import *

#sys.path.append('/home/wcgrizolli/pythonWorkspace/srw/wpuools4srw')
#from wgtools4srw import *



##=========================================================#
# %% auxiliar functions
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


##=========================================================#
# %% sampling definition
##=========================================================#
wavelength = 12.398e-9  # 100eV
[Lx,Ly] = [4e-3, 4e-3]
# Mx = Lx^2/wavelength/z
[Mx,My] = [251, 251]
dx = Lx/Mx
dy = Ly/My

zz = 20.00 # dist to propag


print('sampling x='  + str(Mx))
print('sampling y='  + str(My))

# %%
if Mx > 1001 or  My > 1001:
    print('Sampling bigger than 1001^2, stoping the program')
#    sys.exit()

##=========================================================#
# %% 2D u1 function
##=========================================================#

# % circular


wx = 150e-6
wy = 240e-6
X,Y = np.meshgrid(np.linspace(-Lx/2,Lx/2,Mx,endpoint=False),
                  np.linspace(-Ly/2,Ly/2,My,endpoint=False))

u1_xy = circ(X, Y, wx, wy)


#    u1_xy = circ(X, Y, wx, wy)*tFuncLens(X, Y, wavelength, fx=(1/5.0+1/zz)**-1)* \
#    u1_xy = gaussianBeam(.2e-3, wavelength, z=0.0, L=Ly, npoints=My)

#    u1_xy = circ(X, Y, wx, wy)* \
#            gaussianBeam(10e-6, wavelength, 5.000, Lx, X.shape[0])

#u1_xy = circ(X, Y, wx, wy, 0, 80e-6) + circ(X, Y, wx, wy, 0,-80e-6)  # double slit

##=========================================================#
# %% Plot u1
##=========================================================#


factorX, unitStrX = wpu.choose_unit(X)
factorY, unitStrY = wpu.choose_unit(Y)
unitStrX = unitStrX + ' m'
unitStrY = unitStrY + ' m'
#
## U1
wpu.plot_profile(X*factorX, Y*factorY, np.abs(u1_xy**2),
                r'$x [' + unitStrX +']$',
                r'$y [' + unitStrY + ']$',
                r'Intensity [a.u.]',
                xo=0.0, yo=0.0,
                xunit=unitStrX, yunit=unitStrY)
plt.show(block=True)

##=========================================================#
# %% Propagation
##=========================================================#

print('WG: Propagation...')


#u2_xy = propTForIR(u1_xy, Lx, Ly, wavelength, zz)
#titleStr = str(r'propTForIR, zz=%.3fmm, Intensity [a.u.]'
#               % (zz*1e3))


u2_xy = propIR_RayleighSommerfeld(u1_xy,Lx,Ly,wavelength,zz)
titleStr = str(r'propIR_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
               % (zz*1e3))
#u2_xy = propTF_RayleighSommerfeld(u1_xy,Lx,Ly,wavelength,zz)
#titleStr = str(r'propTF_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
#               % (zz*1e3))


#    u2_xy, L2 = propFF(u1_xy, Lx, wavelength, zz)
#    titleStr = str(r'propFF, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))
#    X,Y = np.meshgrid(np.linspace(-L2/2,L2/2,Mx,endpoint=False),
#                      np.linspace(-L2/2,L2/2,My),endpoint=False)
#    print('WG: L2: %.5gmm' % (L2*1e3))
#    print('WG: X.shape: ', X.shape)
#
#    Lx2 = Lx/10.00
#    u2_xy = prop2step(u1_xy, Lx, Lx2, wavelength, zz)
#    X, Y = np.meshgrid(np.linspace(-Lx2/2, Lx2/2, Mx),
#                       np.linspace(-Lx2/2, Lx2/2, My))
#    titleStr = str(r'prop2step, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))

print('WG: Power 1: %.5g' % np.sum(np.abs(u1_xy)**2))
print('WG: Power 2: %.5g' % np.sum(np.abs(u2_xy)**2))
print('WG: Propagation: DONE!')

#u1_xy = None  # clear var
#    return u2_xy


##=========================================================#
# %% Plot u2
##=========================================================#

valueToMaskX = -.5e-3
#interpolateFlag = 1

# Crop and increase number of points

if valueToMaskX > 0.0000:

    print('WG: Crop data...')
    # mask2
    idx_1 = np.argmin(np.abs(X[0, :] + valueToMaskX/2))
    idx_2 = np.argmin(np.abs(X[0, :] - valueToMaskX/2))
    idx_3 = np.argmin(np.abs(Y[:, 0] + valueToMaskX/2))
    idx_4 = np.argmin(np.abs(Y[:, 0] - valueToMaskX/2))


    u2_xy = u2_xy[idx_3:idx_4, idx_1:idx_2]
    X = X[idx_3:idx_4, idx_1:idx_2]
    Y = Y[idx_3:idx_4, idx_1:idx_2]

    Lx = X[0,-1] - X[0,0]
    Ly = Y[-1,0] - Y[0,0]
    print('WG: Crop data: DONE!')


#    plt.imshow(np.abs(u2_xy)**2)

#    plt.figure()
#    plt.contourf(X*factorX, Y*factorY, np.abs(u2_xy)**2,256)
#    plt.show(block=True)


factorX, unitStrX = wpu.choose_unit(X)
factorY, unitStrY = wpu.choose_unit(Y)
unitStrX = unitStrX + ' m'
unitStrY = unitStrY + ' m'

wpu.plot_profile(X*factorX, Y*factorY, np.abs(u2_xy)**2,
                r'$x [' + unitStrX +']$',
                r'$y [' + unitStrY + ']$',
                r'Intensity [a.u.]',
                titleStr,
                xunit=unitStrX, yunit=unitStrY)
plt.show(block=True)



# %%
