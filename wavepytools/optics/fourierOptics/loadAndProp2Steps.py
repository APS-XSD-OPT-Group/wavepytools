# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar  3 11:18:30 2015

@author: wcgrizolli
"""

import sys
import numpy as np
import matplotlib.pyplot as plt



sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt
from myFourierLib import *


from memory_profiler import profile

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
# %% sampling and base definition
##=========================================================#
#@profile
def main():
    wavelength = 1.2398e-9  # 1KeV
    Lx = 2e-3
    #zz = 1.0  # XXX: dist t1o propag

    zz = .01000  # XXX: dist to propag
    zoomFactor = 1/500.0
    Lx2 = Lx*zoomFactor



    ##=========================================================#
    # %% 2D analytical function.
    ##=========================================================#

    #npoints = 1001
    #
    #Y, X = np.mgrid[-Lx/2:Lx/2:1j*npoints, -Lx/2:Lx/2:1j*npoints]
    #
    #wx = 200e-6
    #wy = 200e-6
    #
    #print('WG: Creating Source Wave u1...')
    #
    ##u1_xy = circ(X, Y, wx, wy)*tFuncLens(X, Y, wavelength, fx=(1/5.0+1/zz)**-1)
    #
    ## %% gaussian beam
    #u1_xy = (tFuncLens(X, Y, wavelength, fx=(1/5.0+1/zz)**-1) * circ(X, Y, wx, wy) *
    #         gaussianBeam(10e-6, wavelength, 5.000, Lx, X.shape[0]))
    #
    ## %% double slit
    ##u1_xy = circ(X, Y, wx, wy, 0, 80e-6) + circ(X, Y, wx, wy, 0,-80e-6)
    #
    #print('WG: Creating Source Wave u1: DONE!')

    ##=========================================================#
    # %% 2D load data
    ##=========================================================#

    u1_xy = np.load('emWave.npz')['emWave']
    X = np.load('emWave.npz')['x']
    Y = np.load('emWave.npz')['y']

    [Mx,My] = u1_xy.shape

    print('WG: u1_xy.shape: %d, %d' % (Mx, My))

    Lx = X[0, -1] - X[0, 0]
    Ly = Y[-1, 0] - Y[0, 0]

    print('WG: Lx = %.3f mm' % (Lx*1e3))
    print('WG: Ly = %.3f mm' % (Ly*1e3))

    valueToMaskX = 2e-3
    interpolateFlag = 1

    # %% Crop and increase number of points

    if valueToMaskX > 0.0000:

        print('WG: Crop data...')
        # mask2
        idx_1 = np.argmin(np.abs(X[0, :] + valueToMaskX/2))
        idx_2 = np.argmin(np.abs(X[0, :] - valueToMaskX/2))
        idx_3 = np.argmin(np.abs(Y[:, 0] + valueToMaskX/2))
        idx_4 = np.argmin(np.abs(Y[:, 0] - valueToMaskX/2))


        u1_xy = u1_xy[idx_3:idx_4, idx_1:idx_2]
        X = X[idx_3:idx_4, idx_1:idx_2]
        Y = Y[idx_3:idx_4, idx_1:idx_2]

        Lx = X[0,-1] - X[0,0]
        Ly = Y[-1,0] - Y[0,0]

        [Mx,My] = u1_xy.shape

        print('WG: new Lx = %.3f mm' % (Lx*1e3))
        print('WG: new Ly = %.3f mm' % (Ly*1e3))
        print('WG: new shape after crop: %d, %d' % (Mx,My))
        print('WG: Crop data: done!')


    # %% increase resolution using interpolation

    if interpolateFlag:

    #    from scipy import interpolate
        from scipy.interpolate import griddata

        print('WG: Interpolation to increase resolution...')

        nPointsInterp = 1001j

        grid_y, grid_x = np.mgrid[X[0, 0]:X[0, -1]:nPointsInterp,
                                  X[0, 0]:X[0, -1]:nPointsInterp]

        grid_z0_real = griddata(np.concatenate((X.reshape(-1, 1),
                                                Y.reshape(-1, 1)), axis=1),
                                np.real(u1_xy).flat[:],
                                (grid_x, grid_y),
                                method='cubic',
                                fill_value=0)

        grid_z0_im = griddata(np.concatenate((X.reshape(-1, 1),
                                              Y.reshape(-1, 1)), axis=1),
                              np.imag(u1_xy).flat[:],
                              (grid_x, grid_y),
                              method='cubic',
                              fill_value=0)

        u1_xy = grid_z0_real + 1j*grid_z0_im
        X = grid_x
        Y = grid_y
        Lx = X[0,-1] - X[0,0]
        Ly = Y[-1,0] - Y[0,0]

        [Mx,My] = u1_xy.shape

        print('WG: Lx = %.3f mm' % (Lx*1e3))
        print('WG: Ly = %.3f mm' % (Ly*1e3))

        print('WG: done!')
        print('WG: new shape resize: %d, %d' % (Mx, My))
        print('WG: new Lx = %.3f mm' % (Lx*1e3))
        print('WG: new Ly = %.3f mm' % (Ly*1e3))


    # %% add lens, etc to wave from data

    wx = 200e-6
    wy = 200e-6
    #u1_xy = circ(X, Y, wx, wy)*tFuncLens(X, Y, wavelength, fx=(1/5.0+1/zz)**-1)*u1_xy
    u1_xy = circ(X, Y, wx, wy)*tFuncLens(X, Y, wavelength, fx=(1/5.0+1/zz)**-1)*u1_xy

    ##=========================================================#
    # %% Plot u1
    ##=========================================================#


    saveFigure = 0

    ## U1
    if saveFigure:
        xo, yo = 0.0, 0.0
    else:
        xo, yo = None, None

    print('WG: Plot u1...')



    factorX, unitStrX = wgt.chooseUnit(X)
    factorY, unitStrY = wgt.chooseUnit(Y)

    unitStrX = unitStrX + ' m'
    unitStrY = unitStrY + ' m'

    # %% U1
    #phase = np.angle(u1_xy)*circ(X, Y, wx, wy)
    #phase = -(np.unwrap(np.unwrap(np.unwrap(np.unwrap(phase), axis=0)), axis=0)/np.pi*
    #        circ(X, Y, wx, wy))

    wgt.plotProfile(X*factorX, Y*factorY, np.abs(u1_xy)**2,
                    r'$x [' + unitStrX + ']$',
                    r'$y [' + unitStrY + ']$',
                    r'Intensity [a.u.]',
                    r'u1_xy',
                    xo=xo, yo=yo,
                    unitX=unitStrX, unitY=unitStrY)
    if saveFigure:
        outputFigureName = wgt.datetimeNowStr() + '_u1.png'
        plt.savefig(outputFigureName)
        print('WG: Figure saved at %s!\n' % (outputFigureName))
        plt.close()
    else:
        plt.show(block=True)
        plt.close()


    print('WG: Plot u1: DONE!')

    #phase = None
    ##=========================================================#
    # %% Propagation
    ##=========================================================#

    print('WG: Propagation...')


#    u2_xy = propTForIR(u1_xy,Lx,Ly,wavelength,zz)
#    titleStr = str(r'propTForIR, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))


#    u2_xy = propIR_RayleighSommerfeld(u1_xy,Lx,Ly,wavelength,zz)
#    titleStr = str(r'propIR_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))
#    u2_xy = propTF_RayleighSommerfeld(u1_xy,Lx,Ly,wavelength,zz)
#    titleStr = str(r'propTF_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))


#    u2_xy, L2 = propFF(u1_xy, Lx, wavelength, zz)
#    titleStr = str(r'propFF, zz=%.3fmm, Intensity [a.u.]'
#                   % (zz*1e3))
#    X,Y = np.meshgrid(np.linspace(-L2/2,L2/2,Mx,endpoint=False),
#                      np.linspace(-L2/2,L2/2,My),endpoint=False)
#    print('WG: L2: %.5gmm' % (L2*1e3))
#    print('WG: X.shape: ', X.shape)
#
#    Lx2 = Lx/1.00
    u2_xy = prop2step(u1_xy, Lx, Lx2, wavelength, zz)
    X, Y = X,Y = np.meshgrid(np.linspace(-Lx/2,Lx/2,Mx,endpoint=False),
                      np.linspace(-Ly/2,Ly/2,My,endpoint=False))
    titleStr = str(r'prop2step, zz=%.3fmm, Intensity [a.u.]'
                   % (zz*1e3))

    print('WG: Power 1: %.5g' % np.sum(np.abs(u1_xy)**2))
    print('WG: Power 2: %.5g' % np.sum(np.abs(u2_xy)**2))
    print('WG: Propagation: DONE!')

    X2, Y2 = X, Y

    del X, Y
    u1_xy = None  # clear var

    ##=========================================================#
    # %% Plot u2
    ##=========================================================#

    print('WG: Plot u2...')

    factorX2, unitStrX2 = wgt.chooseUnit(X2)
    factorY2, unitStrY2 = wgt.chooseUnit(Y2)

    unitStrX2 = unitStrX2 + ' m'
    unitStrY2 = unitStrY2 + ' m'


    if saveFigure:
        xo, yo = 0.0, 0.0
    else:
        xo, yo = None, None

    #phase = np.angle(u2_xy)
    #phase = -(np.unwrap(np.unwrap(np.unwrap(np.unwrap(phase), axis=0)), axis=0)/np.pi*
    #        circ(X, Y, wx, wy))

    wgt.plotProfile(X2*factorX2, Y2*factorY2, np.abs(u2_xy)**2,
                    r'$x [' + unitStrX2 + ']$',
                    r'$y [' + unitStrY2 + ']$',
                    r'Intensity [a.u.]',
                    titleStr,
                    xo=xo, yo=yo,
                    unitX=unitStrX2, unitY=unitStrY2)

    if saveFigure:
        outputFigureName = wgt.datetimeNowStr() + '_u2.png'
        plt.savefig(outputFigureName)
        print('WG: Figure saved at %s!\n' % (outputFigureName))
        plt.close()
    else:
        plt.show(block=False)

    print('WG: Plot u2: DONE!')
    # %%

if __name__ == '__main__':
    main()
