# -*- coding: utf-8 -*-  #
"""
Created on Mon Sep 12 16:40:44 2016

@author: grizolli
"""


import numpy as np

from numpy.fft import fft, ifft


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import wavepy.utils as wpu


# =============================================================================
# %% parameters
# =============================================================================

rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias


lim = 2.0 # *np.pi
xVec = np.linspace(-lim, lim, 1001)
delx = xVec[1] - xVec[0]

# %%
#===============================================================================
# diferential functions
#===============================================================================

#func =  np.exp(np.cos(xVec*5))


#func =  -xVec**2/2


#func = xVec


#func =  -xVec**3/30 - 5*xVec**2


#func =  np.cumsum(np.exp(-xVec**2/.15))*delx


func =  (xVec - np.pi)**2/2 + 1


#func =  np.cos(xVec*50)**2


#func = np.sinc(xVec*2.5)**2 + 0.05*np.sin(200*xVec)

#func = 5*np.sin(xVec) + 2*np.sin(xVec*3) + np.sin(xVec*4) + .5*np.sin(xVec**2*100)


#==============================================================================
# Derivative
#==============================================================================


del_func_1D = np.diff(func)/(delx)  # np.exp(-xVec**2/50)
del_func_1D = np.concatenate(([0.0],del_func_1D))

# the diff result has size n-1, so I add an element. Note that this may results
# in a lateral shift. To have this correct we need to define whether the n
# elelent of diff correspond to the position x_n, x_n+1 or x_{n/2}


# %%
#===============================================================================
# integration 1D
#===============================================================================





def fourier_integration1DShift(del_f_del_x, xvec):

    fx = np.fft.fftfreq(xvec.size,xvec[1] - xvec[0])

    fo = np.abs(fx[1]/2) # shift value

    phaseShift = np.exp(1j*2*np.pi*fo*xvec)  # exp factor for shift
    mult_factor = 1/(2*1j*np.pi*(fx - fo))


    bigGprime = fft(del_f_del_x*phaseShift)
    bigG = bigGprime*mult_factor

    func_g = ifft(bigG) /phaseShift

    func_g -= func_g[0]  # since the integral have and undefined constant,
                         # here it is applied an arbritary offset

    return func_g

# %%

result = fourier_integration1DShift(del_func_1D, xVec)


integrated_1d_abs = np.abs(result)
integrated_1d_r = np.real(result)
integrated_1d_i = np.imag(result)

integrated_1d = integrated_1d_r

# %% Plot integrals

plt.figure()
plt.plot(xVec, func , '-ko', label='f')
plt.plot(xVec-0.5*delx, integrated_1d, '-bo', label='f from FourierIntegral')

offset = integrated_1d + np.mean(func - integrated_1d)
plt.plot(xVec-0.5*delx, offset, '-rx', label='offset f')
# the 0.5*delx accounts for the different ways that you can define the discrete
# differential function. I did't check it formally for each case, but my
# uderstanding is that it should be either ±delx or ±delx/2.

plt.legend(loc='best')

plt.show()


# %% Calculate the derivative from the result



plt.figure()

plt.plot(xVec, del_func_1D,
         '-gx', markersize=10, label='derivative f')
plt.plot(xVec[0:-1]+delx, np.diff(func)/delx,
         '-ko', label='numerical derivative f')

plt.legend(loc='best')
plt.show()


# %% DUMP


#==============================================================================
# def shift(xs, n):
#     e = np.empty_like(xs)
#     e[:-n] = xs[n:]
#     e[-n:] = xs[:n]
#     return e
#
# def fourier_integration1D(del_f_del_x, xvec):
#
#     fx = np.fft.fftfreq(xvec.size,xvec[1] - xvec[0])
#     fx[0] = NAN
#     mult_factor = np.where(np.isfinite(fx), 1/(2*np.pi*fx), 1e200)
#     func_g = -1j*fft(del_f_del_x)*mult_factor
#     return func_g, ifft(func_g)
#
#
#
# def fourier_integration1DShiftOLD(del_f_del_x, xvec):
#
#     fx = np.fft.fftfreq(xvec.size,xvec[1] - xvec[0])
#     fshift = np.abs(fx[1]/10)
#     fx += fshift
#
#     mult_factor = 1/(2*np.pi*fx)
#     func_g = -1j*fft(del_f_del_x)*mult_factor
#
# #    func_g[0] = 1/np.sum(np.real(func_g))
#
#     print('fx[0]: ' + np.str(fx[0]))
#
#     res = ifft(func_g)*np.exp(1j*2*np.pi*xVec*fshift)
#
#     res -= res[0]
#
#     return func_g, res
#
#==============================================================================



