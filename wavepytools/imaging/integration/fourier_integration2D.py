# -*- coding: utf-8 -*-  #
"""
Created on Mon Sep 12 16:40:44 2016

@author: grizolli
"""


import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import wavepy.utils as wpu


# =============================================================================
# %% parameters
# =============================================================================

rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias



size = (601,501)

pixelsize = .0100



xx, yy = wpu.realcoordmatrix(size[1], pixelsize, size[0], pixelsize)



delx = pixelsize
dely = pixelsize


# %%
#===============================================================================
# diferential functions
#===============================================================================

#func =  np.exp(np.cos(xx*5) )

#func =  -xx**2/.2 -yy**2/.3 + 0.5*np.cos(100*np.sqrt(xx**2+yy**2))


#func = xx**3




#func =  -xx/30 + .1##- yy/50


#func = np.where( xx**2/.15 + yy**2/.05 < 10, 1.0, 0.0)

#func = 1.0/(1 + np.exp((np.sqrt(xx**2+yy**2/.5)-1.1)/.01)) # + .05 + .1*np.random.rand(size[0],size[1]) # + 0.05*np.cos(100*np.sqrt(xx**2+yy**2))

#

#del_func_2d_x = np.exp(- xx**2/.15 - yy**2/.15)*(- 2*xx/.15)
#del_func_2d_y = np.exp(- xx**2/.15 - yy**2/.15)*(- 2*yy/.15)
#func =  np.cumsum(np.exp(- xx**2/.15 - yy**2/.15),axis=0)


#func =  (xx - np.pi)**2/2 + 1000


#func =  np.cos(xx*10 + yy*15)**2


#func =  -xx**2/2 -yy**2/4


#func = np.sinc(xx*1)**2*np.sinc(yy*2)**2 + 0.05*np.cos(50*np.sqrt(xx**2+yy**2))

#func = 5*np.sin(xx+yy) + 2*np.sin(xx*3*yy) + np.sin(xx*4) + .5*np.sin(xx**2*50)


func = 5*np.sin(xx+yy) + 2*np.sin(xx*3*yy) + np.sin(xx*4) + .5*np.sin(xx**2*50)




#padEdge = 100
#func = np.pad(func[padEdge:-padEdge,padEdge:-padEdge],
#              padEdge, 'edge')


#
#func = np.pad(func, padEdge,
#              'constant', constant_values=(0.0,) )
#
#func = np.pad(func,padEdge, 'linear_ramp')








#==============================================================================
# Padding
#==============================================================================



del_func_2d_x = np.diff(func, axis=1)/pixelsize

del_func_2d_x = np.pad(del_func_2d_x,((0,0),(1,0)), 'edge')

del_func_2d_y = np.diff(func, axis=0)/pixelsize

del_func_2d_y = np.pad(del_func_2d_y,((1,0),(0,0)), 'edge')


del_func_2d_xy =  np.diff(del_func_2d_x, axis=0)/dely
del_func_2d_yx =  np.diff(del_func_2d_y, axis=1)/delx


# strange padding



func = np.pad(func,((0,func.shape[0]),(0,func.shape[1])),'reflect')



def reflec_pad_grad_fields(del_func_x, del_func_y):

    del_func_x_c1 =  np.concatenate((del_func_x,
                                     del_func_x[::-1,:]),axis=0)

    del_func_x_c2 =  np.concatenate((-del_func_x[:,::-1],
                                     -del_func_x[::-1,::-1]),axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)


    del_func_y_c1 =  np.concatenate((del_func_y,
                                     -del_func_y[::-1,:]),axis=0)

    del_func_y_c2 =  np.concatenate((del_func_y[:,::-1],
                                     -del_func_y[::-1,::-1]),axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)


    return del_func_x, del_func_y


foo, bar = reflec_pad_grad_fields(wpu.dummy_images('Shapes', noise = 1),
                                  wpu.dummy_images('Shapes', noise = 1))
# %%
#plt.figure()
#plt.imshow(foo)
##plt.imshow(bar)
#plt.colorbar()
#plt.show()

# %%


del_func_2d_x, del_func_2d_y = reflec_pad_grad_fields(del_func_2d_x,
                                                      del_func_2d_y)



#
#padEdge = 100
#del_func_2d_y = np.pad(del_func_2d_x,padEdge, 'edge')
#del_func_2d_y = np.pad(del_func_2d_y,padEdge, 'edge')
#func = np.pad(func, padEdge, 'edge')




#padEdge = 5
#del_func_2d_x = np.pad(del_func_2d_x,padEdge, 'linear_ramp')
#del_func_2d_y = np.pad(del_func_2d_y,padEdge, 'linear_ramp')
#
#
#padEdge = 100
#del_func_2d_x = np.pad(del_func_2d_x,padEdge,
#              'constant', constant_values=(0.0) )
#del_func_2d_y = np.pad(del_func_2d_y,padEdge,
#              'constant', constant_values=(0.0) )
#
#
##padEdge = 100
#func = np.pad(func, padEdge,
#              'constant', constant_values=(0.0,) )



# %%
#===============================================================================
# integration 2d
#===============================================================================

def fourier_integration2DShift(del_f_del_x, delx,
                               del_f_del_y, dely):

    '''
    This function is the direct use of the CONTINOUS formulation of
    Frankot-Chellappa, eq 21 in the article:

    T. Frankot and R. Chellappa
        A Method for Enforcing Integrability in Shape from Shading Algorithms,
        IEEE Transactions On Pattern Analysis And Machine Intelligence, Vol 10,
        No 4, Jul 1988

    In addition, it uses the CONTINOUS shift property to avoid singularities
    at zero frequencies

    '''

    fx, fy = np.meshgrid(np.fft.fftfreq(del_f_del_x.shape[1], delx),
                         np.fft.fftfreq(del_f_del_x.shape[0], dely))

    xx, yy = wpu.realcoordmatrix(del_f_del_x.shape[1], delx,
                                 del_f_del_x.shape[0], dely)

    fo_x = - np.abs(fx[0,1]/15) # shift fx value
    fo_y = - np.abs(fy[1,0]/15) # shift fy value


    phaseShift = np.exp(2*np.pi*1j*(fo_x*xx + fo_y*yy))  # exp factor for shift

    mult_factor = 1/(2*np.pi*1j)/(fx - fo_x - 1j*fy + 1j*fo_y )


    bigGprime = fft2((del_f_del_x - 1j*del_f_del_y)*phaseShift)
    bigG = bigGprime*mult_factor

    func_g = ifft2(bigG) / phaseShift

    func_g -= np.min(np.real(func_g))  # since the integral have and undefined constant,
                         # here it is applied an arbritary offset


    return func_g





def fourier_integrationDiscrete(del_f_del_x, del_f_del_y):

    '''
    This function is the direct use of the DISCRETE formulation of
    Frankot-Chellappa, eq 22 in the article.

    It has a difference though: the differential uses point at n+1 and n-1.
    I am using n and n+1

    In addition, it uses the DISCRETE shift property to avoid singularities
    at zero frequencies

    '''

    j2pi = 2 * np.pi * 1j

    MM = del_f_del_x.shape[1]
    NN = del_f_del_x.shape[0]



    qq, ll = np.meshgrid(np.fft.fftfreq(MM)*MM,
                         np.fft.fftfreq(NN)*NN, indexing='xy')

#    coef_delx_n = np.conjugate(np.exp(j2pi*2*ll/NN) - 1.0)
#    coef_dely_m = np.conjugate(np.exp(j2pi*2*qq/MM) - 1.0)

#    coef_delx_n = -1j*np.sin(2* np.pi *ll/NN)
#    coef_dely_m = -1j*np.sin(2* np.pi *qq/MM)


    coef_delx_n = -j2pi*ll/NN
    coef_dely_m = -j2pi*qq/MM

    numerator = coef_delx_n*fft2(del_f_del_x) + coef_dely_m*fft2(del_f_del_y)

    denominator = np.abs(coef_delx_n)**2 + np.abs(coef_dely_m)**2 + np.finfo(float).eps


    res = ifft2(numerator/denominator)

    return res - np.mean(np.real(res))


def frankotchellappa(del_f_del_x,del_f_del_y):

    # Frankt-Chellappa Algrotihm
    # Input gx and gy
    # Output : reconstruction
    # Author: Amit Agrawal, 2005

    NN, MM = del_f_del_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                         np.fft.fftfreq(NN)*2*np.pi, indexing='xy')



    numerator = -1j*wx*fft2(del_f_del_x) -1j*wy*fft2(del_f_del_y)

    denominator = (wx)**2 + (wy)**2 + np.finfo(float).eps



    div = numerator/denominator


    res = ifft2(div)

    res -= np.mean(np.real(res))

    return res

# %%
#


result1 = fourier_integration2DShift(del_func_2d_x, pixelsize, del_func_2d_y, pixelsize)



#result1 = fourier_integrationDiscrete(del_func_2d_x*pixelsize, del_func_2d_y*pixelsize)


result2 = frankotchellappa(del_func_2d_x*pixelsize, del_func_2d_y*pixelsize)

#padEdge *= 2
#
#del_func_2d_x = wpu.crop_matrix_at_indexes(del_func_2d_x, [padEdge,-padEdge, padEdge,-padEdge])
#
#del_func_2d_y = wpu.crop_matrix_at_indexes(del_func_2d_y, [padEdge,-padEdge, padEdge,-padEdge])
#
#result1 = wpu.crop_matrix_at_indexes(result1, [padEdge,-padEdge, padEdge,-padEdge])
#result2 = wpu.crop_matrix_at_indexes(result2, [padEdge,-padEdge, padEdge,-padEdge])
#
#func = wpu.crop_matrix_at_indexes(func, [padEdge,-padEdge, padEdge,-padEdge])

integrated_2d_mod1 = np.abs(result1)
integrated_2d_r1 = np.real(result1)
integrated_2d_i1 = np.imag(result1)

integrated_2d1 = integrated_2d_r1

integrated_2d2 = np.real(result2)


def oneForthOfArray(array):

    array, _ = np.array_split(array,2,axis=0)
    return np.array_split(array,2,axis=1)[0]


#func = oneForthOfArray(func)
#del_func_2d_x = oneForthOfArray(del_func_2d_x)
#del_func_2d_y = oneForthOfArray(del_func_2d_y)
#integrated_2d1 = oneForthOfArray(integrated_2d1)
#integrated_2d2 = oneForthOfArray(integrated_2d2)


xx, yy = wpu.realcoordmatrix(func.shape[1], pixelsize, func.shape[0], pixelsize)

# %%


midleX = xx.shape[1]//4
midleY = yy.shape[1]//4



# %%

plt.figure()
plt.plot(xx[midleX,:], del_func_2d_x[midleX,:], '-kx', markersize=10, label='delx f')

plt.plot(yy[:, midleY], del_func_2d_y[:,midleY], '-rx', markersize=10, label='dely f')

plt.legend()

plt.show()


# %%
plt.figure()
plt.contourf(xx, yy, func, 101 )
plt.title('Test Func')
plt.colorbar()

plt.show()

# %%
for integrated_2d, method in [[integrated_2d1, 'WG'],[integrated_2d2, 'FC']]:

    plt.figure()
    plt.contourf(xx, yy, integrated_2d, 101 )
    plt.title('Integration ' + method)
    plt.colorbar()
    plt.show()


    plt.figure()
    plt.plot(xx[midleX,:], func[midleX,:], '-kx', markersize=10, label='f x')

    #plt.plot(xx[midleX,:]-delx/2, integrated_2d[midleX,:], '-bo', label='f from FourierIntegral')

    shifted = integrated_2d[midleX,:] + np.mean(func[midleX,:]-integrated_2d[midleX,:])
    plt.plot(xx[midleX,:]-delx/2, shifted, '-ro', label='integrated shifted f x')

    plt.title(method)
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(yy[:,midleY], func[:,midleY], '-kx', markersize=10, label='f y')

    #plt.plot(xx[midleX,:]-delx/2, integrated_2d[midleX,:], '-bo', label='f from FourierIntegral')

    shifted = integrated_2d[:,midleY] + np.mean(func[:,midleY]-integrated_2d[:,midleY])
    plt.plot(yy[:,midleY]-dely/2, shifted, '-ro', label='integrated shifted f y')

    plt.title(method)
    plt.legend()
    plt.show(block=True)



# %%


ampl = np.abs(np.max(func) - np.min(func))/2


plt.figure()

plotThis = func - integrated_2d1
plotThis -= np.mean(plotThis)
plotThis /= ampl

sigmaThisPlot = np.std(plotThis)

plt.contourf(xx, yy, plotThis, 101, cmap='RdGy', vmin=-3*sigmaThisPlot, vmax=3*sigmaThisPlot)
plt.title(r'Diff method WG to Func, $\sigma$ = {:.3f}'.format(sigmaThisPlot))
plt.colorbar()

plt.show(block=True)

# %%
plt.figure()

plotThis = func - integrated_2d2
plotThis -= np.mean(plotThis)
plotThis /= ampl
sigmaThisPlot = np.std(plotThis)

plt.contourf(xx, yy, plotThis, 101, cmap='RdGy', vmin=-2*sigmaThisPlot, vmax=2*sigmaThisPlot)
plt.title('Diff method FC to Func, $\sigma$ = {:.3f}'.format(np.std(plotThis)))
plt.colorbar()

plt.show(block=True)













