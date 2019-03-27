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



size = (501,501)

pixelsize = 1.00



xx, yy = wpu.realcoordmatrix(size[1], pixelsize, size[0], pixelsize)



# %%
#===============================================================================
# diferential functions
#===============================================================================

#func =  np.exp(np.cos(xx*5) )

#func =  -xx**2/.2 -yy**2/.3 + 0.5*np.cos(100*np.sqrt(xx**2+yy**2))


#func = xx**3




#func =  -xx/30 + .1##- yy/50


#func = np.where( xx**2/.15 + yy**2/.05 < 10, 1.0, 0.0)
#
#func = 1.0/(1 + np.exp((np.sqrt(xx**2+yy**2/.25)-0.5)/.1)) + .05 + .1*np.random.rand(size[0],size[1])+ 0.05*np.cos(100*np.sqrt(xx**2+yy**2))
#
#
#func += 1.0/(1 + np.exp((np.sqrt((xx-100*pixelsize)**2/.3+(yy+200*pixelsize)**2)-0.25)/.0001))
#func += 1.0/(1 + np.exp((np.sqrt((xx-30*pixelsize)**2+(yy+100*pixelsize)**2)/.3-0.5)/.01))
#func += wpu.dummy_images('Shapes', (501, 501))
#
#func = np.zeros(size)
#
#
#
#list_vals = [[-150, -150 , .01],
#             [150, -150 , .1],
#             [-150, 150 , 1.],
#             [150, 150 , 5]]
#
#for xo, yo, sigma in list_vals:
#    print(sigma)
#    func += 1.0/(1 + np.exp((np.sqrt((xx-xo)**2+(yy-yo)**2)-50)/sigma)) # 1.0/(1 + np.exp((np.sqrt((xx-xo)**2+(yy-yo)**2)-0.5)/sigma))


#plt.imshow(func)
#
#
#del_func_2d_x = np.exp(- xx**2/.15 - yy**2/.15)*(- 2*xx/.15)
#del_func_2d_y = np.exp(- xx**2/.15 - yy**2/.15)*(- 2*yy/.15)
#func =  np.cumsum(np.exp(- xx**2/.15 - yy**2/.15),axis=0)


#func =  (xx - np.pi)**2/2 + 1000


#func =  np.cos(xx*10 + yy*15)**2


#func =  -xx**2/2 -yy**2/4

#
#func = np.sinc(xx*1)**2*np.sinc(yy*2)**2 + 0.05*np.cos(50*np.sqrt(xx**2+yy**2))

func = 5*np.sin(xx+yy) + 2*np.sin(xx*3*yy) + np.sin(xx*4) + .5*np.sin(xx**2*5)
#func *= 10*(xx-xx[0,0])**2 + 3*(yy-yy[0,0])**2


#padEdge = 100
#func = np.pad(func[padEdge:-padEdge,padEdge:-padEdge],
#              padEdge, 'edge')


#
#func = np.pad(func, padEdge,
#              'constant', constant_values=(0.0,) )
#
#func = np.pad(func,padEdge, 'linear_ramp')



plt.imshow(func)

#==============================================================================
# Derivatives
#==============================================================================



del_func_2d_x = np.diff(func, axis=1)
del_func_2d_x = np.pad(del_func_2d_x,((0,0),(1,0)), 'edge')
#del_func_2d_x +=  5*(2*np.random.rand(size[0],size[1]) -1)

del_func_2d_y = np.diff(func, axis=0)
del_func_2d_y = np.pad(del_func_2d_y,((1,0),(0,0)), 'edge')
#del_func_2d_y += 5*(2*np.random.rand(size[0],size[1]) -1)




# %% Integration


from wavepy.surface_from_grad import frankotchellappa

result = frankotchellappa(del_func_2d_x, del_func_2d_y, reflec_pad=True)

# %%


integrated_2d_mod = np.abs(result)
integrated_2d_r = np.real(result)
integrated_2d_i = np.imag(result)

integrated_2d = integrated_2d_r




xx, yy = wpu.realcoordmatrix(func.shape[1], pixelsize, func.shape[0], pixelsize)

# %%integrated_2d


midleX = xx.shape[1]//2
midleY = yy.shape[1]//2



# %%

plt.figure()
plt.plot(xx[midleX,:], del_func_2d_x[midleX,:], '-kx',
         markersize=10, label='delx f')

plt.plot(yy[:, midleY], del_func_2d_y[:,midleY], '-rx',
         markersize=10, label='dely f')

plt.legend()

plt.show()


# %%
plt.figure()
plt.contourf(xx, yy, func, 101 )
plt.title('Test Func')
plt.colorbar()

plt.show()


plt.figure()
plt.contourf(xx, yy, integrated_2d, 101 )
plt.title('Integration FC')
plt.colorbar()
plt.show()

# %%

plt.figure()
plt.plot(xx[midleX,:], func[midleX,:], '-kx', markersize=10, label='f x')

plt.plot(xx[midleX,:]-pixelsize/2 + xx[midleX,-1]-xx[midleX,0], func[midleX,:],
         '-gx', markersize=10, label='f x')


shifted = integrated_2d[midleX,:] + np.mean(func[midleX,:]-integrated_2d[midleX,:])
plt.plot(xx[midleX,:]-pixelsize/2, shifted, '-ro', label='integrated shifted f x')
plt.plot(xx[midleX,:]-pixelsize/2 + xx[midleX,-1]-xx[midleX,0], shifted, '-bo',
         label='integrated shifted f x')

plt.title('FC')
plt.legend()
plt.show()


# %%
fig = plt.figure(figsize=(14, 10))


plt.subplot(121)
plt.plot(xx[midleX,:], func[midleX,:], '-kx', markersize=10, label='f x')


shifted = integrated_2d[midleX,:]  + np.mean(func[midleX,:]-integrated_2d[midleX,:])
plt.plot(xx[midleX,:]-pixelsize/2, shifted, '-ro', label='integrated shifted f x')

plt.legend()


plt.subplot(122)
plt.plot(yy[:,midleY], func[:,midleY], '-kx', markersize=10, label='f y')


shifted = integrated_2d[:,midleY] + np.mean(func[:,midleY]-integrated_2d[:,midleY])
plt.plot(yy[:,midleY]-pixelsize/2, shifted, '-ro', label='integrated shifted f y')


plt.legend()



plt.show(block=False)

# %%



def _grad(func):

    del_func_2d_x = np.diff(func, axis=1)
    del_func_2d_x = np.pad(del_func_2d_x,((0,0),(1,0)), 'edge')

    del_func_2d_y = np.diff(func, axis=0)
    del_func_2d_y = np.pad(del_func_2d_y,((1,0),(0,0)), 'edge')

    return del_func_2d_x, del_func_2d_y

# %%
def _error(del_f_del_x, del_f_del_y, func, plot_error=False):


    delx, dely = _grad(func)

    del_f_del_x -= np.mean(del_f_del_x)
    del_f_del_y -= np.mean(del_f_del_y)

    delx -= np.mean(delx)
    dely -= np.mean(dely)

    amp_x = np.max(del_f_del_x) - np.min(del_f_del_x)
    amp_y = np.max(del_f_del_y) - np.min(del_f_del_y)


    delta_x = (delx - del_f_del_x)/amp_x*100.

    delta_y = (dely - del_f_del_y)/amp_y*100.


    if plot_error:
        plt.figure()
        plt.imshow(delta_x, cmap='Spectral')
        plt.title('Integration FC x, error: {:.3f}'.format(np.average(np.abs(delta_x))))
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(delta_y, cmap='Spectral')
        plt.title('Integration FC y, error: {:.3f}'.format(np.average(np.abs(delta_y))))
        plt.colorbar()
        plt.show()



        fig = plt.figure(figsize=(14, 5))
        fig.suptitle('Histograms to evaluate data quality', fontsize=16)

        plt.subplot(121)
        plt.hist(delta_x.flatten(), 51)
        plt.title(r'delta_x', fontsize=16)

        plt.subplot(122)
        plt.hist(delta_y.flatten(), 51)
        plt.title(r'delta_y', fontsize=16)

        plt.show(block=True)

    return (np.average(np.abs(delta_x)) + np.average(np.abs(delta_y)))/2


# %%
def _error2(del_f_del_x, del_f_del_y, func, pixelsize, plot_error=False):

    delx, dely = _grad(func)

    del_f_del_x -= np.mean(del_f_del_x)

    delx -= np.mean(delx)
    dely -= np.mean(dely)

    amp_x = np.max(del_f_del_x) - np.min(del_f_del_x)
    amp_y = np.max(del_f_del_y) - np.min(del_f_del_y)


    delta_x = (delx - del_f_del_x) /amp_x*100.

    delta_y = (dely - del_f_del_y)/amp_y*100.


    if plot_error:

        plt.figure(figsize=(14, 10))


        plt.subplot(221)
        plt.plot(xx[midleX,:], del_f_del_x[midleX,:], '-kx',
                 markersize=10, label='dx data')
        plt.plot(xx[midleX,:], delx[midleX,:], '-rx',
                 markersize=10, label='dx reconstructed')
        plt.legend()

        plt.subplot(223)
        #    plt.plot(xx[midleX,:]-pixelsize/2,
        #             np.cumsum(np.abs(delta_x[midleX,:]))/delx.shape[0],
        #             '-g', label='cumulative error x')
        #    plt.legend()
        plt.plot(xx[midleX,:]-pixelsize/2,
                 np.abs(delta_x[midleX,:]),
                 '-g')



        plt.subplot(222)
        plt.plot(yy[:,midleY], del_f_del_y[:,midleY], '-kx',
                 markersize=10, label='dy data')
        plt.plot(yy[:,midleY], dely[:,midleY], '-rx',
                 markersize=10, label='dy reconstructed')


        plt.legend()

        plt.subplot(224)

        #    plt.plot(yy[:,midleY]-pixelsize/2,
        #             np.cumsum(np.abs(delta_y[:,midleY]))/dely.shape[1],
        #             '-g', label='cumulative error y')
        #    plt.legend()

        plt.plot(yy[:,midleY]-pixelsize/2,
                 np.abs(delta_y[:,midleY]),
                 '-g')




        plt.show(block=False)

    return (np.average(np.abs(delta_x)) + np.average(np.abs(delta_y)))/2


# %%

error = _error2(del_func_2d_x, del_func_2d_y, integrated_2d, pixelsize)


# %%
#print('func error {:.3f}'.format( _error(del_func_2d_x, del_func_2d_cumsumy, func, pixelsize)))


# %%

ampl = np.abs(np.max(func) - np.min(func))/2


plt.figure()

plotThis = func - integrated_2d
plotThis -= np.mean(plotThis)
plotThis /= ampl

sigmaThisPlot = np.std(plotThis)

plt.contourf(xx, yy, plotThis, 101, cmap='RdGy', vmin=-3*sigmaThisPlot, vmax=3*sigmaThisPlot)
plt.title(r'Diff method FC to Func, $\sigma$ = {:.3f}, error  = {:.3f} '.format(sigmaThisPlot, error))
plt.colorbar()

plt.show(block=True)
