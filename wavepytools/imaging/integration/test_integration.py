# -*- coding: utf-8 -*-  #
"""
Created on Mon Sep 12 16:40:44 2016

@author: grizolli
"""


import numpy as np
import matplotlib.pyplot as plt
import wavepy.utils as wpu





# =============================================================================
# %% parameters
# =============================================================================

size = (501,491)
pixelsize = .0093085684600
xx, yy = wpu.realcoordmatrix(size[1], pixelsize, size[0], pixelsize)



#===============================================================================
# %% Functions
#===============================================================================

parab_waved =  -xx**2/.2 -yy**2/.3 + 0.5*np.cos(100*np.sqrt(xx**2+yy**2))

fermidirac = 1.0/(1 + np.exp((np.sqrt(xx**2+yy**2/.5)-1.1)/.1))

xo = -2
yo = -3.54356
trigfunc = 5*np.sin((xx-xo)+(yy-yo)) + 2*np.sin((xx-xo)*6*(yy-yo)) + \
            np.sin((xx-xo)*5) + .5*np.sin((xx-xo)**2*5)

# Uncomment one of the functions below

#func = parab_waved
#func = fermidirac
func = trigfunc

noise = False  # noise will be added to the differential functions


#==============================================================================
# %% Derivatives
#==============================================================================



del_func_2d_x = np.diff(func, axis=1)
del_func_2d_x = np.pad(del_func_2d_x,((0,0),(1,0)), 'edge')

del_func_2d_y = np.diff(func, axis=0)
del_func_2d_y = np.pad(del_func_2d_y,((1,0),(0,0)), 'edge')

if noise is True:
    del_func_2d_x += .01*(2*np.random.rand(size[0],size[1]) -1)
    del_func_2d_y += .01*(2*np.random.rand(size[0],size[1]) -1)




#==============================================================================
# %% Integration
#==============================================================================


from wavepy.surface_from_grad import frankotchellappa, error_integration

result = frankotchellappa(del_func_2d_x, del_func_2d_y,
                                 reflec_pad=True)

integrated_2d_mod = np.abs(result)
integrated_2d_r = np.real(result)
integrated_2d_i = np.imag(result)

integrated_2d = integrated_2d_r




xx, yy = wpu.realcoordmatrix(func.shape[1], pixelsize, func.shape[0], pixelsize)

#==============================================================================
# %% Plot derivatives
#==============================================================================

midleX = xx.shape[1]//2
midleY = yy.shape[1]//2

plt.figure()
plt.plot(xx[midleX,:], del_func_2d_x[midleX,:], '-kx',
         markersize=10, label='delx f')

plt.plot(yy[:, midleY], del_func_2d_y[:,midleY], '-rx',
         markersize=10, label='dely f')


plt.title('Derivative Functions')

plt.legend()

plt.show()

#==============================================================================
# %% Plot integrated 2d
#==============================================================================

plt.figure()
plt.contourf(xx, yy, func, 101 )
plt.title('Test Func')
plt.colorbar()

plt.show()


plt.figure()
plt.contourf(xx, yy, integrated_2d, 101 )
plt.title('Integration Result')
plt.colorbar()
plt.show()

#==============================================================================
# %% Plot profiles and check periodicity
#==============================================================================

plt.figure()
plt.plot(xx[midleX,:], func[midleX,:], '-kx', markersize=10, label='f x')

plt.plot(xx[midleX,:]-pixelsize/2 + xx[midleX,-1]-xx[midleX,0], func[midleX,:],
         '-gx', markersize=10, label='f x')


shifted = integrated_2d[midleX,:] + np.mean(func[midleX,:]-integrated_2d[midleX,:])
plt.plot(xx[midleX,:]-pixelsize/2, shifted, '-ro', label='integrated shifted f x')
plt.plot(xx[midleX,:]-pixelsize/2 + xx[midleX,-1]-xx[midleX,0], shifted, '-bo',
         label='integrated shifted f x')

plt.title('Test Funtion and Integration Result')
plt.legend()
plt.show()


#==============================================================================
# %% Plot profiles x and y
#==============================================================================

fig = plt.figure(figsize=(14, 7))


plt.subplot(121)
plt.plot(xx[midleX,:], func[midleX,:], '-kx', markersize=10, label='f x')


shifted = integrated_2d[midleX,:]  + np.mean(func[midleX,:]-integrated_2d[midleX,:])
plt.plot(xx[midleX,:]-pixelsize/2, shifted, '-ro', label='integrated shifted f x')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'Profile $x$')


plt.subplot(122)
plt.plot(yy[:,midleY], func[:,midleY], '-kx', markersize=10, label='f y')


shifted = integrated_2d[:,midleY] + np.mean(func[:,midleY]-integrated_2d[:,midleY])
plt.plot(yy[:,midleY]-pixelsize/2, shifted, '-ro', label='intemidleXgrated shifted f y')
plt.xlabel(r'$y$')
plt.legend()
plt.title(r'Profile $y$')

plt.show(block=True)


# %% Plot Integration Qaulity


errorx, errory = error_integration(del_func_2d_x, del_func_2d_y, integrated_2d, pixelsize, plot_flag=True)

# %%

#exit()


# %%
foo = del_func_2d_y[midleX,:]

plt.figure()
plt.plot(xx[midleX,:], foo, '-kx', markersize=10)


plt.plot(xx[midleX,:]+pixelsize/2, wpu.shift_subpixel_1d(foo, 2), '-r.', markersize=10)
plt.plot(xx[midleX,:]+pixelsize/4, wpu.shift_subpixel_1d(foo, 4), '-b.', markersize=10)


plt.show(block=True)



# %%
foo = del_func_2d_y

oversampling = 2



# %%
bar1 = wpu.ourier_spline_2d_axis(foo,oversampling, axis=0)
bar = wpu.fourier_spline_2d_axis(bar1, oversampling, axis=1)


# %%


# %%
bar =  np.real(bar)


# %%

yy2 = wpu.fourier_spline_2d(yy, n=oversampling)


# %%
plt.figure()
plt.plot(yy[:,0], foo[:,midleX//2], '-kx', markersize=10)


plt.plot(yy2[:,0], bar[:,midleX//2*oversampling], '-r.', markersize=10)
#

plt.show()





