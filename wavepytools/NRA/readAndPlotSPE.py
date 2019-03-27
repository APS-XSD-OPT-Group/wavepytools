#! /bin/python
# -*- coding: utf-8 -*-  #
"""
Created on Tue Jun 14 10:07:59 2016

@author: wcgrizolli
"""

from pywinspec import SpeFile, test_headers


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import sys

sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools/')
import wgTools as wgt



figCount = 0
filename = wgt.selectFile('*SPE', 3)[:-4]
img = SpeFile(filename + '.SPE')

img1 = img.data[0]
X1, Y1 = np.meshgrid(np.linspace(-25., 25., img1.shape[0]),
                   np.linspace(-25., 25., img1.shape[1]))


# %%
#fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))
#
#
#plt.contourf(X1, Y1, img1, 51, cmap=cm.binary)
##plt.imshow(img1)
#
#plt.colorbar()
#figCount += 1
#plt.savefig(filename + '_fig_' + str(figCount) + '.png')
#plt.show(block=True)


#X2, Y2, img2 = wgt.selectROI(X1*1e6, Y1*1e6, img1,
#                             arg4graph={'nbins': 51, 'cmap':cm.binary})

idx4cropX1, idx4cropX2, idx4cropY1, idx4cropY2 = 1011, 1630, 634, 1274
X2 = wgt.cropMatrixAtIndexes(X1, [idx4cropY1,
                                    idx4cropY2,
                                    idx4cropX1,
                                    idx4cropX2])
Y2 = wgt.cropMatrixAtIndexes(Y1, [idx4cropY1,
                                    idx4cropY2,
                                    idx4cropX1,
                                    idx4cropX2])
img2 = wgt.cropMatrixAtIndexes(img1, [idx4cropY1,
                                    idx4cropY2,
                                    idx4cropX1,
                                    idx4cropX2])



# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))


plt.contourf(X2, Y2, img2, 101, cmap=cm.Spectral)

plt.colorbar()

figCount += 1
plt.savefig(filename + '_fig_' + str(figCount) + '.png')
plt.show(block=True)


# %% FFT part


from numpy.fft import *


(My,Mx)=np.shape(img2)    #get input field array size
Lx = X2[0,0] - X2[0,-1]
Ly = Y2[0,0] - Y2[-1,0]
dx = X2[0,1] - X2[0,0]   #sample interval
dy = Y2[1,0] - Y2[0,0]    #sample interval
k = 2*np.pi/1e-10 #wavenumber


fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Mx)
fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,My)     #freq coords
[FX,FY]=np.meshgrid(fx,fy)

fftimg2 = ifftshift(fft2(fftshift(img2)))*dx*dy

# %%
intensityF = np.abs(fftimg2)
result = intensityF - np.min(intensityF)


result = result/np.max(result)

# %%
mask = wgt.nanMaskThreshold(result, threshold=.015j)

# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))


plt.contourf(FX, FY,  result*mask, 256)

plt.colorbar()

figCount += 1
#plt.savefig(filename + '_fig_' + str(figCount) + '.png')
plt.show(block=True)

# %%

#mask45 = np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < 45 + 5,
#                        np.arctan2(FY,FX)*np.rad2deg(1) > 45 - 5)
#mask135 = np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < 135 + 5,
#                        np.arctan2(FY,FX)*np.rad2deg(1) > 135 - 5)
#maskMinus45 = np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < -45 + 5,
#                        np.arctan2(FY,FX)*np.rad2deg(1) > -45 - 5)
#maskMinus135 = np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < -135 + 5,
#                        np.arctan2(FY,FX)*np.rad2deg(1) > -135 - 5)

mask_angle = 1.0*(np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < -20. + 5,
                        np.arctan2(FY,FX)*np.rad2deg(1) > -20. - 5) +
             np.logical_and(np.arctan2(FY,FX)*np.rad2deg(1) < 160. + 5,
                        np.arctan2(FY,FX)*np.rad2deg(1) > 160. - 5))

idx_angle_x, idx_angle_y = np.where(mask_angle>.5)





# %%
import scipy.signal

mask2 = wgt.nanMaskThreshold(result, threshold=.001)



idx_x_axis_0, idx_y_axis_0 = scipy.signal.argrelmax(result*mask2, axis=0, order = 3)
idx_x_axis_1, idx_y_axis_1 = scipy.signal.argrelmax(result*mask2, axis=1, order = 3)

foo1 = np.zeros(np.shape(mask2))
foo1[idx_x_axis_0[:], idx_y_axis_0[:]] = 1.0
foo2 = np.zeros(np.shape(mask2))
foo2[idx_x_axis_1[:], idx_y_axis_1[:]] = 1.0


foo3 = foo1*foo2*mask_angle
idx_x, idx_y = np.where(foo3>.5)

# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))

plt.plot(result*mask2, 'x-')
plt.plot(idx_x, result[idx_x, idx_y], 'ro')
plt.show(block=True)


# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))

plt.plot(FX[idx_x, idx_y]*1e-10,FY[idx_x, idx_y]*1e-10, 'ro', alpha=.5, ms=10)
plt.show(block=True)

# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))

plt.plot(FX[idx_angle_x,idx_angle_y]*1e-10,FY[idx_angle_x,idx_angle_y]*1e-10, 'ro', alpha=.5, ms=10)
plt.show(block=True)


# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))

plt.contourf(FX*1e-10, FY*1e-10, result*mask, 51)
plt.plot(FX[idx_x, idx_y]*1e-10,FY[idx_x, idx_y]*1e-10,
         'ko', mew=2, mfc="None", ms=5 )



plt.colorbar()

plt.show(block=True)


# %%



plt.figure(figsize=plt.figaspect(.75), facecolor="white")
plt.hist(np.arctan2(FY[idx_x,idx_y], FX[idx_x,idx_y])*np.rad2deg(1), 600)

plt.show(block=False)





# %%
fig = plt.figure(facecolor="white", figsize=plt.figaspect(.6))

#plt.plot(result, 'x-')
plt.plot(FX[idx_x, idx_y]*1e-10, result[idx_x, idx_y], '-ro')
plt.show(block=True)







