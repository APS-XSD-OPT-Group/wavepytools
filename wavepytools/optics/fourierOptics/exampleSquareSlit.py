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
wavelength = 1e-9
[Lx,Ly] = [1e-3, 1e-3]
# Mx = Lx^2/wavelength/z
[Mx,My] = [2001, 2001]
dx = Lx/Mx
dy = Ly/My


print 'sampling x='  + str(Mx)
print 'sampling y='  + str(My)

# %%
if Mx > 3001 or  My > 3001:
    print('Sampling bigger than 1001^2, stoping the program')
    sys.exit()

##=========================================================#
# %% 2D u1 function
##=========================================================#

# % rectangular slit
def rect(X, Y, wx, wy, Xo=0.0, Yo=0.0):

    out = X*0.0

    out[np.logical_and(abs((X-Xo)/wx) < 0.5, abs((Y-Yo)/wy) < 0.5)] = 1.0
    out[np.logical_and(abs((X-Xo)/wx) == 0.5, abs((Y-Yo)/wy) < 0.5)] = .50
    out[np.logical_and(abs((X-Xo)/wx) < 0.5, abs((Y-Yo)/wy) == 0.5)] = .50

    return out

wx = 100e-6
wy = 100e-6

Nf = 100.0  # Fresnel number
zz = wx**2/wavelength/Nf # dist to propag
print('WG: distance of propagation: %.3gm' % zz)

X,Y = np.meshgrid(np.linspace(-Lx/2,Lx/2,Mx),np.linspace(-Ly/2,Ly/2,My))

u1_xy = rect(X, Y, wx, wy, 0e-6,0e-6)

#u1_xy = rect(X, Y, wx, wy, 0, 80e-6) + circ(X, Y, wx, wy, 0,-80e-6)  # double slit



##=========================================================#
# %% Propagation
##=========================================================#

print('\nWG: proapgating...')
u2_xy = propTForIR(u1_xy,Lx,Ly,wavelength,zz)

print('WG: DONE!')

##=========================================================#
# %% Plot u1
##=========================================================#

factorX, unitStrX = wgt.chooseUnit(X)
factorY, unitStrY = wgt.chooseUnit(Y)

unitStrX = unitStrX + ' m'
unitStrY = unitStrY + ' m'


## U1

#wgt.plotProfile(X*factorX, Y*factorY, np.abs(u1_xy),
#                r'$x [' + unitStrX +']$',
#                r'$y [' + unitStrY + ']$',
#                r'Intensity [a.u.]',
#                xo=0.0, yo=0.0,
#                unitX=unitStrX, unitY=unitStrY)
##=========================================================#
# %% Plot u2
##=========================================================#

data = np.abs(u2_xy)
ny, nx = data.shape

# %% mask


#print('WG: Mask...')
#[i_min, i_max, j_min, j_max] = wgt.indexForSquareMaskThreshold(data, np.max(data)*.005)
#print('WG: DONE!')
#
## crop the matrices based on the mask
#data = data[i_min:i_max, j_min:j_max]
#X = X[i_min:i_max, j_min:j_max]
#Y = Y[i_min:i_max, j_min:j_max]
#ny, nx = data.shape


# %%
wgt.plotProfile(X*factorX, Y*factorY, data,
                r'$x [' + unitStrX +']$',
                r'$y [' + unitStrY + ']$',
                r'Intensity [a.u.]',
                str(r'$z$ = %.3fm, $N_F$ = %.2f' % (zz, Nf)),
                xo=0.0, yo=0.0,
                unitX=unitStrX, unitY=unitStrY)

plt.savefig(('profile_NF_%.2f.png' % Nf))
plt.close()


# %%



fig = plt.figure(figsize=(15, 7.6))
ax = fig.add_subplot(1, 1, 1)
plt.rcParams.update({'font.size': 32})


ax.plot(X[0,:]*factorX, data[int(ny/2),:], lw=2, c='k')

#plt.title(r'$z$=0m', fontsize=28, weight='bold')
plt.title( str(r'$z$ = %.3fm, $N_F$ = %.2f' % (zz, Nf)), fontsize=28, weight='bold')
#plt.legend(('MAXII x', 'MAXII y',
#            'MAXIV 1.5GeV x', 'MAXIV 1.5GeV y',
#            'MAXIV 3GeV x', 'MAXIV 3GeV y'))

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.spines['left'].set_bounds(0, 1.01*np.max(np.abs(data[int(ny/2),:])))
#ax.spines['bottom'].set_bounds(X[0,0]*1.1e6 , X[0,-1]*1.1e6)
#ax.spines['bottom'].set_bounds(-750 , 750)

plt.xlim([-400.000, 400.0])

ax.xaxis.set_label_coords(1.05, 0.10, transform=None)
ax.yaxis.set_label_coords(.50, .95, transform=None)

plt.xticks( rotation=90)
ax.xaxis.set_tick_params(width=3, length=10)
plt.yticks([])

#ax.set_yticks([])

#plt.xlabel(r'$x\times10^{-6}$')
#plt.ylabel(r'$y(x)$', rotation=0)
plt.xlabel(r'$x [' + unitStrX +']$', fontsize=28, rotation=90)
#plt.ylabel(r'Intensity', fontsize=19, rotation=0)

plt.savefig(('demo_NF_%.2f.svg' % Nf), transparent=True)

plt.show(block=False)






