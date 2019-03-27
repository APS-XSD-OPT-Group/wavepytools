# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar 17 11:20:58 2015

@author: wcgrizolli
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from multiprocessing import Pool, cpu_count
import itertools


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt

from myFourierLib import gaussianBeam

time_i = wgt.timeNowStr()
print 'WG: Time now:' + time_i
wgt.output2logfile('log/log_' + wgt.datetimeNowStr() + '.log')



#==============================================================================
#%% auxiliar functions
#==============================================================================


def circ(wy, wz, y_vec, z_vec):  # circular

    Y, Z = np.meshgrid(y_vec, z_vec)
    out = Y*0.0
    out[abs((Y/wy)**2 + (Z/wz)**2) < 0.5**2] = 1.0
    out[abs((Y/wy)**2 + (Z/wz)**2) == 0.5**2] = .50
    return out



#==============================================================================
#%%  Some definitions
#==============================================================================

wavelength = 1.239842e-6/1e3
r = 10.00

ny1, nz1 = 21, 21
ny2, nz2 = 21, 21

ntasks = 15 # cpu_count()  # XXX:

#if ntasks > cpu_count():
#    wgt.color_print("WG: ntasks bigger than number of CPU's")
#    ntasks = cpu_count() - 1


wgt.color_print('WG: Total number of points: %.4g' % (ny1*nz1*ny2*nz2))
print('ny1, nz1: %d, %d' % (ny1, nz1))
print('ny2, nz2: %d, %d' % (ny2, nz2))
print("WG: Available number of cpu's: %d" % cpu_count())

print("WG: Number of cpu's to be used: %d" % ntasks)
wgt.wait_keyboard()

#==============================================================================
#%%  U1
#==============================================================================

Ly1, Lz1 = 1e-3, 1e-3

y1_vec = np.mgrid[-Ly1/2:Ly1/2:ny1*1j]
z1_vec = np.mgrid[-Lz1/2:Lz1/2:nz1*1j]

dy1 = y1_vec[1] - y1_vec[0]
dz1 = z1_vec[1] - z1_vec[0]

u1_yz = circ(.15e-3, .35e-3, y1_vec, z1_vec)


#u1_yz = gaussianBeam(.4e-3, wavelength, z=0.0, L=Ly1, npoints=ny1)


#==============================================================================
#%%  U2
#==============================================================================

Ly, Lz = 1e-3, 1e-3

y2_vec = np.mgrid[-Ly/2:Ly/2:ny2*1j]
z2_vec = np.mgrid[-Lz/2:Lz/2:nz2*1j]

#Y2, Z2 = np.meshgrid(y2_vec, z2_vec)
u2_yz = np.zeros((nz2, ny2), dtype=complex)


#==============================================================================
#%%  Integration
#==============================================================================

print("WG: Integration...")

# defining function to multitask loop
def func4map((j2, i2)):

    print('.'),
    sys.stdout.flush()

#    print('################ i: ' + str(i2))
#    print('%%%%%%%%% j:' + str(j2))

    for j1 in range(len(y1_vec)):
        for i1 in range(len(z1_vec)):

            PL = np.sqrt((y1_vec[j1]-y2_vec[j2])**2 +
                         (z1_vec[i1]-z2_vec[i2])**2 + r**2)

            gFunc = np.sum((np.exp(1j*2*np.pi/wavelength*PL)/PL**2*dy1*dz1))
            u2_yz[i2, j2] += gFunc*u1_yz[i1, j1]*dy1*dz1

    return u2_yz[i2, j2]
#
#def func4map((j2, i2)):
#
#    print('.'),
#    sys.stdout.flush()
##    u2 = 0.0 + 0*1j
#
##    print('################ i: ' + str(i2))
##    print('%%%%%%%%% j:' + str(j2))
#
#    for j1 in range(len(y1_vec)):
#        for i1 in range(len(z1_vec)):
#
#            PL = np.sqrt((y1_vec[j1]-y2_vec[j2])**2 +
#                         (z1_vec[i1]-z2_vec[i2])**2 + r**2)
#
#            gFunc = np.sum((np.exp(1j*2*np.pi/wavelength*PL)/PL**2*dy1*dz1))
#            u2_yz[i2, j2] += gFunc*u1_yz[i1, j1]*dy1*dz1
##            u2 += gFunc*u1_yz[i1, j1]*dy1*dz1
#
##    return u2
#    return u2_yz[i2, j2]
##    return j2


# %% Multitask loop

print('WG: IT IS COMMON TO OVERLAP THE OUTPUTS OF FUNCTION f(x)\n\n')

p = Pool(ntasks)
u2_yz = p.map(func4map,
            itertools.product(range(len(y2_vec)),range(len(z2_vec))))


#p.map(func4map, itertools.product(range(len(y2_vec)),range(len(z2_vec))))

#print u2_yz
u2_yz = np.array(u2_yz).reshape(ny2,nz2)
u2_yz = u2_yz.T


Y2, Z2 = np.meshgrid(y2_vec, z2_vec)

#==============================================================================
#%%  save for external use
#==============================================================================

filename = 'u2_fk_' + wgt.datetimeNowStr()
np.savez(filename, u2_yz=u2_yz, Y=Y2, Z=Z2)
print('\nWG: File saved!! Filename: ' + filename)
print("Bye!")

#==============================================================================
#%%  Plot
#==============================================================================


plt.contourf(Y2*1e3, Z2*1e3, np.abs(u2_yz), 256)
plt.title(r'$N_1 \times N_2: %d^2\times%d^2$' % (ny1, ny2))

#ny1


plt.show()


print 'WG: Time now:' + wgt.timeNowStr()









