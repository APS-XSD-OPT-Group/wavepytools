# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar 17 11:20:58 2015

@author: wcgrizolli
"""

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
import sys

import itertools


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt
from myFourierLib import gaussianBeam

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

time_i = wgt.timeNowStr()
print 'TIME NOW: ' + time_i
if rank == 0:
    wgt.output2logfile('/home/wcgrizolli/pythonWorkspace/optics/fk_integral/log/log_' + wgt.datetimeNowStr() + '.log')

print("Hello, World! I am process %d of %d on %s." % (rank, size, name))


#==============================================================================
# %% auxiliar functions
#==============================================================================


def circ(wy, wz, y_vec, z_vec):  # circular

    Y, Z = np.meshgrid(y_vec, z_vec)
    out = Y*0.0
    out[abs((Y/wy)**2 + (Z/wz)**2) < 0.5**2] = 1.0
    out[abs((Y/wy)**2 + (Z/wz)**2) == 0.5**2] = .50
    return out


def paraboloidSurface(w, l, R, rho):
    return w**2/R + l**2/rho


#==============================================================================
# %% Some definitions
#==============================================================================

wavelength = 1.239842e-6/1e3
r = 10.00


#ny1, nz1 = 11, 11
#ny2, nz2 = 51, 51
#nl, nw = 21, 21

ny1, nz1 = 31, 31
ny2, nz2 = 21, 21

if rank == 0:
    wgt.color_print('WG: Total number of points: %.4g' % (ny1*nz1*ny2*nz2))
    print('ny1, nz1: %d, %d' % (ny1, nz1))
    print('ny2, nz2: %d, %d' % (ny2, nz2))
#wgt.wait_keyboard()

#==============================================================================
# %% U1
#==============================================================================

Ly1, Lz1 = 1e-3, 1e-3

y1_vec = np.mgrid[-Ly1/2:Ly1/2:ny1*1j]
z1_vec = np.mgrid[-Lz1/2:Lz1/2:nz1*1j]


dy1 = y1_vec[1] - y1_vec[0]
dz1 = z1_vec[1] - z1_vec[0]

#u1_yz = circ(.5e-3, .4e-3, y1_vec, z1_vec)

# %%
u1_yz = gaussianBeam(.2e-3, wavelength, z=5.0, L=Ly1, npoints=ny1)

Y1, Z1 = np.meshgrid(y1_vec, z1_vec)
factorY, unitStrY = wgt.chooseUnit(Y1)
factorZ, unitStrZ = wgt.chooseUnit(Z1)

unitStrY = unitStrY + ' m'
unitStrZ = unitStrZ + ' m'

wgt.plotProfile(Y1*factorZ, Z1*factorY, np.angle(u1_yz),
                r'$y [' + unitStrY +']$',
                r'$z [' + unitStrZ + ']$',
                r'Intensity [a.u.]',
                r'u2_yz')

plt.show()
#==============================================================================
# %% U2
#==============================================================================

Ly2, Lz2 = 1e-3, 1e-3

y2_vec = np.mgrid[-Ly2/2:Ly2/2:ny2*1j]
z2_vec = np.mgrid[-Lz2/2:Lz2/2:nz2*1j]

Y2, Z2 = np.meshgrid(y2_vec, z2_vec)

if rank == 0:
    Y2_chunk_list = np.array_split(Y2, size)
    Z2_chunk_list = np.array_split(Z2, size)
else:
    Y2_chunk_list = None
    Z2_chunk_list = None
#    u2_yz = None
#    chunks = None


Y2_chunk = comm.scatter(Y2_chunk_list, root=0)
Z2_chunk = comm.scatter(Z2_chunk_list, root=0)

Y2_chunk = np.asarray(Y2_chunk)
Z2_chunk = np.asarray(Z2_chunk)


#==============================================================================
# %% Integration
#==============================================================================

print("WG: Integration process %d of %d..."  % (rank, size))

# defining function to multitask loop


def func4map(y2, z2):
    u2 = 0.0 + 0*1j
#    for i1 in range(len(y1_vec)):
#    print('################ y2: ' + str(y2*1e3) + 'mm' )
#    for j1 in range(len(z1_vec)):
#    print('%%%%%%%%% z2: ' + str(z2*1e3) + 'mm')

    for j1 in range(len(y1_vec)):
        for i1 in range(len(z1_vec)):

            PL = np.sqrt((y1_vec[j1]-y2)**2 +
                         (z1_vec[i1]-z2)**2 + r**2)

            gFunc = np.sum((np.exp(1j*2*np.pi/wavelength*PL)/PL**2*dy1*dz1))
            u2 += gFunc*u1_yz[i1, j1]*dy1*dz1

#            ## temp
#            from mpl_toolkits.mplot3d import Axes3D
#            import matplotlib.cm as cm
#
#            fig = plt.figure(figsize=(10., 7.))
#            ax = Axes3D(fig)
#
#            PLoo = PL[np.logical_and(w == 0.0, l == 0.0)]
#
#            surf = ax.plot_surface(l, w, PL - PLoo,
#                                   linewidth=0.0, rstride=1, cstride=1,
#                                   cmap=cm.jet, shade=True)
#
#
#            plt.xlabel(r'$l$')
#            plt.ylabel(r'$w$')
#
##            fig.colorbar(surf, shrink=0.5, aspect=8)
#            #plt.title('Intensity [ph/s/.1%bw/mm^2]', weight='bold')
#
#            plt.show(block=False)
#            plt.figure()
#            PL_w = PL[w == 0.0]
#            plt.plot(l[:,0], PL_w - PLoo, '.-')
#            plt.show(block=True)


    return u2


u2_yz_chunk = Y2_chunk*(0.0 + 0.0*1j)

if Y2_chunk.size != 0:  # this avoid problems due to empty chunks
    sizeChunck1, sizeChunck2 = Y2_chunk.shape
else:
    sizeChunck1, sizeChunck2 = 0, 0

for i, j in itertools.product(range(sizeChunck1), range(sizeChunck2)):

    u2_yz_chunk[i, j] = func4map(Y2_chunk[i, j], Z2_chunk[i, j])


print("WG: Integration process %d of %d... DONE!!!" % (rank, size))

u2_yz_chynk_list = comm.gather(u2_yz_chunk, root=0)

if rank == 0:           # because the arrays have different lenghts I need
                        # to loop in the list and concatane to a single array

    u2_yz = u2_yz_chynk_list[0]

    for i in xrange(1, len(u2_yz_chynk_list)):
        if len(u2_yz_chynk_list[i]) == 0: continue  # this avoid problems due to empty chunks
        u2_yz = np.concatenate((u2_yz, u2_yz_chynk_list[i]), axis=0)


    #==============================================================================
    # %% save for external use
    #==============================================================================

    filename = 'u2_fk_' + wgt.datetimeNowStr() + '_mpi'
    np.savez(filename, u2_yz=u2_yz, Y=Y2, Z=Z2)
    print('\nWG: File saved!! Filename: ' + filename + '.npz')

    print 'Starting time : ' + time_i
    print 'Finishing time: ' + wgt.timeNowStr()

    #==============================================================================
    # %% Plot
    #==============================================================================

    plt.contourf(Y2*1e3, Z2*1e3, np.abs(u2_yz), 256)
    plt.title(r'$N_1 \times N_2: %d^2 \times %d^2$' % (ny1, ny2))


    plt.savefig(filename + '.png')
    plt.show()
    print('WG: Image saved!! Filename: ' + filename + '.png')


    print("Bye!")











