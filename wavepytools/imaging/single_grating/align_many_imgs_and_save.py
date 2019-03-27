#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

'''
Author: Walan Grizolli



'''
# %%

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wavepy.utils as wpu
from wavepy.utils import easyqt
import numpy as np

import dxchange

import sys, glob, os

# %%

def align_many_imgs_linearshifts(samplefileName,
                                 totalShift=None,
                                 displayPlots=False):

    if displayPlots:
        plt.ion()
    else:
        plt.ioff()

    fextension = samplefileName.rsplit('.', 1)[1]

    if '/' in samplefileName:

        data_dir = samplefileName.rsplit('/', 1)[0]
        os.chdir(data_dir)

    listOfDataFiles = glob.glob('*.' + fextension)
    listOfDataFiles.sort()
    print('MESSAGE: Loading files ' +
          samplefileName.rsplit('_', 1)[0] + '*.' + fextension)

    if 'tif' in fextension:
        fextension = 'tiff'  # data exchange uses tiff instead of tif
    else:
        raise Exception('align_many_tif: cannot open this file format.')

    os.makedirs('aligned_' + fextension, exist_ok=True)
    os.makedirs('aligned_png', exist_ok=True)

    shift_Vert = np.int16(np.rint(np.linspace(0, totalShift[0],
                                              len(listOfDataFiles))))

    shift_Hor = np.int16(np.rint(np.linspace(0, totalShift[1],
                                             len(listOfDataFiles))))

    allShifts = np.array([shift_Vert*-1, shift_Hor*-1])
    allShifts = allShifts.T

    outFilesList = []

    for imgfname, shift_i, shift_j in zip(listOfDataFiles, shift_Vert, shift_Hor):

        img = dxchange.read_tiff(imgfname)
        print('MESSAGE: aligning ' + imgfname)

        if shift_j > 0:

            img = np.pad(img[:, :-shift_j],
                          ((0, 0), (shift_j, 0)), 'constant')
        else:

            img = np.pad(img[:, -shift_j:],
                          ((0, 0), (0, -shift_j)), 'constant')

        if shift_i > 0:

            img = np.pad(img[:-shift_i, :],
                          ((shift_i, 0), (0, 0)), 'constant')
        else:

            img = np.pad(img[-shift_i:, :],
                          ((0, -shift_i), (0, 0)), 'constant')

        # save files
        outfname = 'aligned_' + fextension + "/" + \
                   imgfname.split('.')[0].rsplit('/', 1)[-1] + \
                   '_aligned.' + fextension

        outFilesList.append(outfname)

        if 'tif' in fextension:
            dxchange.write_tiff(img, outfname)

        print('MESSAGE: file ' + outfname + ' saved.')

        plt.figure(figsize=(8, 7))
        plt.imshow(img, cmap='viridis')
        plt.title('ALIGNED, ' + imgfname.split('/')[-1])

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.savefig(outfname.replace(fextension, 'png'))

        if displayPlots:
            plt.show(block=False)
            plt.pause(.1)
        else:
            plt.close()

    return outFilesList, allShifts

# %%


inifname = '/home/grizolli/workspace/pythonWorkspace/imaging/single_grating/.align_many_imgs_and_save.ini'

# %%
idx4crop = list(map(int, (wpu.get_from_ini_file(inifname, 'Parameters',
                                                    'Crop').split(','))))
defaults = wpu.load_ini_file(inifname)
deafaultfileName = defaults['Files'].get('Reference')

# %% Load

if len(sys.argv) == 1:
    samplefileName = easyqt.get_file_names("Choose the reference file " +
                                           "for alignment")

    if samplefileName==[]:
        samplefileName=deafaultfileName
    else:
        samplefileName = samplefileName[0]
        wpu.set_at_ini_file(inifname, 'Files', 'reference', samplefileName)

    optionRef = easyqt.get_choice('Align images to reference ot to previous image?',
                                  choices=['Reference', 'Previous', 'Manual'])

    fixRef = (optionRef == 'Reference')

    if optionRef!='Manual':
        option = easyqt.get_choice('Crop or Pad?', choices=['Pad', 'Crop'])
else:
    samplefileName = sys.argv[1]
    option = sys.argv[2]
    fixRef = ('Ref' in sys.argv[2])

displayPlots = False


# %%

if optionRef == 'Manual':

    totalShift_i = easyqt.get_int('Total Vertical Shift', default_value=0,
                                  min_=-5000, max_=5000)
    totalShift_j = easyqt.get_int('Total Horizontal Shift', default_value=0,
                                  min_=-5000, max_=5000)

    _, allShifts = align_many_imgs_linearshifts(samplefileName,
                                                totalShift=[-totalShift_i,
                                                            -totalShift_j])

else:

    img_ref = dxchange.read_tiff(samplefileName)



    if easyqt.get_yes_or_no('New Crop?'):

        [colorlimit,
         cmap] = wpu.plot_slide_colorbar(img_ref,
                                         title='SELECT COLOR SCALE,\n' +
                                         'Raw Image, No Crop',
                                         xlabel=r'x [$\mu m$ ]',
                                         ylabel=r'y [$\mu m$ ]')

        idxROI = wpu.graphical_roi_idx(img_ref,
                                       kargs4graph={'cmap': cmap,
                                                    'vmin': colorlimit[0],
                                                    'vmax': colorlimit[1]})

        wpu.set_at_ini_file(inifname, 'Parameters', 'Crop',
                            '{}, {}, {}, {}'.format(idxROI[0], idxROI[1],
                                                    idxROI[2], idxROI[3]))
    else:
        idxROI = idx4crop

    _, allShifts = wpu.align_many_imgs(samplefileName, idxROI=idxROI,
                                       option=option.lower(),
                                       fixRef=fixRef,
                                       displayPlots=displayPlots)

# %%

wpu.save_csv_file([np.arange(0, allShifts.shape[0]),
                   allShifts[:, 0], allShifts[:, 1]],
                  'aligned_png/displacements.csv')

# %%

plt.figure()
#    plt.plot(allShifts[:, 1], allShifts[:, 0], '-o')  # allShifts is ij coordenates

for i in range(allShifts.shape[0]-1):
    if (allShifts[i, 1]==allShifts[i+1, 1] and allShifts[i, 0] == allShifts[i+1, 0]):
        pass
    else:
        plt.quiver(allShifts[i, 1], allShifts[i, 0],
                   allShifts[i+1, 1]-allShifts[i, 1],
                   allShifts[i+1, 0]-allShifts[i, 0],
                   scale_units='xy', angles='xy', scale=1)

plt.autoscale(axis='both')
plt.title('Image displacement')
plt.xlabel('Horizontal Shift [Pixels]')
plt.ylabel('Vertical Shift [Pixels]')
plt.grid('on')

plt.gca().invert_yaxis()
plt.savefig('aligned_png/displacements.png')

plt.show(block=True)


# %%

wpu.print_red('MESSAGE: Done!')


# %%
#
#
#
#import numpy as np
#
#plt.ioff()
#count = 0
#for i in range(-10, 10, 1):
#
#    foo = np.pad(wpu.dummy_images('Shapes', (500, 500), noise=1), 100, 'edge')[50+i:650+i, 50-i:650-i]
#
#    foo = np.pad(foo, 200, 'edge')
#
#    foo = np.uint16((foo-foo.min())/(foo.max()-foo.min())*(2**16-1))
#    dxchange.writer.write_tiff(foo - foo.min(), 'data_{:0>2}.tiff'.format(count), dtype=np.uint16)
#    count += 1
#
#
#plt.close('all')


