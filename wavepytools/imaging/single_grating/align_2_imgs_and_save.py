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


import numpy as np

import matplotlib.pyplot as plt
import wavepy.utils as wpu
import dxchange

# %% Load
img1fname, img2fname = wpu.gui_load_data_dark_filenames()

img1 = dxchange.read_tiff(img1fname)
img2 = dxchange.read_tiff(img2fname)


# %% alignment 1
img1_aligned, img2_aligned, pixel_shift = wpu.gui_align_two_images(img1, img2, option='pad')


# save files
outfname = wpu.get_unique_filename(img1fname.split('.')[0] + '_aligned',
                                   'tiff')
dxchange.write_tiff(img1_aligned, outfname)
wpu.print_blue('MESSAGE: file ' + outfname + ' saved.')

outfname = wpu.get_unique_filename(img2fname.split('.')[0] + '_aligned',
                                   'tiff')
dxchange.write_tiff(img2_aligned, outfname)
wpu.print_blue('MESSAGE: file ' + outfname + ' saved.')


# %%

plt.figure(figsize=(12, 12))
plt.imshow(img1_aligned[::5, ::5], cmap='viridis')
plt.title('img1')
wpu.save_figs_with_idx('aligned')
plt.show(block=False)

plt.figure(figsize=(12, 12))
plt.imshow(img2_aligned[::5, ::5], cmap='viridis')
plt.title('img2')
wpu.save_figs_with_idx('aligned')
plt.show(block=True)
print('Bye')
