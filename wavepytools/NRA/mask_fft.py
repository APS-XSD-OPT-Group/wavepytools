# -*- coding: utf-8 -*-  #
"""
Created on Thu Oct 20 14:32:46 2016

@author: grizolli
"""






import numpy as np
import skimage.data
import matplotlib.pyplot as plt

import scipy.ndimage
from numpy.fft import *

import wavepy.utils as wpu


#img = np.load('samples/1k_anneau_Al003_crop.jpg')


# %%

fname = '/home/grizolli/workspace/pythonWorkspace/NRA/samples/1k_anneau_Al003_crop.jpg'


img = 220.0 - 150 - scipy.ndimage.imread(fname, mode='F')


img[img < 0] = 0.0

# %%


from skimage.feature import match_template

template = np.zeros((10,10))

template[3:7,3:7] = 1.0


result = match_template(img, template)
ij = np.unravel_index(np.argmax(result), result.shape)
ii, jj = ij[::-1]


fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

ax1.imshow(template)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(img)
ax2.set_axis_off()
ax2.set_title('img')
# highlight matched region
htemplate, wtemplate = template.shape
rect = plt.Rectangle((ii, jj), wtemplate, htemplate, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(ii, jj, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()



# %%

from scipy.ndimage import gaussian_filter

idx_angle_j, idx_angle_i = np.where(gaussian_filter(result,2)>.25)


foo = result*0.0
foo = result[scipy.ndimage.filters.maximum_filter(result, (5,5))==result]



# %%

plt.figure()
plt.imshow(result)
plt.plot(idx_angle_i, idx_angle_j, 'go', alpha=.5, ms=10)

plt.show()




