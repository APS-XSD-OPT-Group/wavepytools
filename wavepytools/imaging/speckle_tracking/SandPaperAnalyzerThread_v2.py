# -*- coding: utf-8 -*-  #
"""
Created on Sun Jan 31 14:47:48 2016

@author: xhxiao
"""
import numpy as np
import skimage.feature as skf
import skimage.io as sio
import matplotlib.pylab as mpl
import scipy.misc as sm
import h5py
from bokeh.plotting import figure, output_notebook, show
from multiprocessing import Pool,Process, Array
import ctypes
import math
import timeit



def child(index, procs, iiStart, iiEnd, jjStart, jjEnd):

    step = int((iiEnd-iiStart)*1.0/procs+1)
    start = step*index + iiStart
    stop = start + step

    if start >= iiEnd:
        return 0

    if stop > iiEnd:
        stop = iiEnd

    errorMax = 1e10
    shiftX = 0
    shiftY = 0

    for ii in range(start,stop):
        for jj in range(jjStart,jjEnd):
            for kk in range(-srhSz,srhSz):
                for ll in range(-srhSz,srhSz):
                    shift, error, diffphase = skf.register_translation(rImage[(ii+kk):(ii+kk+winSz),(jj+ll):(jj+ll+winSz)], sImage[ii:(ii+winSz),jj:(jj+winSz)])
                    if error < errorMax:
                        errorMax = error
                        shiftY = shift[0]
                        shiftX = shift[1]
            shift, error, diffphase = skf.register_translation(rImage[ii+shiftY:ii+shiftY+winSz,jj+shiftX:jj+shiftX+winSz], sImage[ii:ii+winSz,jj:jj+winSz],upsample_factor=100)

            # if(ii == 303 and jj == 1000):
            #     print ii, jj , shift, error, diffphase
    #        phaseGrad[ii-srhSz,jj-srhSz,:] = [shiftY+shift[0],shiftX+shift[1]]
            phaseGradY[ii-iiStart,jj-jjStart] = shiftY+shift[0]
            phaseGradX[ii-iiStart,jj-jjStart] = shiftX+shift[1]
            # print ii,jj,phaseGradY[ii-300,jj-1000], phaseGradX[ii-300,jj-1000]
            shiftX = 0
            shiftY = 0
            errorMax = 1e10
            shiftMax = 0

    return 1

if __name__ == "__main__":
    winSz = 15
    srhSz = 10
#    refImage_filename = "/local/kyue/anlproject/correlation-Xianghui/1600grift_0002.hdf"
#    samImage_filename = "/local/kyue/anlproject/correlation-Xianghui/6333R_1600grift_0003.hdf"
#    bkgImage_filename = "/local/kyue/anlproject/correlation-Xianghui/6333R_1600grift_dark_0001.hdf"

#    f = h5py.File(refImage_filename,'r')
#    refImg = np.array(f['/exchange/data_dark'])
#    #print refImg
#    f.close()
#    dp,dy,dx = refImg.shape
#
#    f = h5py.File(samImage_filename,'r')
#    samImg = np.array(f['/exchange/data_dark'])
#    f.close()
#
#    f = h5py.File(bkgImage_filename,'r')
#    bkgImg = np.array(f['/exchange/data_dark'])
#    f.close()

#    sImage = samImg[5,:,:] - np.mean(bkgImg,axis=0)
#    rImage = refImg[5,:,:] - np.mean(bkgImg,axis=0)

    refImage_filename = "/media/2BM_Backup43_/2016_02/MoS2_20keV_sandpaper1600_speckle/reference.tif"
    samImage_filename = "/media/2BM_Backup43_/2016_02/MoS2_20keV_sandpaper1600_speckle/sample.tif"
    bkgImage_filename = "/media/2BM_Backup43_/2016_02/MoS2_20keV_sandpaper1600_speckle/dark.tif"
    refImg = sm.imread(refImage_filename)
    samImg = sm.imread(samImage_filename)
    bkgImg = sm.imread(bkgImage_filename)
    sImage = samImg - bkgImg
    rImage = refImg - bkgImg


    norSamImg = sImage/rImage
    norSamImg[np.isinf(norSamImg)] = 0

# mpl.figure(0)
# mpl.imshow(norSamImg,vmin=0,vmax=1)
# mpl.figure(1)
# mpl.imshow(sImage)
# mpl.show()

#output_notebook('bokeh_test')
#p = figure(tools="pan,box_zoom,reset,save",ya)

#phaseGrad = np.ndarray([dy-winSz-2*srhSz,dx-winSz-2*srhSz,2])
#phaseGrad = np.ndarray([100,400,2])
    # phaseGradY = np.ndarray([100,400])
    # phaseGradX = np.ndarray([100,400])

    start1 = timeit.default_timer()

# Start of the multiprocessing
    iiStart = 300
    iiEnd = 320
    jjStart = 1000
    jjEnd = 1020
    procs = 10

    iiX = iiEnd-iiStart
    jjY = jjEnd-jjStart

    sharedY = Array(ctypes.c_double, iiX*jjY, lock=False)
    sharedX = Array(ctypes.c_double, iiX*jjY, lock=False)

    phaseGradY = np.frombuffer(sharedY, dtype=ctypes.c_double)
    phaseGradY = phaseGradY.reshape(iiX, jjY)
    phaseGradX = np.frombuffer(sharedX, dtype=ctypes.c_double)
    phaseGradX = phaseGradX.reshape(iiX, jjY)
# print samImg.shape,refImg.shape
#for ii in range(srhSz,dy-winSz-srhSz):
#    for jj in range(srhSz,dx-winSz-srhSz):


    workjob = []
    for i in range(0, procs):
        process = Process(target=child, args = (i,procs, iiStart, iiEnd, jjStart, jjEnd))
        workjob.append(process)

    for j in workjob:
        j.start()

    for j in workjob:
        j.join()
# End of the multiprocessing

    # print pGrady[0:4,0]

    stop1 = timeit.default_timer()
    print("end processing", (stop1 - start1))



    phaseGradY2 = np.ndarray([20,20])
    phaseGradX2 = np.ndarray([20,20])
    errorMax = 1e10
    shiftX = 0
    shiftY = 0
    start1 = timeit.default_timer()
    for ii in range(iiStart,iiEnd):
        for jj in range(jjStart,jjEnd):
            for kk in range(-srhSz,srhSz):
                for ll in range(-srhSz,srhSz):
                    shift, error, diffphase = skf.register_translation(rImage[(ii+kk):(ii+kk+winSz),(jj+ll):(jj+ll+winSz)], sImage[ii:(ii+winSz),jj:(jj+winSz)])
                    if error < errorMax:
                        errorMax = error
                        shiftY = shift[0]
                        shiftX = shift[1]
            shift, error, diffphase = skf.register_translation(rImage[ii+shiftY:ii+shiftY+winSz,jj+shiftX:jj+shiftX+winSz], sImage[ii:ii+winSz,jj:jj+winSz],upsample_factor=100)
    #        phaseGrad[ii-srhSz,jj-srhSz,:] = [shiftY+shift[0],shiftX+shift[1]]
            phaseGradY2[ii-iiStart,jj-jjStart] = shiftY+shift[0]
            phaseGradX2[ii-iiStart,jj-jjStart] = shiftX+shift[1]
            # print ii,jj,phaseGradY[ii-300,jj-1000], phaseGradX[ii-300,jj-1000]
            shiftX = 0
            shiftY = 0
            errorMax = 1e10
            shiftMax = 0

    stop1 = timeit.default_timer()
    print("end processing", (stop1 - start1))
    resultY = phaseGradY2 - phaseGradY
    resultX = phaseGradX2 - phaseGradX
#
#    print phaseGradX[3][0], phaseGradX2[3][0], resultX[3][0]
    print phaseGradY, phaseGradX
    print phaseGradY2, phaseGradX2
    print resultX, resultY

    sio.imsave("/media/2BM_Backup43_/2016_02/MoS2_20keV_sandpaper1600_speckle/phaseGradY_Y300-400_X1000-1400.tif",phaseGradY)
    sio.imsave("/media/2BM_Backup43_/2016_02/MoS2_20keV_sandpaper1600_speckle/phaseGradX_Y300-400_X1000-1400.tif",phaseGradX)
    mpl.figure(1)
    mpl.imshow(phaseGradY)
    mpl.figure(2)
    mpl.imshow(phaseGradX)
    mpl.show()
