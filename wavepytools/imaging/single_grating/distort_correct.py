# this file contains all the functions needed to pre-process the data
# 1. eliminate the electronic noise: dark image
# 2. eliminate the non-uniform affect from Scintillator: flat image
# 3. eliminate the imaging optics distortion: distort image

'''
    23/07/2019
    provide support function for the main code
'''
import os
import sys
import numpy as np
import tifffile as tiff
import glob
import scipy.interpolate as sfit

def dark_err(dark_img, method='median'):
    '''
        this function can eliminate the electronic noise by using dark image
        input:
            dark_img:           file path or dark image data
            method:             'median' or 'average'
        output:
            dark_img_err:       dark image error
    '''
    # if the dark image is a file path, open and load data
    if os.path.isfile(dark_img):
        dark_data = tiff.imread(dark_img)
    elif isinstance(dark_img, np.ndarray):
        # if the dark_img is ndarray data, continue
        dark_data = np.copy(dark_img)
    else:
        print('error dark image type. It should be a file path or ndarray data')
        sys.exit()
    
    # process the dark image
    if method == 'median':
        dark_img_err = np.median(dark_data)
    elif method == 'average':
        dark_img_err = np.average(dark_data)
    else:
        print('wrong dark image processing method. It should be \'median\' or \'average\'.')
        sys.exit()

    return dark_img_err


def flat_err(flat_img, saveFlag=False, savePath=None):
    '''
        this function is used to process the flat image and correct the scintillator
        non-uniformaty. the flat_img can be a file path or single image
        input:
            flat_img:       flat image file path or single processed image
            saveFlag:       to save the flat image or not.
            savePath:       the path to save the flat image
        output:
            flat_data:     processed image

    '''
    if os.path.isdir(flat_img):
        # if the flat_img is a directory, read all the data under this folder
        listOfFiles = glob.glob(flat_img + '/*.tif')
        listOfFiles.sort()
        listOfData = []
        for fname in listOfFiles:
            print('\033[32m'+'MESSAGE: Open File ' + fname+'\033[0m')
            temp_data = tiff.imread(fname)
            listOfData.append(temp_data)

        flat_data = np.array(listOfData)
        flat_data = np.sum(flat_data, axis=0)/flat_data.shape[0]
    elif os.path.isfile(flat_img):
        # if the flat image is already saved
        flat_data = tiff.imread(flat_img)
    elif isinstance(flat_img, np.ndarray):
        if len(flat_img.shape) > 2:
            flat_data = np.sum(flat_img, axis=0)/flat_img.shape[0]
        elif len(flat_img.shape) == 2:
            flat_data = np.copy(flat_img)
        else:
            print('\033[32m'+'MESSAGE: wrong dimension of flat image ' + fname+'\033[0m')
            sys.exit()
    else:
        print('\033[32m'+'MESSAGE: wrong flat image data. should be folder path, file, or 3D/2D ndarray data' + fname+'\033[0m')
        sys.exit()
    
    if saveFlag:
        file_flat = os.path.join(savePath,
                   'flat_image' + '.tiff')
        tiff.imsave(file_flat, np.array(flat_data, dtype='uint16'))
        # np.savetxt(file_flat, flat_data, 
        #            fmt = '%.18e', delimiter = '\t', newline = '\n')

    return flat_data

def image_corr(img, dark_err=None, flat_err=None):
    '''
        this function use the error data from the dark and flat image to correct the raw data
        input:
            img:            raw image, 2D or 3D stack
            dark_err:       dark image error, if absent, the noise is 1e-3
            flat_err:       flat image error, if absent, do not use flat field
        output:
            img_corr:       image after correction
    '''
    if dark_err is None:
        dark_err_last = 1e-3
    else:
        dark_err_last = dark_err
    eps = np.finfo(float).eps

    if len(img.shape) == 2:

        if flat_err is None:
            img_corr = (img - dark_err_last) * ((img - dark_err_last) > 0) + eps * ((img - dark_err_last) <=0)
        else:
            flat_err_diff = (flat_err - dark_err_last) * ((flat_err - dark_err_last) > 0) + eps * ((flat_err - dark_err_last) <=0)
            img_corr = (img - dark_err_last) / flat_err_diff
    elif len(img.shape) == 3:
        img_corr = []
        for img_each in img:
            if flat_err is None:
                img_temp = (img_each - dark_err_last) * ((img_each - dark_err_last) > 0) + eps * ((img_each - dark_err_last) <=0)
            else:
                flat_err_diff = (flat_err - dark_err_last) * ((flat_err - dark_err_last) > 0) + eps * ((flat_err - dark_err_last) <=0)
                img_temp = (img_each - dark_err_last) / flat_err_diff
            img_corr.append(img_temp)
        img_corr = np.array(img_corr)
    else:
        img_corr = []
    return img_corr

def distort_corr(img, dis_img, position):
    '''
        distort_corr function to correct the influence of image optics distortion
        the distortion maps are used, which can be got from the speckle tracking
        input:
            img:            raw image, 2D or 3D stack
            dis_img:        file path to the distortion map image,[file_dis_img_H, file_dis_img_V]
            position:       the distortion data position in the raw image

        output:
            udist_img:      processed image
    '''
    dist_map_h = np.loadtxt(dis_img[0])
    dist_map_v = np.loadtxt(dis_img[1])
    Y_start, Y_end, X_start, X_end = position
    if dist_map_h.shape[0] != Y_end - Y_start + 1 or dist_map_h.shape[1] != X_end - X_start + 1:
        print('Error, the position is not consistent with the distortion data')
        sys.exit()

    if len(img.shape) == 2:
        # if the image data is a 2D matrix
        img_temp = np.copy(img)
        [raw_y, raw_x] = img.shape
        y_axis = np.arange(raw_y) - round(raw_y/2)
        x_axis = np.arange(raw_x) - round(raw_x/2)

        [YY, XX] = np.meshgrid(y_axis, x_axis, indexing='ij')
        # need to be done. here the dimension of dist_XX and dist_map_h might be different. need to broadcast
        # here the size of the distortion map is supposed to be smaller or same size.
        if dist_map_h.shape[0] != raw_y or dist_map_h.shape[1] != raw_x:
            # here assums the raw image has larger size than the distortion map data
                dist_XX = np.copy(XX)
                dist_YY = np.copy(YY)
                temp_dx = np.zeros(dist_XX.shape)
                temp_dy = np.zeros(dist_YY.shape)
                temp_dx[Y_start:Y_end+1, X_start:X_end+1] = dist_map_h
                temp_dy[Y_start:Y_end+1, X_start:X_end+1] = dist_map_v
                dist_XX = dist_XX + temp_dx
                dist_YY = dist_YY + temp_dy

        else:
            dist_XX = XX + dist_map_h
            dist_YY = YY + dist_map_v

        # regular grid interplation
        f = sfit.RegularGridInterpolator((y_axis, x_axis), img_temp, bounds_error = False, method = 'nearest', fill_value = 0)

        pts = (np.array([np.ndarray.flatten(dist_YY), np.ndarray.flatten(dist_XX)])
            .transpose())

        img_interp = f(pts)
        img_interp = np.reshape(img_interp,(raw_y, raw_x))
        
        # # 2D interplation, use 'cubic', 'quintic'
        # f = scipy.interpolate.interp2d(dist_XX, dist_YY, img_temp, bounds_error=False, method='cubic', fill_value=np.finfo(float).eps)
        # img_interp = f(XX, YY)
    elif len(img.shape) == 3:
        # if the img data contains a 3D matrix, the 1st dimension is the stack axis, then do all the
        # process for all the images.
        img_interp = []
        for s_image in img:
            img_temp = np.copy(s_image)
            [raw_y, raw_x] = img_temp.shape
            y_axis = np.arange(raw_y) - round(raw_y/2)
            x_axis = np.arange(raw_x) - round(raw_x/2)

            [YY, XX] = np.meshgrid(y_axis, x_axis, indexing='ij')
            # need to be done. here the dimension of dist_XX and dist_map_h might be different. need to broadcast
            # here the size of the distortion map is supposed to be smaller or same size.
            if dist_map_h.shape[0] != raw_y or dist_map_h.shape[1] != raw_x:
                # here assums the raw image has larger size than the distortion map data
                dist_XX = np.copy(XX)
                dist_YY = np.copy(YY)
                temp_dx = np.zeros(dist_XX.shape)
                temp_dy = np.zeros(dist_YY.shape)
                temp_dx[Y_start:Y_end+1, X_start:X_end+1] = dist_map_h
                temp_dy[Y_start:Y_end+1, X_start:X_end+1] = dist_map_v

                dist_XX = dist_XX + temp_dx
                dist_YY = dist_YY + temp_dy
            else:
                dist_XX = XX + dist_map_h
                dist_YY = YY + dist_map_v

            # regular grid interplation
            f = sfit.RegularGridInterpolator((y_axis, x_axis), img_temp, bounds_error = False, method = 'nearest', fill_value = 0)

            pts = (np.array([np.ndarray.flatten(dist_YY), np.ndarray.flatten(dist_XX)])
                .transpose())

            img_temp = f(pts)
            img_interp.append(np.reshape(img_temp, (raw_y, raw_x)))
            
            # # 2D interplation, use 'cubic', 'quintic'
            # f = scipy.interpolate.interp2d(XX, YY, img_temp, bounds_error=False, kind='cubic', fill_value=np.finfo(float).eps)
            # img_interp.append(f(dist_XX, dist_YY))
        img_interp = np.array(img_interp)

    return img_interp
        
    
