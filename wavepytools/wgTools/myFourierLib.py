# -*- coding: utf-8 -*-  #
"""
Created on Tue Mar  3 11:00:26 2015

@author: wcgrizolli
"""

import numpy as np
from numpy.fft import *

# %% PROPAGATORS

def propIR(u1,Lx,Ly,wavelength,z):

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    dy=Ly/My    #sample interval
    k=2*np.pi/wavelength #wavenumber
    x=np.linspace(-Lx/2,Lx/2-dx, Mx)  #spatial coords
    y=np.linspace(-Ly/2,Ly/2-dy, My)

    X,Y=np.meshgrid(x,y)

    h=1/(1j*wavelength*z)*np.exp(1j*k/(2*z)*(X**2+Y**2)) # impulse

    H=fft2(fftshift(h))*dx*dy#create trans func

    U1=fft2(fftshift(u1)) #shift, fft src field

    U2=H*U1  #multiply

    u2=ifftshift(ifft2(U2)) #inv fft, center obs field

    return u2

def propTF(u1,Lx,Ly,wavelength,z):

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    dy=Ly/My    #sample interval
    k=2*np.pi/wavelength #wavenumber


    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Mx)
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,My)     #freq coords
    [FX,FY]=np.meshgrid(fx,fy)

    H = np.exp(-1j*np.pi*wavelength*z*(FX**2+FY**2))       #trans func
    H = fftshift(H)     #shift trans func

    U1=fft2(fftshift(u1))     #shift, fft src field
    U2=H*U1      #multiply

    u2=ifftshift(ifft2(U2)) #inv fft, center obs field

    return u2


def propIR_RayleighSommerfeld(u1,Lx,Ly,wavelength,z):

    print('Propagation Using RayleighSommerfeld TF')
    print('WG: min number of points for oversampling: %d '
          % RayleighSommerfeldMinSampling(Lx, z, wavelength))

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    dy=Ly/My    #sample interval
    k=2*np.pi/wavelength #wavenumber
    x=np.linspace(-Lx/2,Lx/2-dx, Mx)  #spatial coords
    y=np.linspace(-Ly/2,Ly/2-dy, My)

    X,Y=np.meshgrid(x,y)

    r = np.sqrt(X**2 + Y**2 + z**2)
#    h=1/(1j*wavelength*z)*np.exp(1j*k/(2*z)*(X**2+Y**2)) # impulse
    h=z/(1j*wavelength)/r**2*np.exp(1j*k*r) # impulse RayleighSommerfeld

    H=fft2(fftshift(h))*dx*dy#create trans func

    U1=fft2(fftshift(u1)) #shift, fft src field

    U2=H*U1  #multiply

    u2=ifftshift(ifft2(U2)) #inv fft, center obs field

    return u2


def propTF_RayleighSommerfeld(u1, Lx, Ly, wavelength, z):

    print('Propagation Using RayleighSommerfeld TF')

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    dy=Ly/My    #sample interval
    k=2*np.pi/wavelength #wavenumber


    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Mx)
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,My)     #freq coords
    [FX,FY]=np.meshgrid(fx,fy)


#    H = np.exp(-1j*np.pi*wavelength*z*(FX**2+FY**2))       #trans func
    H = np.exp(1j*k*z*np.sqrt(1.0-(wavelength*FX)**2-(wavelength*FY)**2))
                                        #trans func RayleighSommerfeld
    H = fftshift(H)     #shift trans func

    U1=fft2(fftshift(u1))     #shift, fft src field
    U2=H*U1      #multiply

    u2=ifftshift(ifft2(U2)) #inv fft, center obs field

    return u2


def propTForIR(u1,Lx,Ly,wavelength,zz):

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval

    if dx > wavelength*zz/Lx:
        print('Propagation Using Fresnel TF')
        return propTF(u1,Lx,Ly,wavelength,zz)
    elif dx < wavelength*zz/Lx:
        print('Propagation Using Fresnel IR')
        return propIR(u1,Lx,Ly,wavelength,zz)
    else:
        print('I could use either TF or IR')
        print('Propagation Using Fresnel TF')
        return propTF(u1,Lx,Ly,wavelength,zz)


def prop2step(u1,L1,L2,wavelength,z):
    #  propagation - 2 step Fresnel diffraction method
    #  assumes uniform sampling and square array
    #  u1 - complex field at source plane
    #  L1 - source plane side-length
    #  L2 - observation plane side-length
    #  wavelength - wavelength
    #  z - propagation distance
    #  u2 - output field at observation plane

    print('Propagation Using Fresnel Two Steps Propagator')


    (M, N) = np.shape(u1)
    # input array size
    k = 2*np.pi/wavelength
    # wavenumber

    #  source plane
    dx1 = L1/M
    x1 = np.linspace(-L1/2, L1/2 - dx1, M)
    [X, Y] = np.meshgrid(x1, x1)
    u = u1*np.exp(1j*k/(2*z*L1)*(L1-L2)*(X**2+Y**2))
    u = fft2(fftshift(u))

    #  dummy (frequency) plane
    fx1 = np.linspace(-1/(2*dx1), 1/(2*dx1) - 1/L1, M)
    fx1 = fftshift(fx1)
    [FX1, FY1] = np.meshgrid(fx1, fx1)
    u = np.exp(-1j*np.pi*wavelength*z*L1/L2*(FX1**2+FY1**2))*u
    u = ifftshift(ifft2(u))

    #  observation plane
    dx2 = L2/M
    x2 = np.linspace(-L2/2, L2/2-dx2, M)
    [X, Y] = np.meshgrid(x2, x2)
    u2 = (L2/L1)*u*np.exp(-1j*k/(2*z*L2)*(L1-L2)*(X**2+Y**2))
    u2 = u2*dx1**2/dx2**2
    # x1 to x2 scale adjustment
    return u2


def propFF(u1, L1, wavelength, z):
    # ASSUMES FRESNEL APPROXIMATION
    # propagation - Fraunhofer pattern
    # assumes uniform sampling
    # u1 - source plane field
    # L1 - source plane side length
    # wavelength - wavelength
    # z - propagation distance
    # L2 - observation plane side length
    # u2 - observation plane field
    #

    (M, N) = np.shape(u1)
    # input array size
    k = 2*np.pi/wavelength
    # wavenumber

    #  source plane
    dx1 = L1/M

    #get input field array size
    #source sample interval
    #wavenumber
    #
    L2 = wavelength*z/dx1
    #obs sample interval
    dx2 = wavelength*z/L1
    x2 = np.linspace(-L2/2, L2/2 - dx2, M)
    #obs coords

    [X2, Y2] = np.meshgrid(x2, x2)
    #
    c = 1/(1j*wavelength*z)*np.exp(1j*k/(2*z)*(X2**2+Y2**2))
    u2 = c*ifftshift(fft2(fftshift(u1)))*dx1**2

    return u2, L2


def checkParaxialCondition(apperture, propagationDist, wavelength,
                           checkValue=1e-3, verbose=False):
    '''
    According to eq. 4.1-13 in Saleh
    '''

    thirdTermOfExpansion = apperture**4.0/4.0/propagationDist**3/wavelength

    if verbose:

        print('\nFresnel Approximation, Condition of Validity:')
        print('thirdTermOfExpansion must be << 1')
        print('\nWe have thirdTermOfExpansion: %.5g' % thirdTermOfExpansion)
        # if thirdTermOfExpansion

        print('\n')
        print('For a=%.3gmm and lambda=%.4gnm, d must be >> %g m' % \
              (apperture*1e3, wavelength*1e9,
               (apperture**4.0/4.0/wavelength)**(1.0/3.0)))

    return thirdTermOfExpansion < checkValue


def fresnelNumber(a, d, wavelength, verbose=False):

    nf = a**2/wavelength/d

    if verbose:
        print(('Nf = %.5g' % np.abs(nf)))
        print('Conditions:')
        print('Nf << 1 Fraunhofer regime;')
        print('Nf >> 1 Geometric Optic;')
        print('Nf -> 1 Fresnel Difraction.')

    return nf


def RayleighSommerfeldMinSampling(rhoMax, zMin, wavelength):
    '''
    Return minumum number of points in one dimmension to obtain eversampling.
    Acording to eq. 22 in the article "Fast-Fourier-transform based
    numerical integration method for the Rayleigh–Sommerfeld
    diffraction formula", doi: 10.1364/AO.45.001102
    '''

    from math import sqrt

    return rhoMax/(sqrt(wavelength**2 + rhoMax**2 + 2*wavelength*sqrt(rhoMax**2 + zMin**2)) - rhoMax)













