# -*- coding: utf-8 -*-  #
"""
Created on Mon Feb 23 17:42:31 2015

@author: wcgrizolli


How to import:

sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import myOpticsLib


"""

import numpy as np

### Physics formulas

### gaussian source

#def gaussianBeam(fwhm, wavelength, z, L, npoints):
#    ''' Create gaussian beam acconrding to equation 3.1-7 from Saleh
#    '''
#
#    Y, X = np.mgrid[-L/2:L/2:1j*npoints, -L/2:L/2:1j*npoints]
#
#    # equation 3.1-7 from Saleh
#    Wo = fwhm*0.84932180028801907  # Wo = 2 sigma
#    zo = np.pi*Wo**2/wavelength
#    Wz = Wo*np.sqrt(1.0 + (z/zo)**2)
##    Rz = z*(1.0 + (zo/z)**2)
#    inverseRz = z/(z**2 + zo**2)  # inverse of Rz, to avoid division by zero
#    zeta = np.arctan(z/zo)
#    k = 2.0*np.pi/wavelength
#
#    rho2 = X**2 + Y**2
#
#    return Wo/Wz*np.exp(-rho2/Wz**2)*np.exp(-1j*k*z-1j*k*rho2/2*inverseRz+1j*zeta)


def gaussianBeam(x, y, fwhm, z, wavelength):
    ''' Create gaussian beam according to equation 3.1-7 from Saleh
    '''

    # equation 3.1-7 from Saleh
    Wo = fwhm*0.84932180028801907  # Wo = 2 sigma
    zo = np.pi*Wo**2/wavelength
    Wz = Wo*np.sqrt(1.0 + (z/zo)**2)
#    Rz = z*(1.0 + (zo/z)**2)
    inverseRz = z/(z**2 + zo**2)  # inverse of Rz, to avoid division by zero
    zeta = np.arctan(z/zo)
    k = 2.0*np.pi/wavelength

    rho2 = x**2 + y**2  # rho**2

    return Wo/Wz*np.exp(-rho2/Wz**2)*np.exp(-1j*k*z-1j*k*rho2/2*inverseRz+1j*zeta)


def gaussianBeamAst(x, y, fwhm_x, fwhm_y, z=0, zxo=0, zyo=0,  wavelength=1e-9):
    '''
    Create astigmatic Gaussian beam, where the horizontal and vertical
    waists are at different positions

    Parameters
    ----------
    x, y : 2D ndarray
        matrix with the values of x and y
    fwhm_x, fwhm_y : float
        Beam waist in x and y (FWHM)
    z : float (or 2D ndarray)
        Propagation distance
    zxo, zyo : float
        distances of x and y waists from z=0 (origin of reference frame)
    wavelength : floats
        wavelength of the radiation
    

    Returns
    -------
    emf : 2D complex array
        EM field of Gaussian beam at z
        
    '''

    Wox = fwhm_x*0.84932180028801907  # Wo = 2 sigma
    zox = np.pi*Wox**2/wavelength
    Wzx = Wox*np.sqrt(1.0 + ((z - zxo)/zox)**2)
    inverseRzx = (z - zxo)/((z - zxo)**2 + zox**2)  # inverse of Rz, to avoid division by zero
    zetax = np.arctan((z - zxo)/zox)
    
    Woy = fwhm_y*0.84932180028801907  # Wo = 2 sigma
    zoy = np.pi*Woy**2/wavelength
    Wzy = Woy*np.sqrt(1.0 + ((z - zyo)/zoy)**2)
    inverseRzy = (z - zyo)/((z - zyo)**2 + zoy**2)  # inverse of Rz, to avoid division by zero
    zetay = np.arctan((z - zyo)/zoy)

    
    k = 2.0*np.pi/wavelength

    return (np.sqrt(Wox/Wzx*Woy/Wzy)* 
            np.exp(-x**2/Wzx**2)* 
            np.exp(-1j*k*(z - zxo)-1j*k*x**2/2*inverseRzx+1j*zetax) * 
            np.exp(-y**2/Wzy**2)* 
            np.exp(-1j*k*(z - zyo)-1j*k*y**2/2*inverseRzy+1j*zetay))

### optics

def imageDistThinLens(focus, p):
    return 1.0/(1.0/focus - 1.0/p)


def imageDistTang(p, R, theta):
    from math import cos
    return 1.0/(2/R/cos(theta) - 1.0/p)


def imageDistSag(p, rho, theta):

    return 1.0/(2/rho*np.cos(theta) - 1.0/p)

### Toroidal Surfaces


deg2rad = np.pi/180.00
rad2deg = 180.00/np.pi

def curvatureRadiusTang(p, q, theta=None, alpha=None):
    ''' calculate the tangential curvature radius. alpha is the angle related to
    the normal and theta is the grazing incidence angle'''
    pi = np.pi

    if alpha is not None:
        theta = pi/2.0 - alpha


    return p*q/(p + q)*2.0/np.sin(theta)


def curvatureRadiusSag(p, q, theta=None, alpha=None):
    ''' calculate the tangential curvature radius. alpha is the angle related to
    the normal and theta is the grazing incidence angle'''
    pi = np.pi

    if alpha is not None:
        theta = pi/2.0 - alpha

    return p*q/(p + q)*2.0*np.sin(theta)


def curvatureRadiusToroid(p, q, theta=None, alpha=None):
    return curvatureRadiusTang(p, q, theta, alpha), curvatureRadiusSag(p, q, theta, alpha)

### Grating


def gratingAngle(alpha, phEnergy, lineDensity, nOrderDiff=1):
    '''
    alpha in rad
    phEnergy in eV
    lineDensity in line/mm
    '''

    hc = 1.23984193e-6  # [eV*meter]
    return np.arcsin(nOrderDiff*lineDensity*1e3*hc/phEnergy - np.sin(alpha))


def gratingAnglesFF(phEnergy, lineDensity, cff, nOrderDiff=1):
    '''
    Grating angles at Fixed Focus Condition
    returns alpha and beta in rad
    phEnergy in eV
    lineDensity in line/mm
    '''

    hc = 1.239842e-6  # [eV*meter]
    alpha = np.arcsin(-nOrderDiff*lineDensity*1e3*hc/phEnergy/(cff**2-1) +
                 np.sqrt(1+(-nOrderDiff*lineDensity*1e3*hc/phEnergy*cff/(cff**2-1))**2))

    return [alpha, gratingAngle(alpha, phEnergy, lineDensity, nOrderDiff)]


def gratingEnergy(alpha, beta, lineDensity, nOrderDiff=1):
    '''
    alpha and beta in rad
    lineDensity in line/mm
    '''

    hc = 1.239842e-6  # [eV*meter]
    return hc/(np.sin(alpha) + np.sin(beta))*nOrderDiff*lineDensity*1e3


def blazeAngle(Eph, cff, linesMM, nOrderDiff=1):
    '''
    alpha and beta in rad
    lineDensity in line/mm
    '''

    [alpha, beta] = gratingAnglesFF(Eph, linesMM, cff, nOrderDiff=nOrderDiff)

    return (alpha + beta)/2.0


def sx700parameters(phEnergy, cff, linDensity,
                    distPreMirrToPG, distPrevOeToPG, nOrderDiff=-1, printFlag=1):
    '''
    phEnergy     :    photon energy in eV for single values calculation
    cff        :    fixed focus constant
    nOrderDiff    : number of the harmonic of interest;
    linDensity    :    line density of the grating [lines/milimeter]
    distPreMirrToPG    :    distance plane mirror to plane grating (l).
    distPrevOeToPG    :    distance grating to source (dg)
    '''


    alpha, beta = gratingAnglesFF(phEnergy, linDensity, cff, nOrderDiff=1)

    theta= - (beta-alpha)/2.0

    preMirrSrcPlan = (distPrevOeToPG-distPreMirrToPG/np.abs(np.tan(beta-alpha)))  # dm

    preMirrorImPlan = np.abs(distPreMirrToPG/np.sin(beta-alpha)/2)  # dmg/2

    if printFlag:
        print('### Grating properties:')
        print('### %.2f eV, cff = %.3f, %f lines/mm\n' % (phEnergy, cff, linDensity))
        print('### Geometric values for SX700:')
        print('### alpha: %.8f deg, beta: %.8f deg,' % (alpha*rad2deg, beta*rad2deg))
        print('### theta: %.8f deg' % (theta*rad2deg))
        print('### dm: %.8f m' % (preMirrSrcPlan))
        print('### dmg/2: %.8f m\n\n\n' % (preMirrorImPlan))



    return [alpha*rad2deg, beta*rad2deg, theta*rad2deg, preMirrSrcPlan, preMirrorImPlan]





