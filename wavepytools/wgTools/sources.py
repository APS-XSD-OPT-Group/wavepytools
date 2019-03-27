# -*- coding: utf-8 -*-  #
"""
Created on Wed Sep  3 13:18:12 2014

@author: wcgrizolli

auxiliar functions for synchrotron sources



"""





from __future__ import print_function  # Python 2.7 compatibility


import sys
import numpy as np

sys.path.append('/home/wcgrizolli/usr/local/SRW-master/env/work/srw_python')


sys.path.append('/home/wcgrizolli/pythonWorkspace/wgTools')
import wgTools as wgt

sys.path.append('/home/wcgrizolli/usr/local/SRW-master/env/work/srw_python')
from srwlib import * #analysis:ignore


# Some useful constants

#pi = np.pi
#
#deg2rad = pi/180.00
#rad2deg = 180.00/pi

hc = 1.239842e-6
sdv2fwhm = 2.35482
fwhm2sdv = 1.0/sdv2fwhm

def energyToTuneUnd(phEnergy, numPer, nHarmonic=1, beta=-1):

    return phEnergy/(nHarmonic*numPer+beta)*nHarmonic*numPer

def openingAngleSourceUnd(phEnergy, lengthUnd, beta=0.0):
    if beta > 1.0:
        return float('Nan')
    else:

        return np.sqrt((1-beta)*2/lengthUnd*hc/phEnergy)


def divRadSourceUndFWHM(phEnergy, lengthUnd, beta=0.0):
    if beta > .442:
        return float('Nan')
    else:

        return 2*np.sqrt((.442-beta)*2/lengthUnd*hc/phEnergy)


def sizeRadSourceUndFWHM(phEnergy, lengthUnd, beta=0.0):
    if beta > .442:
        return float('Nan')
    else:

        return 11.045*hc/phEnergy/4/np.pi/divRadSourceUndFWHM(phEnergy, lengthUnd, beta=beta)
        # approximation eq. 17 Lic Thesis


def divRadSourceUndGaussian(phEnergy, lengthUnd):
    wavelenght = 1.239842e-6/phEnergy
    return np.sqrt(wavelenght/2.0/lengthUnd)  # Pietman 2.6.2


def sizeRadSourceUndGaussian(phEnergy, lengthUnd):
    wavelenght = 1.239842e-6/phEnergy
    return np.sqrt(wavelenght*lengthUnd/2)/2/np.pi  # Nielsen Chapter 2.4.6



def BfieldUndulator(gap, lambda_u, par_a, par_b, par_c):
    return par_a*np.exp(par_b*gap/lambda_u + par_c*(par_b*gap/lambda_u)**2)
    # Article Design considerations for a 1 Å SASE undulator


def BfieldUndVertField(gap, lambda_u):
    if np.any(gap/lambda_u < 1) and np.any(gap/lambda_u > .1):
        return BfieldUndulator(gap, lambda_u, par_a=2.076, par_b=-3.24, par_c=0.0)
    else:
        return float('Nan')*gap
    # Pure Permanent Magnet planar vertical field


def BfieldUndHorField(gap, lambda_u):
    if np.any(gap/lambda_u < 1) and np.any(gap/lambda_u > .1):
        return BfieldUndulator(gap, lambda_u, par_a=2.400, par_b=-5.69, par_c=1.46)
    else:
        return float('Nan')*gap
    # Pure Permanent Magnet planar horizontal field


def divRadSourceBMsdv(phEnergy, criticalEnergy, Ering):
    if np.any(criticalEnergy/phEnergy < .2) or np.any(criticalEnergy/phEnergy > 100):
        return phEnergy*float('Nan')
    else:
        return 570/Ering*511e3*((criticalEnergy/phEnergy)**.43)*1e-3  # [rad]
        # eq 2.1 Peatman


def sizeRadSourceBMsdv(phEnergy, criticalEnergy, Ering):

    return hc/phEnergy/4/np.pi/divRadSourceBMsdv(phEnergy, criticalEnergy, Ering)


def angPeakUnd(phEnergy, lengthUnd, beta=0.0):
    if beta >= 0.0:
        return 0.0
    else:

        return np.sqrt(-beta*2/lengthUnd*hc/phEnergy)

def Kparameter(phEnergy1st, period, ringEnergy):

    return np.sqrt(4.0*(ringEnergy/511e3)**2*hc/phEnergy1st/period - 2.0)


def calculateUndSourcePropertiesFWHM(ring, phEnergy, lengthUnd, beta=0):
    '''
    calculateUndSourcePropertiesFWHM(ring, phEnergy, lengthUnd, beta=0)

    return np.array([[fwhmRad, fwhmDivRad,
                      fwhmXe, fwhmDivXe,
                      fwhmYe, fwhmDivYe,
                      fwhmXt, fwhmDivXt,
                      fwhmYt, fwhmDivYt,
                      maxOpenAngleX, maxOpenAngleY]])

    WG: 20150514, change of the return var structure
    '''

    fwhmXe = ring.sigX*sdv2fwhm
    fwhmYe = ring.sigY*sdv2fwhm
    fwhmDivXe = ring.sigXp*sdv2fwhm
    fwhmDivYe = ring.sigYp*sdv2fwhm

    fwhmDivRad = divRadSourceUndFWHM(phEnergy, lengthUnd, beta)
    fwhmRad = sizeRadSourceUndFWHM(phEnergy, lengthUnd, beta)

    fwhmXt = np.sqrt(fwhmRad**2 + fwhmXe**2)
    fwhmYt = np.sqrt(fwhmRad**2 + fwhmYe**2)

    fwhmDivXt = np.sqrt(fwhmDivRad**2 + fwhmDivXe**2)
    fwhmDivYt = np.sqrt(fwhmDivRad**2 + fwhmDivYe**2)

    maxOpenAngleRad = openingAngleSourceUnd(phEnergy, lengthUnd, beta)
    maxOpenAngleX = np.sqrt(maxOpenAngleRad**2 + fwhmDivXe**2)
    maxOpenAngleY = np.sqrt(maxOpenAngleRad**2 + fwhmDivYe**2)

    return np.array([fwhmRad, fwhmDivRad,
                      fwhmXe, fwhmDivXe,
                      fwhmYe, fwhmDivYe,
                      fwhmXt, fwhmDivXt,
                      fwhmYt, fwhmDivYt,
                      maxOpenAngleX, maxOpenAngleY], ndmin=1)


def calculateUndSourcePropertiesSDV(ring, phEnergy, lengthUnd, beta=0):
    '''
    calculateUndSourcePropertiesSDV(ring, phEnergy, lengthUnd, beta=0)

    return np.array([[sdvRad, sdvDivRad,
                      sdvXe, sdvDivXe,
                      sdvYe, sdvDivYe,
                      sdvXt, sdvDivXt,
                      sdvYt, sdvDivYt,
                      maxOpenAngleX, maxOpenAngleY]])

    WG: 20150514, change of the return var structure
    '''


    return calculateUndSourcePropertiesFWHM(ring, phEnergy, lengthUnd, beta)*fwhm2sdv

def calculateBMSourcePropertiesSDV(ring, BM, phEnergy):

    sigXe = ring.sigX
    sigYe = ring.sigY
    sigDivXe = ring.sigXp
    sigDivYe = ring.sigYp

    sigDivRad = divRadSourceBMsdv(phEnergy, BM.criticalEnergy, ring.elecEnergy)
    sigRad = sizeRadSourceBMsdv(phEnergy, BM.criticalEnergy, ring.elecEnergy)

    sigXt = np.sqrt(sigRad**2 + sigXe**2)
    sigYt = np.sqrt(sigRad**2 + sigYe**2)

    sigDivXt = np.sqrt(sigDivRad**2 + sigDivXe**2)
    sigDivYt = np.sqrt(sigDivRad**2 + sigDivYe**2)

    return [[sigRad, sigDivRad],
            [sigXe, sigDivXe, sigYe, sigDivYe],
            [sigXt, sigDivXt],
            [sigYt, sigDivYt]]




def calculateUndSourcePropertiesGaussian(ring, phEnergy, lengthUnd):

    sigXe = ring.sigX
    sigYe = ring.sigY
    sigDivXe = ring.sigXp
    sigDivYe = ring.sigYp

    sigDivRad = divRadSourceUndGaussian(phEnergy, lengthUnd)
    sigRad = sizeRadSourceUndGaussian(phEnergy, lengthUnd)

    sigXt = np.sqrt(sigRad**2 + sigXe**2)
    sigYt = np.sqrt(sigRad**2 + sigYe**2)

    sigDivXt = np.sqrt(sigDivRad**2 + sigDivXe**2)
    sigDivYt = np.sqrt(sigDivRad**2 + sigDivYe**2)

    return [[sigRad, sigDivRad],
            [sigXe, sigDivXe, sigYe, sigDivYe],
            [sigXt, sigDivXt],
            [sigYt, sigDivYt]]

def printUndSourceSizeFWHM(ring, phEnergy, lengthUnd, beta=0):
    '''
    SAME THAN LIC: THESIS. beta = 0

    WG: 20150514, change of the return var structure
    '''

    [fwhmRad, fwhmDivRad,
      fwhmXe, fwhmDivXe,
      fwhmYe, fwhmDivYe,
      fwhmXt, fwhmDivXt,
      fwhmYt, fwhmDivYt,
      maxOpenAngleX, maxOpenAngleY] = calculateUndSourcePropertiesFWHM(ring,
                                                                        phEnergy,
                                                                        lengthUnd,
                                                                        beta)



    wgt.color_print('WG: Calculation FWHM Source Size and Divergences at %.2f ' % (phEnergy)
                    + 'eV for beta = %.2f:' % (beta))

    wgt.color_print('WG: %s' % (ring.name))
    wgt.color_print('WG: Undulator Length: %f' % (lengthUnd))

    print('WG: fwhmRad: %.4g um' % (fwhmRad*1e6))
    print('WG: fwhmDivRad: %.4g urad' % (fwhmDivRad*1e6))
    print('WG: fwhmXt: %.4g um' % (fwhmXt*1e6))
    print('WG: fwhmYt: %.4g um' % (fwhmYt*1e6))
    print('WG: fwhmDivXt: %.4g urad' % (fwhmDivXt*1e6))
    print('WG: fwhmDivYt: %.4g urad' % (fwhmDivYt*1e6))
    print('WG: max apert cent. cone X: ± %.4g urad' % (maxOpenAngleX*1e6))
    print('WG: max apert cent. cone Y: ± %.4g urad' % (maxOpenAngleY*1e6))

    print('WG: done!\n\n')


def printUndSourceSizeSDV(ring, phEnergy, lengthUnd, beta=0):
    '''
    SAME THAN LIC: THESIS. beta = 0




    WG: 20150514, change of the return var structure
    '''

    [sdvRad, sdvDivRad,
      sdvXe, sdvDivXe,
      sdvYe, sdvDivYe,
      sdvXt, sdvDivXt,
      sdvYt, sdvDivYt,
      maxOpenAngleX, maxOpenAngleY] = calculateUndSourcePropertiesSDV(ring,
                                                                        phEnergy,
                                                                        lengthUnd,
                                                                        beta)



    wgt.color_print('WG: Calculation sdv Source Size and Divergences at %.2f ' % (phEnergy)
                    + 'eV for beta = %.2f:' % (beta))

    wgt.color_print('WG: %s' % (ring.name))
    wgt.color_print('WG: Undulator Length %f' % (lengthUnd))

    print('WG: sdvRad: %.4g um' % (sdvRad*1e6))
    print('WG: sdvDivRad: %.4g urad' % (sdvDivRad*1e6))
    print('WG: sdvXt: %.4g um' % (sdvXt*1e6))
    print('WG: sdvYt: %.4g um' % (sdvYt*1e6))
    print('WG: sdvDivXt: %.4g urad' % (sdvDivXt*1e6))
    print('WG: sdvDivYt: %.4g urad' % (sdvDivYt*1e6))
    print('WG: max apert cent. cone X: ± %.4g urad' % (maxOpenAngleX*1e6))
    print('WG: max apert cent. cone Y: ± %.4g urad' % (maxOpenAngleY*1e6))

    print('WG: done!\n\n')






def printUndSourceSizeGaussianApprox(ring, phEnergy, lengthUnd):
    # SAME THAN LIC: THESIS. beta = 0

    [[sigRad, sigDivRad],
     _,
     [sigXt, sigDivXt],
     [sigYt, sigDivYt]] = calculateUndSourcePropertiesGaussian(ring, phEnergy,
                                                               lengthUnd)



    wgt.color_print('WG: Calculation Sig Source Size and Divergences at %.2f eV' %
                    (phEnergy))
    wgt.color_print('WG: It uses gaussian approximation (not sinc function)')

    wgt.color_print('WG: %s' % (ring.name))
    wgt.color_print('WG: Undulator Length %f' % (lengthUnd))

    print('WG: sigRad: %.4g um' % (sigRad*1e6))
    print('WG: sigDivRad: %.4g urad' % (sigDivRad*1e6))
    print('WG: sigXt: %.4g um' % (sigXt*1e6))
    print('WG: sigYt: %.4g um' % (sigYt*1e6))
    print('WG: sigDivXt: %.4g urad' % (sigDivXt*1e6))
    print('WG: sigDivYt: %.4g urad' % (sigDivYt*1e6))

    print('WG: done!\n\n')

####### Coherent Properties of the Source

def sourceCohProperties(phEnergy, ring, lengthUnd):
    '''
    Calculations based on the article by Vartanyants:
    "Coherence properties of hard x-ray synchrotron sources
    and x-ray free-electron lasers"
    '''

    k = 2*np.pi/hc*phEnergy

    [_, _, _, _, _, _,
     sdvXt, sdvDivXt,
     sdvYt, sdvDivYt,
     _, _] = calculateUndSourcePropertiesSDV(ring, phEnergy, lengthUnd, beta=0)

    sigEmmitanceXt = sdvXt*sdvDivXt
    sigEmmitanceYt = sdvYt*sdvDivYt

    qx = 2 / np.sqrt(4*k**2*sigEmmitanceXt**2-1)  # eq. 33 and 34
    qy = 2 / np.sqrt(4*k**2*sigEmmitanceYt**2-1)

    lCohx = qx*sdvXt  # eq. 33
    lCohy = qx*sdvYt

    zetax = 1/np.sqrt(1+4/qx**2)  # eq. 35
    zetay = 1/np.sqrt(1+4/qy**2)

    divX = 1/2./k/lCohx*np.sqrt(4+qx**2)  # eq. 25
    divY = 1/2./k/lCohy*np.sqrt(4+qy**2)

    divCohX = 1/2./k/sdvXt*np.sqrt(4+qx**2)  # eq. 29
    divCohY = 1/2./k/sdvYt*np.sqrt(4+qy**2)


    return [qx, qy, lCohx, lCohy, zetax, zetay, divX, divY, divCohX, divCohY]

def sourceCohPropertiesPropag(phEnergy, ring, lengthUnd, zPropag):
    '''
    Calculations based on the article by Vartanyants:
    "Coherence properties of hard x-ray synchrotron sources
    and x-ray free-electron lasers"
    '''

    [qx, qy, lCohx, lCohy, zetax, zetay, divX, divY, divCohX, divCohY] = \
                            sourceCohProperties(phEnergy, ring, lengthUnd)

    [_, _, _, _, _, _,
     sdvXt, sdvDivXt,
     sdvYt, sdvDivYt,
     _, _] = calculateUndSourcePropertiesSDV(ring, phEnergy, lengthUnd, beta=0)


    sizeX = np.sqrt(sdvXt**2 + divX**2*zPropag**2)  # eq. 24
    sizeY = np.sqrt(sdvYt**2 + divY**2*zPropag**2)

    sizeCohX = np.sqrt(lCohx**2 + divCohX**2*zPropag**2)  # eq. 28
    sizeCohY = np.sqrt(lCohy**2 + divCohY**2*zPropag**2)

    return [sizeX, sizeY, sizeCohX, sizeCohY]

def qParameter(phEnergy, ring, lengthUnd):
    '''Eq. 33 and 34 of Vartanyants 2010 article'''

    res = sourceCohProperties(phEnergy, ring, lengthUnd)
    return [res[0], res[1]]

def lCohSource(phEnergy, ring, lengthUnd):

    res = sourceCohProperties(phEnergy, ring, lengthUnd)
    return [res[2], res[3]]

def lCohVCZ(phEnergy, ring, lengthUnd, zPropag=10.00, beta=0):


    [_, _,
      fwhmXe, fwhmDivXe,
      fwhmYe, fwhmDivYe,
      fwhmXt, fwhmDivXt,
      fwhmYt, fwhmDivYt,
      _, _] = calculateUndSourcePropertiesFWHM(ring, phEnergy, lengthUnd, beta)

    wavelength = hc/phEnergy

    lCohx = .61*zPropag*wavelength*2.0/fwhmXt  # VCZ for plane circular aperture
    lCohy = .61*zPropag*wavelength*2.0/fwhmYt  # VCZ for plane circular aperture

    return [lCohx, lCohy, lCohx/zPropag/fwhmDivXt, lCohy/zPropag/fwhmDivYt]


def printLCohVCZ(phEnergy, ring, lengthUnd, zPropag=10.00, beta=0):


    [lCohx, lCohy,
     qx, qy] = lCohVCZ(phEnergy, ring, lengthUnd, zPropag, beta)

    print('\nWG: ### Coherent length %.2fm from the source, FWHM VALUES!' % (zPropag))
    print('WG: ### Based on VCZ theorem for plane wave illumintaion, circular scree.')
    print('WG: lCohx: %.2f um' % (lCohx*1e6))
    print('WG: lCohy: %.2f um' % (lCohy*1e6))
    print('WG: ### ratio lcoh/(beam size)')
    print('WG: qx: %.2f%% um' % (qx*100))
    print('WG: qy: %.2f%% um' % (qy*100))

def globalDegreeOfCoherence(phEnergy, ring, lengthUnd):
    '''
    global degree of coherence of a GSM source
    eq. 7 and 35 of Vartanyants 2010 article
    '''

    res = sourceCohProperties(phEnergy, ring, lengthUnd)
    return [res[4], res[5]]

def printCohProperties(phEnergy, ring, lengthUnd, zPropag=10.0):

    k = 2*np.pi/hc*phEnergy

    [qx, qy, lCohx, lCohy, zetax, zetay, divX, divY, divCohX, divCohY] = \
                                        sourceCohProperties(phEnergy, ring, lengthUnd)

    [sizePropX, sizePropY, sizeCohPropX, sizeCohPropY] = \
                            sourceCohPropertiesPropag(phEnergy, ring, lengthUnd, zPropag)


    print('\nWG: ### Coherent length at source, RMS VALUES!')
    print('WG: lCohx: %.3f um' % (lCohx*1e6))
    print('WG: lCohy: %.3f um' % (lCohy*1e6))
    print('WG: ### Coherent degre')
    print('WG: qx: %.2f%%' % (qx*100))
    print('WG: qy: %.2f%%' % (qy*100))
    print('WG: zetax: %.2f%%' % (zetax*100))
    print('WG: zetay: %.2f%%' % (zetay*100))
    print('WG: ### Divergence:')
    print('WG: divX: %.3f urad' % (divX*1e6))
    print('WG: divY: %.3f urad' % (divY*1e6))
    print('WG: divCohX: %.3f urad' % (divCohX*1e6))
    print('WG: divCohY: %.3f urad' % (divCohY*1e6))
    print('WG: ### Beam propagated by %.3f meters:' % (zPropag))
    print('WG: sizePropX: %.3f um' % (sizePropX*1e6))
    print('WG: sizePropY: %.3f um' % (sizePropY*1e6))
    print('WG: sizeCohPropX: %.3f um' % (sizeCohPropX*1e6))
    print('WG: sizeCohPropY: %.3f um' % (sizeCohPropY*1e6))

def fractionCoherFluxUnd(phEn_list, ring, undL):
    '''
    Simple relation between rad emmitance and electron emmitance
    '''
    beamPars = (np.asarray([calculateUndSourcePropertiesSDV(ring, 1e3, undL)
                    for phEn in phEn_list]))

    sdvXt = beamPars[:,2]
    sdvDivXt = beamPars[:,4]
    sdvYt = beamPars[:,6]
    sdvDivYt = beamPars[:,8]

    wavelength = hc/phEn_list

    return [(wavelength/4/np.pi)/(sdvXt*sdvDivXt),
            (wavelength/4/np.pi)/(sdvYt*sdvDivYt)]

class MAXIV3GeV(object):

    """Object with the Electron Beam Parameters for MAXIV3GeV"""

    def __init__(self):

        self.name = 'MAXIV 3GeV ring'
        #finite emmitance parameters for MAX-IV
        self.elecEnergy = 3.00e9
        self.gamma = self.elecEnergy/0.51099890221e6
        self.Iring = .5  # Average Current [A]
        self.sigEmmitanceX = 0.2630e-9  # Horizontal Emmitance [meters . rad]
        self.sigEmmitanceY = .030412*self.sigEmmitanceX
        # Vertical Emmitance [meters . rad]
        self.betaX = 9.00	 # Horizontal beta function value  [meters]
        self.betaY = 4.8  # Vertical beta function value  [meters]
        self.dispersionX = 0.0000

        self.sigEperE = 0.001000  # relative RMS energy spread
        self.sigX = (np.sqrt(self.sigEmmitanceX*self.betaX +
                     (self.dispersionX*self.sigEperE)**2))
                     # horizontal RMS size of e-beam [m]
        self.sigXp = np.sqrt(self.sigEmmitanceX/self.betaX)
        # horizontal RMS angular divergence [rad]
        self.sigY = np.sqrt(self.sigEmmitanceY*self.betaY)
        # vertical RMS size of e-beam [m]
        self.sigYp = np.sqrt(self.sigEmmitanceY/self.betaY)
        # vertical RMS angular divergence [rad]

class MAXIV1p5GeV(object):

    """Object with the Electron Beam Parameters for MAXIV3GeV"""

    def __init__(self):

        self.name = 'MAXIV 1.5GeV ring'
        #finite emmitance parameters for MAX-IV
        self.elecEnergy = 1.5000e9
        self.gamma = self.elecEnergy/0.51099890221e6
        self.Iring = .5  # Average Current [A]
        self.sigEmmitanceX = 6.00e-9  # Horizontal Emmitance [meters . rad]
        self.sigEmmitanceY = .01*self.sigEmmitanceX
        # Vertical Emmitance [meters . rad]
        self.betaX = 5.8	 # Horizontal beta function value  [meters]
        self.betaY = 3.00  # Vertical beta function value  [meters]
        self.dispersionX = 0.0000

        self.sigEperE = 0.0075000  # relative RMS energy spread
        self.sigX = (np.sqrt(self.sigEmmitanceX*self.betaX +
                     (self.dispersionX*self.sigEperE)**2))
                     # horizontal RMS size of e-beam [m]
        self.sigXp = np.sqrt(self.sigEmmitanceX/self.betaX)
        # horizontal RMS angular divergence [rad]
        self.sigY = np.sqrt(self.sigEmmitanceY*self.betaY)
        # vertical RMS size of e-beam [m]
        self.sigYp = np.sqrt(self.sigEmmitanceY/self.betaY)
        # vertical RMS angular divergence [rad]

class MAXII(object):

    """Object with the Electron Beam Parameters for MAXIV3GeV"""

    def __init__(self):

        self.name = 'MAXII ring'
        #finite emmitance parameters for MAX-II
        self.elecEnergy = 1.50e9
        self.gamma = self.elecEnergy/0.51099890221e6
        self.Iring = .5  # Average Current [A]
        self.sigEmmitanceX = 8.842e-9	 # Horizontal Emmitance [meters . rad]
        self.sigEmmitanceY = self.sigEmmitanceX*.1  # Vertical Emmitance [meters . rad]
        self.betaX = 13.00	 # Horizontal beta function value  [meters]
        self.betaY = 2.5  # Vertical beta function value  [meters]
        self.dispersionX = 0.05  # dispersion function

        self.sigEperE = 0.00200  # relative RMS energy spread
        self.sigX = (np.sqrt(self.sigEmmitanceX*self.betaX +
                     (self.dispersionX*self.sigEperE)**2))
                     # horizontal RMS size of e-beam [m]
        self.sigXp = np.sqrt(self.sigEmmitanceX/self.betaX)
        # horizontal RMS angular divergence [rad]
        self.sigY = np.sqrt(self.sigEmmitanceY*self.betaY)
        # vertical RMS size of e-beam [m]
        self.sigYp = np.sqrt(self.sigEmmitanceY/self.betaY)
        # vertical RMS angular divergence [rad]


def eBeam(ring):
    from srwlib import SRWLPartBeam #analysis:ignore

    ################################ Electron Beam ################################
    elecBeam = SRWLPartBeam()

    elecBeam.Iavg = ring.Iring # Average Current [A]
    elecBeam.partStatMom1.x = 0.0
    # Initial Transverse Coordinates (initial Longitudinal
    # Coordinate will be defined later on) [m]
    elecBeam.partStatMom1.y = 0.
    elecBeam.partStatMom1.z = 0.
    # Initial Longitudinal Coordinate (set before the ID)
    elecBeam.partStatMom1.xp = 0  # Initial Relative Transverse Velocities
    elecBeam.partStatMom1.yp = 0
    elecBeam.partStatMom1.gamma = ring.gamma  # Relative Energy

    #finite emmitance parameters for MAX-IV
    sigEmmitanceX = ring.sigEmmitanceX # Horizontal Emmitance [meters . rad]
    sigEmmitanceY = ring.sigEmmitanceY # Vertical Emmitance [meters . rad]
    sigEperE = ring.sigEperE  # relative RMS energy spread
    sigX = ring.sigX  # horizontal RMS size of e-beam [m]
    sigXp = sigEmmitanceX/sigX  # horizontal RMS angular divergence [rad]
    sigY = ring.sigY  # vertical RMS size of e-beam [m]
    sigYp = sigEmmitanceY/sigY  # vertical RMS angular divergence [rad]

    #2nd order stat. moments:
    elecBeam.arStatMom2[0] = sigX*sigX  # <(x-<x>)^2>
    elecBeam.arStatMom2[1] = 0  # <(x-<x>)(x'-<x'>)>
    elecBeam.arStatMom2[2] = sigXp*sigXp  # <(x'-<x'>)^2>
    elecBeam.arStatMom2[3] = sigY*sigY  # <(y-<y>)^2>
    elecBeam.arStatMom2[4] = 0  # <(y-<y>)(y'-<y'>)>
    elecBeam.arStatMom2[5] = sigYp*sigYp  # <(y'-<y'>)^2>
    elecBeam.arStatMom2[10] = sigEperE*sigEperE  # <(E-<E>)^2>/<E>^2

    return elecBeam


class PlanarUndAtRing(object):

    def __init__(self, und, ring, phEnergyTuneUnd, nHarmonic=1, name=None):

        from math import sqrt



        self.numPer = und.numPer
        self.period = und.period
        self.length = und.length

        self.name = name

        self.nHarmonic = nHarmonic
        self.phEnergyTuneUnd = phEnergyTuneUnd
        self.wavelenght1st = hc/phEnergyTuneUnd*nHarmonic
        # photon wavelenght of the first harmonic [m]

        self.Kx = 0.0  # Kparameter Horizontal
        self.Ky = sqrt(4.0*ring.gamma**2*self.wavelenght1st/self.period - 2.0)
        # Kparameter Vertical
        self.Bx = 0  # Peak Horizontal field [T]
        self.By = self.Ky/self.period/93.4  # Peak Vertical field [T]
        self.phBx = 0  # Initial Phase of the Horizontal field component
        self.phBy = 0  # Initial Phase of the Vertical field component
        self.sBx = -1  # Symmetry of the Horizontal field component vs Longitudinal position
        self.sBy = 1  # Symmetry of the Vertical field component vs Longitudinal position
        self.xcID = 0  # Transverse Coordinates of Undulator Center [m]
        self.ycID = 0
        self.zcID = 0  # Longitudinal Coordinate of Undulator Center [m]


class Undulator(object):

    def __init__(self, numPer=None, period=None, length=None, name=None):

        if numPer is None:
            numPer = length/period
        elif period is None:
            period = length/numPer
        elif length is None:
            length = numPer*period

        self.numPer = numPer
        self.period = period
        self.length = length

        self.name = name

class BendingMagnet(object):

    def __init__(self, fieldB=None, criticalEnergy=None, ringE=None, name=None):

        if fieldB is None:
            fieldB = criticalEnergy/(.665*ringE**2*1e-18)
        elif criticalEnergy is None:
            criticalEnergy = .665*ringE**2*1e-18*fieldB*1e3

        self.fieldB = fieldB
        self.criticalEnergy = criticalEnergy
        self.ringE = ringE

        self.name = name


def magFldCntUndulator(und):

    from srwlib import SRWLMagFldU, SRWLMagFldC #analysis:ignore

    undMagFld = SRWLMagFldU([SRWLMagFldH(1, 'v', und.By, und.phBy, und.sBy, 1),
                             SRWLMagFldH(1, 'h', und.Bx, und.phBx, und.sBx, 1)],
                            und.period, und.numPer)  # Ellipsoidal Undulator

    return SRWLMagFldC([undMagFld], array('d', [und.xcID]), array('d', [und.ycID]),
                       array('d', [und.zcID]))  # Container of all Field Elements


def printPlanUndPar(undAtRing):



    print('WG: Undulator tuned to %.4f eV ' % undAtRing.phEnergyTuneUnd)
    print('WG: Ky: %.4f' % undAtRing.Ky)
    print('WG: By: %.4f T' % undAtRing.By)
    print('WG: 1st harmonic wavelenght: %.4g m ' % undAtRing.wavelenght1st)
    print('WG: 1st harmonic energy: %.4f eV ' % (hc/undAtRing.wavelenght1st))


if __name__ == '__main__':

    ring = MAXIV3GeV()

    printUndSourceSizeFWHM(ring, 700.00, 3.766)
    printCohProperties(700.00, ring, 3.766, zPropag=10.0)

################
################
################
################
################
################
################
################


