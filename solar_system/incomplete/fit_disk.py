# fit_disk.py - Utilities for fitting a limb-darkened disk to CASA MS visibilities 

import os
import numpy as np 
from numpy import sin, cos, sqrt, pi
from scipy.special import gamma, jv
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import pyfits as fits
import pickle


def limb_disk(Io, radius, u, v, phi, b, p):
    """ Generates a limb darkened disk in visibility-space 

    See Butler and Bastian Chapter on Solar System Objects 
    in Synthesis Imaging in Radio Astronomy II

    :param Io: Disk intensity in Jansky units 
    :param radius: disk radius in arcseconds
    :param u: U baseline coordinates in wavelengths
    :param v: V baseline coordinates in wavelengths
    :param phi: counterclockwise rotation off axis
    :param b: ratio of long to short axis of disk
    :param p: limb darkening parameter

    :return: Disk visibilities for input uv data  
    """

    phi = np.radians(phi)

    u_p = u * cos(phi) - v * sin(phi)
    v_p = b * (u * sin(phi) + v * cos(phi))
    radius = pi / (180 * 3600) * radius  # Convert radius to radians
    beta = radius * sqrt(u_p**2 + v_p**2)

    arg = 2 * pi * beta
    q = 1 + p / 2
    lambdafunc = gamma(q + 1) * (0.5 * arg)**(-q) * jv(q, arg)  # From Solar System Objects paper in NRAO Synth Imaging
    vis = Io * pi * radius**2 * lambdafunc

    return vis


def disk_cf(params, u, v, real, uniform=False, verbose=False): 
    """ Cost function to minimize for fitting disk to data
        See the function description for 'fit_disk'
        :param params: Tuple of disk intensity in Jy, disk radius in radians, and limb-darkening parameter
        :param u: U baselines
        :param v: V baselines
        :param real: Real part of visibilities
        :param uniform: Determines if visibilities are generated for uniform disk
                        as opposed to limb-darkened. Default is False 
        :param verbose: Toggles output of current cost function value. Default is False
    """
    Io = params[0]
    p = params[1]
    radius = params[2]
    if uniform:
        p = 0
    modelreal = limb_disk(Io, radius, u, v, 0, 1, p)
    out = np.sum((real - modelreal)**2)
    # No negative fluxes should result
    if Io < 0: 
        out += 1e20
    if verbose: 
        print(out)
    return out


def uniform_disk_image(outname, imagesize, cellsize, Io, radius): 
    """ Generates a uniform disk model in image space and writes
        the results as a 2D array in a blank FITS image. 
        Dev. note:  This may take some modification to be compatible
                    with CASA 
        
        :param imagesize: Image length in pixels (for a square image only)
        :param cellsize: Pixel length in arcseconds 
        :param Io: Disk intensity in Jy 
        :param radius: Disk radius in arcseconds
    """ 
    image = np.zeros((imagesize, imagesize))
    pixrad = pi / (180 * 3600) * cellsize
    radius = pi / (180 * 3600) * radius
    cellradius = radius / pixrad
    shape = np.shape(image)
    check = np.linspace(-shape[0] / 2, shape[0] / 2, shape[0])  # Assumes square image 
    c_lr, c_ud = np.meshgrid(check, check)
    rad = np.sqrt(c_lr**2 + c_ud**2)
    image[rad < cellradius] = Io * (pi / (180 * 3600) * cellsize)**2
    plt.imshow(image)
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    os.system('rm -rf %s.fits' % outname)
    hdul.writeto('%s.fits' % outname)
    hdul.close()

