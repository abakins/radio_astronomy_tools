# image_space.py - Utilities for image analysis 
# Must be called external to CASA (< 6.0/Python 3)

import numpy as np 
import scipy.constants as spc
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons


def find_center(fitsimage): 
    """ For a continuum, single-channel fits image of a planet, 
        find the disk center and radius 

        :param fitsimage: CASA-generated fits image
    """ 
    hdu = fits.open(fitsimage)
    image = hdu[0].data
    shape = np.shape(image)
    image = image.reshape(*shape[-2:])
    orig_image = image
    shape = np.shape(image)
    max_flux = np.max(image)
    image[image < (0.1 * max_flux)] = 0 
    image[image != 0] = 1 
    rows = np.sum(image, axis=1)
    u, c = np.unique(rows, return_counts=True)
    ydex = np.argmax(rows)
    ydex += np.floor(c[np.argmax(u)] / 2)
    cols = np.sum(image, axis=0)
    u, c = np.unique(cols, return_counts=True)
    xdex = np.argmax(cols)
    xdex += np.floor(c[np.argmax(u)] / 2)
    print('Pixel Center X,Y: {}, {}'.format(xdex, ydex))
    for i in range(shape[0]): 
        if (image[i, :] == 1).any(): 
            yr = i
            break 
    for i in range(shape[1]): 
        if (image[:, i] == 1).any(): 
            xr = i
            break
    meanr = 0.5 * (abs(xr - xdex) + abs(yr - ydex) )
    print('Mean Pixel Radius: {}'.format(meanr))
    print('tclean Mask String: circle[[{}pix,{}pix],{}pix])'.format(xdex, ydex, meanr))

    return image


def get_planetocentric_coords(fitsimage, date_obj, observer='VLA', target='Venus'): 

    # Date time conversion
    time = Time(dmy_string, format='isot')
    if observer == 'VLA': 
        location = '-5'
    elif observer == 'ALMA': 
        location = '-7'
    else: 
        location = 'Earth'

    if target == 'Venus': 
        ident = '299'
    elif target == 'Uranus': 
        ident = '799'

    obj = Horizons(id=ident, location=location, epochs=time.jd, id_type='majorbody')
    htable = obj.ephemerides()
    distance = htable['r'] * spc.au
    solon = htable['PDObsLon'] 
    solat = htable['PDObsLat']

    # Use scipy routines to rotate







