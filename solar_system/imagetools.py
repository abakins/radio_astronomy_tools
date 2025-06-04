
import numpy as np
import scipy.constants as spc 
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS 
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astroquery.jplhorizons import Horizons
import spiceypy as spice


def read_fits_image(hdu, beam_hdu=None): 
    """ Extracts coordinate systems and quantities of interest 
        for input FITS images

        Inputs: 
        hdu: The primary header data unit, accessed via fits.open
        beam_hdu: Default None, if provided, it is used to extract beam information

        Returns: 
        Astropy SkyCoord, SpectralCoord, and StokesCoord grids describing the image 
        An array of synthesized beam quantities with shape len(SpectralCoord) x len(StokesCoord) x 3 (quantities BMAJ, BMIN, BPA in degrees)
        The observation date 
        
    """

    # Get SkyCoord which indexes the image spatial axes
    this_wcs = WCS(hdu.header)
    arraysize = this_wcs.celestial.array_shape
    l_index, m_index = np.mgrid[:arraysize[0], :arraysize[1]]
    skycoord = this_wcs.celestial.array_index_to_world(l_index, m_index)
    spatial_resolution = np.sqrt(this_wcs.celestial.proj_plane_pixel_area())  # Assume its a square pixel 
    
    # Get SpectralCoord which indexes the image spectral axis 
    arraysize = this_wcs.spectral.array_shape
    f_index = np.mgrid[:arraysize[0]]
    spectralcoord = this_wcs.spectral.array_index_to_world(f_index)
    frequency_resolution = this_wcs.spectral.proj_plane_pixel_scales()[0]

    # No easy way to get at the StokesCoord... 
    # Works like this: 1 - Find the Stokes axis, 2 - Get the world coordinate for 4 pixels of the Stokes axis
    # 3 - Truncate by the length of the axis (since it will otherwise just continue to iterate upwards...)
    axtype = this_wcs.get_axis_types()
    rmask = [True if x['coordinate_type'] == 'stokes' else False for x in axtype]
    template = [[0], [0], [0], [0]]
    template[rmask.index(True)] = [0, 1, 2, 3]
    qry = this_wcs.pixel_to_world(*template)
    stokescoord=None
    for i, q in enumerate(qry):
        if type(q) == astropy.coordinates.polarization.StokesCoord: 
            stokescoord = q[:this_wcs.pixel_shape[rmask.index(True)]]    

    obs_str = hdu.header['DATE-OBS']
    date = Time(obs_str, format='isot')

    # Get beams 
    # Shape is always Spectral X Stokes X [BMAJ, BMIN, BPA]
    if beam_hdu is not None: 
        column_data = beam_hdu.columns.info(output=False) 
        bmaj_index = column_data['name'].index('BMAJ')
        bmin_index = column_data['name'].index('BMIN')
        bpa_index = column_data['name'].index('BPA')
        chan_index = column_data['name'].index('CHAN')
        pol_index = column_data['name'].index('POL')
        beam_array = np.zeros((beam_hdu['NCHAN'], beam_hdu['NPOL'], 3)) * u.deg 
        for i, v in enumerate(beam_hdu.data): 
            beam_array[v[chan_index], v[pol_index], 0] = (v[bmaj_index] * u.Unit(column_data['unit'][bmaj_index])).to(u.deg)
            beam_array[v[chan_index], v[pol_index], 1] = (v[bmin_index] * u.Unit(column_data['unit'][bmin_index])).to(u.deg)
            beam_array[v[chan_index], v[pol_index], 1] = (v[bpa_index] * u.Unit(column_data['unit'][bpa_index])).to(u.deg)
    else: 
        # Assume beam definitions in primary header are in degrees 
        beam_array = np.zeros((1, 1, 3)) * u.deg 
        beam_array[0] = hdu.header['BMAJ'] * u.deg
        beam_array[1] = hdu.header['BMIN'] * u.deg
        beam_array[2] = hdu.header['BPA'] * u.deg

    return skycoord, spectralcoord, stokescoord, beam_array, date


def get_horizons_ephemerides(spice_id, obs_id, date, ellipsoid_radius, logger=None): 
    """ Built on the astroquery.jplhorizons module, queries Horizons for object information. 
        Also computes the apparent disk dimensions from the observer's perspective assuming 
        the body is an oblate spheroid. 

        Inputs: spice_id - SPICE ID or string name of the target, e.g. 799 for Uranus 
                obs_id - SPICE ID or string name of the observer, e.g. -5 for the VLA 
                date - Astropy date object
                spheroid_radius - A two-tuple of (equator, pole) radii (km) for an oblate spheroid

        Outputs: Dictionary of astropy quantities 

    """  

    output_dict = {}

    if logger: logger.info('##########EPHEMERIDES##########')

    r_e, r_p = ellipsoid_radius
    eph = Horizons(id=spice_id, location=obs_id, epochs=date.jd).ephemerides()

    output_dict['angular_diameter'] = eph['ang_width'].quantity[0]
    output_dict['north_pole_angle'] = eph['NPole_ang'].quantity[0] # Degrees counter-clockwise of ecliptic north
    output_dict['observer_target_distance'] = eph['delta'].quantity[0].to(u.km)
    output_dict['subobserver_latitude_PD'] = eph['PDObsLat'].quantity[0]
    output_dict['subobserver_latitude_PD'] = eph['PDObsLat'].quantity[0]
    b = r_e / r_p
    ecc = np.sqrt(1 - (1 / b)**2)

    if logger: 
        logger.info('At time of observation - ')
        logger.info('Angular diameter: {} arcsec'.format(output_dict['angular_diameter'].value))
        logger.info('North pole angle: {} deg'.format(output_dict['north_pole_angle'].value))
        logger.info('Distance to Earth: {} AU'.format(output_dict['observer_target_distance'].to(u.au).value))
        logger.info('Planet radius: {} (equatorial), {} (polar) km (b = {})'.format(r_e, r_p, b))
        logger.info('Sub-Observer Planetodetic Lat, Lon: {0}, {1} deg'.format(output_dict['subobserver_latitude_PD'].value, output_dict['subobserver_longitude_PD'].value))


    rec = spice.georec(output_dict['subobserver_longitude_PD'].to(u.rad).value, output_dict['subobserver_latitude_PD'].to(u.rad).value, 0, r_e, (r_e - r_p) / r_e)
    d, sublon_c, sublat_c = spice.reclat(rec)
    sublon_c = np.degrees(sublon_c)  # Planetocentric
    sublat_c = np.degrees(sublat_c)
    
    if logger: 
        logger.info('Sub-Observer Planetocentric Lat, Lon: {0}, {1} deg'.format(sublat_c, sublon_c))
        logger.info('Computing projected view ellipse')
    plane = spice.nvc2pl(rec, r_e)  # Normal vec to plane 
    ellps = spice.inedpl(r_e, r_e, r_p, plane)  # Intersection of plane and planet 
    center, smajor, sminor = spice.el2cgv(ellps)  # Center and generating vectors of ellipse 
    b_eff = spice.vnorm(smajor) / spice.vnorm(sminor)
    smajor = np.linalg.norm(smajor)
    sminor = np.linalg.norm(sminor)

    output_dict['equator_radius'] = r_e * u.km
    output_dict['pole_radius'] = r_p * u.km
    output_dict['apparent_equator_radius'] = smajor * u.km
    output_dict['apparent_pole_radius'] = sminor * u.km

    if logger: 
        logger.info('Apparent ellipse semimajor and semiminor axes: {}, {} km (b = {})'.format(smajor, sminor, b_eff))
        logger.info('#################################')

    return output_dict


def convolve_with_beam(image, bmaj, bmin, bpa, res): 
    """ Convolves image with a Gaussian beam using astropy 

        Inputs: 
        image: Image to convolve 
        bmaj: Synthesized beam semimajor half power width (same units as res) 
        bmin: Synthesized beam semiminor half power width (same units as res)
        bpa: Beam position angle in degrees
        res: Image resolution per pixel 

        Returns beam-convolved image 
    """

    ystd = (bmaj / res) / (2 * np.sqrt(2 * np.log(2)))
    xstd = (bmin / res) / (2 * np.sqrt(2 * np.log(2)))
    beam = Gaussian2DKernel(x_stddev=xstd, y_stddev=ystd, theta=np.radians(bpa))
    conv_image = convolve_fft(image, beam)
    return conv_image
