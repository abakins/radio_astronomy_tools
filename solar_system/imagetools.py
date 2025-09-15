
import numpy as np
import pandas as pd 
import scipy.constants as spc 
import scipy.optimize as spo 
import scipy.interpolate as spi 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS 
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.stats import sigma_clipped_stats
from astroquery.jplhorizons import Horizons
import cartopy.crs as ccrs 
import spiceypy as spice
from tqdm.auto import tqdm, trange
from radio_beam import Beam, Beams

import pdb

def read_fits_image(hdu, beam_hdu=None): 
    """ Extracts coordinate systems and quantities of interest 
        for input FITS images

        Inputs: 
        hdu: The primary header data unit, accessed via fits.open
        beam_hdu: Default None, if provided, it is used to extract beam information

        Returns: 
        Astropy World Coordinate System
        Tuple of intermediate projection coordinates (l, m)
        Astropy SkyCoord ICRS, SpectralCoord, and StokesCoord grids describing the image 
        A radio_beam Beam or Beams object describing the synthesized beam
        The observation date as an Astropy Date
        
    """

    this_wcs = WCS(hdu.header)

    # Define intermediate projection coordinates - origin at phase center 
    arrayshape = this_wcs.celestial.array_shape  # dec, ra
    y_index, x_index = np.mgrid[:arrayshape[0], :arrayshape[1]]
    x_shift = x_index - hdu.header['CRPIX1']
    y_shift = y_index - hdu.header['CRPIX2']
    x_grid = x_shift * abs(hdu.header['CDELT1'])
    y_grid = y_shift * abs(hdu.header['CDELT2'])
    projcoord = (x_grid * getattr(u, hdu.header['CUNIT1']), y_grid * getattr(u, hdu.header['CUNIT2']))

    # Get SkyCoord on the celestial sphere 
    i_index, j_index = np.mgrid[:arrayshape[0], :arrayshape[1]]
    skycoord = this_wcs.celestial.array_index_to_world(i_index, j_index).icrs
    
    # Get SpectralCoord which indexes the image spectral axis 
    arrayshape = this_wcs.spectral.array_shape
    f_index = np.mgrid[:arrayshape[0]]
    spectralcoord = this_wcs.spectral.array_index_to_world(f_index)

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
    if beam_hdu is not None: 
        synth_beam = Beams.from_fits_bintable(beam_hdu)
    else: 
        try: 
            synth_beam = Beam.from_fits_header(hdu.header)
        except: 
            print('Trying AIPS history ')    
            try: 
                synth_beam = Beam.from_fits_history(hdu.header)
            except:
                print('Warning: Beam information not found')
                synth_beam = None

    output_dict = {'WCS': this_wcs, 
                   'projection_coordinates': projcoord,
                   'sky_coordinates': skycoord, 
                   'spectral_coordinates': spectralcoord, 
                   'stokes_coordinates': stokescoord, 
                   'synthesized_beam': synth_beam, 
                   'observation_date': date }
    return output_dict


def get_horizons_ephemerides(spice_id, obs_id, date, spheroid_radius): 
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

    r_e, r_p = spheroid_radius
    eph = Horizons(id=spice_id, location=obs_id, epochs=date.jd).ephemerides()

    output_dict['right_ascension'] = eph['RA'].quantity[0]
    output_dict['declination'] = eph['DEC'].quantity[0]
    output_dict['angular_diameter'] = eph['ang_width'].quantity[0].to(u.deg)
    output_dict['north_pole_angle'] = eph['NPole_ang'].quantity[0]
    output_dict['observer_target_distance'] = eph['delta'].quantity[0].to(u.km)
    output_dict['sun_target_distance'] = eph['r'].quantity[0].to(u.km)
    output_dict['subobserver_latitude_PD'] = eph['PDObsLat'].quantity[0]
    output_dict['subobserver_longitude_PD'] = eph['PDObsLon'].quantity[0]
    b = r_e / r_p
    # ecc = np.sqrt(1 - (1 / b)**2)

    rec = spice.georec(output_dict['subobserver_longitude_PD'].to(u.rad).value, output_dict['subobserver_latitude_PD'].to(u.rad).value, 0, r_e, (r_e - r_p) / r_e)
    _, sublon_c, sublat_c = spice.reclat(rec)
    sublon_c = np.degrees(sublon_c)  # Planetocentric
    sublat_c = np.degrees(sublat_c)
    output_dict['subobserver_latitude_PC'] = sublat_c * u.deg
    output_dict['subobserver_longitude_PC'] = output_dict['subobserver_longitude_PD']
    
    plane = spice.nvc2pl(rec, r_e)  # Normal vec to plane 
    ellps = spice.inedpl(r_e, r_e, r_p, plane)  # Intersection of plane and planet 
    _, smajor, sminor = spice.el2cgv(ellps)  # Center and generating vectors of ellipse 
    # b_eff = spice.vnorm(smajor) / spice.vnorm(sminor)
    smajor = np.linalg.norm(smajor)
    sminor = np.linalg.norm(sminor)

    output_dict['equator_radius'] = r_e * u.km
    output_dict['pole_radius'] = r_p * u.km
    output_dict['apparent_equator_radius'] = smajor * u.km
    output_dict['apparent_pole_radius'] = sminor * u.km

    return output_dict

def map_sky_projection(projcoord, ghe_dict): 
    """ Computes mapping from sky coordinates to planetodetic latitude and longitude for an 
        oblate spheroid. The mapping is accomplished using relative angular coordinates
    
        Inputs: 
        projcoord: Sky-plane relative coordinate arrays (l, m)
        ghe_dict: Output dictionary for get_horizons_ephemerides
        
        Returns: 
        Planetodetic longitude and latitude coordinates 
        Surface emission angle  
        
    """
    subobs_pc = (ghe_dict['subobserver_longitude_PC'], ghe_dict['subobserver_latitude_PC'])
    ang_diam = ghe_dict['angular_diameter']
    np_angle = ghe_dict['north_pole_angle']
    
    # Extract from Astropy coordinates  
    l = projcoord[0].to(u.rad).value
    m = projcoord[1].to(u.rad).value
    sublon_c, sublat_c = subobs_pc
    sublon_c = sublon_c.to(u.deg).value
    sublat_c = sublat_c.to(u.deg).value
    ang_diam = ang_diam.to(u.deg).value 
    np_angle = np_angle.to(u.deg).value 

    r_e = ghe_dict['equator_radius'].value
    r_p = ghe_dict['pole_radius'].value
    rad_ang_radius = np.radians(ang_diam) / 2 
    b = r_e / r_p 
    ecc = np.sqrt(1 - (1 / b)**2)
    rad_np_angle = np.radians(np_angle)

    # Compensate axes for rotation
    lr = l * np.cos(rad_np_angle) + m * np.sin(rad_np_angle)
    mr = m * np.cos(rad_np_angle) - l * np.sin(rad_np_angle)

    phi0 = np.radians(sublat_c)
    theta0 = np.radians(sublon_c)

    # Solve quadratic, see Butler and Bastian 1999
    A = np.cos(phi0)**2 + b**2 * np.sin(phi0)**2
    B = 2 * mr * np.cos(phi0) * np.sin(phi0) * (b**2 - 1)
    C = lr**2 + mr**2 * (np.sin(phi0)**2 + b**2 * np.cos(phi0)**2) - rad_ang_radius**2
    coefficients = np.vstack([A * np.ones(C.shape).ravel(), B.ravel(), C.ravel()]).T 
    # Vectorized roots compute 
    d = coefficients[:, 1:-1]**2 - 4.0 * coefficients[:, ::2].prod(axis=1, keepdims=True)
    roots = -0.5 * (coefficients[:, 1:-1] + [1, -1] * np.emath.sqrt(d)) / coefficients[:, :1]
    n = np.max(roots, axis=1).reshape(l.shape)
    mask = np.imag(n) != 0 
    n[mask] = 0 
    n = np.real(n)

    # Alternate to Eq. 31-12 - 31-14 in BB99 
    rotation = spice.rotate(theta0, 2) @ spice.rotate(phi0, 1)
    lmn = np.stack([lr, mr, n])
    lpmpnp = np.einsum('ij,j...->i...', rotation, lmn)
    lp = lpmpnp[0]
    mp = lpmpnp[1]
    nps = lpmpnp[2]

    # Compute PC longitude/latitude
    Rp = np.sqrt(lp**2 + mp**2 + nps**2)
    pc_grid_lats = np.arcsin(mp / Rp).astype(float)
    # Warning: Confirm pc_grid_lons convention
    # Either one of the below 
    # pc_grid_lons = -np.arctan2(np.real(nps), np.real(lp))  # Consistent w/ Cartopy? 
    pc_grid_lons = np.arctan2(-np.real(lp), np.real(nps))

    pc_grid_lats[mask] = np.nan
    pc_grid_lons[mask] = np.nan

    # Converting to planetographic using spice 
    # Slow now, but will refactor once cyice comes around... 
    pd_grid_lats = pc_grid_lats.copy() 
    pd_grid_lons = pc_grid_lons.copy() 
    pdlar = pd_grid_lats.ravel() 
    pdlor = pd_grid_lons.ravel() 
    for i in trange(len(pdlar)): 
        if not np.isnan(pdlar[i]): 
            recr = spice.latrec(r_e, pdlor[i], pdlar[i])
            geotuple = spice.recgeo(recr, r_e, (r_e - r_p) / r_e)
            pdlar[i] = geotuple[1]
            pdlor[i] = geotuple[0]

    
    # # Converting to planetographic
    # # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    # This implementation isn't right...  
    # N = r_e / np.sqrt(1 - ecc**2 * np.sin(pc_grid_lats)**2)
    # X = N * np.cos(pc_grid_lats) * np.cos(pc_grid_lons)
    # Y = N * np.cos(pc_grid_lats) * np.sin(pc_grid_lons)
    # Z = (1-ecc**2) * N * np.sin(pc_grid_lats)
    # p = np.sqrt(X**2 + Y**2)

    # pd_grid_lats = pc_grid_lats.copy()
    # ref_lat = np.zeros(pd_grid_lats.shape) 

    # # Convert all points 
    # while not np.allclose(pd_grid_lats, ref_lat, equal_nan=True): 
    #     ref_lat = pd_grid_lats 
    #     N = r_e / np.sqrt(1 - ecc**2 * np.sin(pd_grid_lats)**2)
    #     alt = p / np.cos(pd_grid_lats) - N
    #     print(np.nanmax(alt))
    #     X = (N+alt) * np.cos(pc_grid_lats) * np.cos(pc_grid_lons)
    #     Y = (N+alt) * np.cos(pc_grid_lats) * np.sin(pc_grid_lons)
    #     Z = ((1-ecc**2) * N + alt) * np.sin(pc_grid_lats)
    #     p = np.sqrt(X**2 + Y**2)
    #     pd_grid_lats = np.arctan(Z / p / (1 - ecc**2 * N / (N + alt)))

    # pd_grid_lons = pc_grid_lons    

    # Only the sub-point, using spice 
    recr = spice.latrec(r_e, theta0, phi0)
    geotuple = spice.recgeo(recr, r_e, (r_e - r_p) / r_e)
    phi0_pd = geotuple[1]
    theta0_pd = geotuple[0]


    cos_emang = np.sin(phi0_pd) * np.sin(pd_grid_lats) + np.cos(phi0_pd) * np.cos(pd_grid_lats) * np.cos(pd_grid_lons - theta0)
    emang = np.degrees(np.arccos(cos_emang))
    pc_grid_lons = np.degrees(pc_grid_lons)    
    pd_grid_lats = np.degrees(pd_grid_lats)
    pc_grid_lats = np.degrees(pc_grid_lats)

    output_dict = {'planetocentric_longitude': pc_grid_lons * u.deg, 
                   'planetocentric_latitude': pc_grid_lats * u.deg, 
                   'planetodetic_latitude': pd_grid_lats * u.deg,
                   'surface_incidence_angle': emang * u.deg}
    return output_dict


def map_sky_projection_crs(projcoord, ghe_dict): 
    """ An alternate to map_sky_projection which leverages Cartopy 
        
    Inputs: 
    projcoord: Sky-plane relative coordinate arrays (l, m)
    ghe_dict: Output dictionary for get_horizons_ephemerides
    
    Returns: 
    Planetodetic longitude and latitude coordinates 
    Surface emission angle  
        
    """
    raise NotImplementedError
    # Orthographic finally supports a polar angle rotation parameter 
    # But still doesn't incorporate elliptical globes... 

    subobs_pc = (ghe_dict['subobserver_longitude_PC'], ghe_dict['subobserver_latitude_PC'])
    np_angle = ghe_dict['north_pole_angle']
    dist = ghe_dict['observer_target_distance']
    r_e = ghe_dict['equator_radius'].value  # km
    r_p = ghe_dict['pole_radius'].value  # km 
    flattening = (r_e - r_p) / r_e
    globe = ccrs.Globe(semimajor_axis=r_e, semiminor_axis=r_p, flattening=flattening, inverse_flattening=1/flattening) 
    ortho = ccrs.Orthographic(central_longitude=subobs_pc[0], central_latitude=subobs_pc[1], azimuth=np_angle, globe=globe) 
    plcar = ccrs.PlateCarree(globe=globe) 
    
    # True planar_distance 
    # Assume l and m are true angles 
    l = projcoord[0].to(u.rad).value
    m = projcoord[1].to(u.rad).value
    x = dist * np.tan(l) 
    y = dist * np.tan(m) 
    plcar.transform_points(ortho, x, y)

    





def solve_image_center(image, model, wcs, synthesized_beam, manual_align=False): 
    """ Solves for image center based on comparison with a reference model and returns a centered image 

        Inputs: 
        image: Image to center 
        model: Model to use as centering reference 
        wcs: Image WCS
        synthesized_beam: radio_beam Beam object 
        manual_align: If True (default False), enable an interactive loop to tune image center 

    """ 

    # Normalize image and model to unit sum 
    use_image = image / np.nansum(image)
    use_model = model / np.nansum(model)

    imagesize = use_image.shape
    m_index, l_index = np.mgrid[:imagesize[0], :imagesize[1]]

    res = sum(wcs.proj_plane_pixel_scales()[:2]) / 2  
    if imagesize[0] != wcs.celestial.pixel_shape[0]: 
        # It's resampled, assuming the same in both directions
        res = res * (wcs.celestial.pixel_shape[0] / imagesize[0])

    beam_kernel = synthesized_beam.as_kernel(res)
    use_model = convolve_fft(use_model, beam_kernel)
    def cost_func(x, true_image, model_image): 
        xr = int(x[0])
        yr = int(x[1])
        xrim = np.roll(true_image, xr, axis=1)
        xyrim = np.roll(xrim, yr, axis=0)
        diff = (model_image - xyrim)**2
        diff = diff.flatten()
        diffsum = np.sum(diff)
        return diffsum

    args = (use_image, use_model)
    
    out = spo.minimize(cost_func, [0, 0], args=args, method='Powell')  # Powell is better than others at searching the space 
    xr = int(out.x[0])
    yr = int(out.x[1])

    xrim = np.roll(use_image, xr, axis=1)
    centered_image = np.roll(xrim, yr, axis=0)

    if manual_align: 
        ok_x = 'n'
        ok_y = 'n'
        done = False 

        while not done: 
            xrim = np.roll(use_image, xr, axis=1)
            centered_image = np.roll(xrim, yr, axis=0)
            minus_disk = centered_image - use_model
            plt.figure()
            plt.ion()
            plt.show(block=False)

            im = plt.pcolormesh(l_index, m_index, minus_disk, cmap='viridis', shading='auto')
            plt.colorbar()
            plt.draw()
            plt.pause(0.001)
            plt.draw()
            plt.pause(0.001)

            print('Current offsets - X: {}, Y: {}'.format(xr, yr))
            ok_x = input('Done X? ')
            if ok_x == 'n': 
                xr = int(input('X offset: '))
            ok_y = input('Done Y? ')
            if ok_y == 'n': 
                yr = int(input('Y offset: '))
            if (ok_x != 'n') and (ok_y != 'n'): 
                done = True
            plt.ioff() 
            plt.close()

        xrim = np.roll(use_image, xr, axis=1)
        centered_image = np.roll(xrim, yr, axis=0)
    

    difference_image = centered_image - use_model  # Not flux normalized 

    # Restore flux 
    centered_image = centered_image * np.nansum(image) / np.nansum(centered_image)

    output_dict = {'x_roll': xr, 
                   'y_roll': yr, 
                   'difference_image': difference_image}
    
    return centered_image, output_dict


def brightness_temperature_stats(jy_beam_image, rfi_dict, ghe_dict, mask=None):
    """ Compute image brightness temperature statistics 
        Input: 
        jy_beam_image
        skycoord: Similar to output from read_fits_image
        spectralcoord: Similar to output from read_fits_image
        beam_array: Similar to output from read_fits_image
        ephem_dict: Output from get_horizons_ephemerides
        mask: An optional mask, for only summing brightness over a nominal range. 

    """
    
    # Compute zero point bias 
    mean, median, std = sigma_clipped_stats(jy_beam_image, sigma=3.0, maxiters=20)
    zero_point_bias = mean 
    
    wcs = rfi_dict['WCS']
    beam = rfi_dict['synthesized_beam']
    frequency = rfi_dict['spectral_coordinates'].value
    km_dist = ghe_dict['observer_target_distance'].value
    smajor = ghe_dict['apparent_equator_radius'].value
    sminor = ghe_dict['apparent_pole_radius'].value

    tb_image = planck_function(frequency, (jy_beam_image / beam.sr).value) + 2.726
    beam_per_px = wcs.proj_plane_pixel_area().to(u.sr) / beam.sr
    jpp = jy_beam_image * beam_per_px
    average_tb = planck_function(frequency, np.sum(jpp)) * km_dist**2 / (np.pi * smajor * sminor) + 2.726
    return tb_image, average_tb

def spatial_binning(image, wcs, synthesized_beam, msp_dict, bin_resolution=1, cutoff_angle=80.): 
    """ Image per-pixel and per-beam binning by latitude, longitude, emission angle

        Inputs: 
        image - Image array
        wcs - Image WCS
        synthesized beam -  radio_beam Beam object
        msp_dict - Quantities from map_sky_projection
        bin_resolution - Latitude and incidence angle resolution for binning (in degrees)
        
        Returns binned statistics 

    """

    latitude = msp_dict['planetodetic_latitude'].value
    longitude = msp_dict['planetocentric_longitude'].value
    emission_angle = msp_dict['surface_incidence_angle'].value
    cos_emang = np.cos(np.radians(emission_angle))

    res = sum(wcs.proj_plane_pixel_scales()[:2]) / 2  
    beam = synthesized_beam.as_kernel(res)

    beam_mask = np.zeros(beam.array.shape, dtype=bool)
    # Choose the half power metric 
    beam_mask[(beam.array / np.max(beam.array)) >= 0.5] = True
    half = np.ceil(len(beam_mask) / 2).astype(int)  # Assume beam is square
    mask = ~np.isnan(latitude)
    use_image = np.squeeze(image.copy())
    use_image[~mask] = np.nan
    image_coords = np.argwhere(mask)

    
    median_im = [] 
    center_im = [] 
    min_im = [] 
    max_im = []
    median_lat = [] 
    center_lat = [] 
    min_lat = []  
    max_lat = [] 
    median_lon = [] 
    center_lon = [] 
    min_lon = []  
    max_lon = [] 
    median_cosemang = [] 
    center_cosemang = [] 
    min_cosemang = [] 
    max_cosemang = []

    # Iterate through the image
    for i, ic in enumerate(tqdm(image_coords)): 
        xcoords = (ic[0] - half, ic[0] + half - 1)
        ycoords = (ic[1] - half, ic[1] + half - 1)
        xmid = (ic[0] - 1, ic[0] + 1)
        ymid = (ic[1] - 1, ic[1] + 1)
        sub_image = use_image[slice(*xcoords), slice(*ycoords)][beam_mask]
        sub_lat_image = latitude[slice(*xcoords), slice(*ycoords)][beam_mask]
        sub_lon_image = longitude[slice(*xcoords), slice(*ycoords)][beam_mask]
        sub_cos_emang_image = cos_emang[slice(*xcoords), slice(*ycoords)][beam_mask]

        # center_im.append(np.nanmean(use_image[slice(*xmid), slice(*ymid)]))
        center_im.append(use_image[tuple(ic)])   
        median_im.append(np.nanmedian(sub_image))
        min_im.append(np.nanmin(sub_image))
        max_im.append(np.nanmax(sub_image))
        # center_lat.append(np.nanmean(latitude[slice(*xmid), slice(*ymid)]))
        center_lat.append(latitude[tuple(ic)])
        median_lat.append(np.nanmedian(sub_lat_image))
        min_lat.append(np.nanmin(sub_lat_image))
        max_lat.append(np.nanmax(sub_lat_image))
        # center_lon.append(np.nanmean(longitude[slice(*xmid), slice(*ymid)]))
        center_lon.append(longitude[tuple(ic)])
        median_lon.append(np.nanmedian(sub_lon_image))
        min_lon.append(np.nanmin(sub_lon_image))
        max_lon.append(np.nanmax(sub_lon_image))
        # center_cosemang.append(np.nanmean(cos_emang[slice(*xmid), slice(*ymid)]))
        center_cosemang.append(cos_emang[tuple(ic)])
        median_cosemang.append(np.nanmedian(sub_cos_emang_image))
        min_cosemang.append(np.nanmin(sub_cos_emang_image))
        max_cosemang.append(np.nanmax(sub_cos_emang_image))

    binned_array = np.vstack([np.array(median_im), 
                             np.array(center_im), 
                             np.array(min_im), 
                             np.array(max_im), 
                             np.array(median_lat),
                             np.array(center_lat),
                             np.array(min_lat), 
                             np.array(max_lat), 
                             np.array(median_lon),
                             np.array(center_lon),
                             np.array(min_lon), 
                             np.array(max_lon), 
                             np.degrees(np.arccos(np.array(median_cosemang))), 
                             np.degrees(np.arccos(np.array(center_cosemang))), 
                             np.degrees(np.arccos(np.array(max_cosemang))), 
                             np.degrees(np.arccos(np.array(min_cosemang)))]).T 
    
        # Shape into pandas frame 
    binned_columns = ['Median image', 'Center image', 'Min image', 'Max image', 
                     'Median latitude', 'Center latitude', 'Min latitude', 'Max latitude', 
                     'Median longitude', 'Center longitude', 'Min longitude', 'Max longitude', 
                     'Median angle', 'Center angle', 'Min angle', 'Max angle']

    binned_frame = pd.DataFrame(binned_array, columns=binned_columns).dropna()

    # First, high resolution
    stats_frame = binned_frame.copy()
    lat_grid = np.arange(-90.5, 90.5 + bin_resolution, bin_resolution)
    angle_grid = np.arange(0, 90 + bin_resolution, bin_resolution)

    stats_frame['Center latitude'] = pd.cut(stats_frame['Center latitude'], lat_grid).apply(lambda x: x.mid.astype(float)).astype(float)
    stats_frame['Center angle'] = pd.cut(stats_frame['Center angle'], angle_grid).apply(lambda x: x.mid.astype(float)).astype(float)
    sfgb = stats_frame.groupby(['Center latitude', 'Center angle']).mean()
    sfgb_beamcount = stats_frame.groupby(['Center latitude', 'Center angle']).count() / (np.sum(np.ravel(beam_mask)))
    sfgb = sfgb.reset_index()
    sfgb_beamcount = sfgb_beamcount.reset_index()
    mask = sfgb['Center angle'] <= cutoff_angle
    sfgb[~mask] = np.nan
    sfgb_beamcount[~mask] = np.nan
    sfgb = sfgb.set_index(['Center latitude', 'Center angle']).dropna()
    sfgb_beamcount = sfgb_beamcount.set_index(['Center latitude', 'Center angle']).dropna()

    # Compute the equivalent image latitude/emission angle resolution and error-per-beam  
    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=20)  # Usually works well for planets... 
    sfgb['Beam count'] = sfgb_beamcount['Median image'] # Its a count, can be any key... 
    sfgb['Error estimate'] = np.sqrt(std**2 / sfgb['Beam count'])
    sfgb_hires = sfgb.copy() # Save the high-res for later 

    delta_lat_res = np.round(np.nanmedian((sfgb['Max latitude'] - sfgb['Min latitude']).values.ravel())).astype(int)
    delta_angle_res = np.round(np.nanmedian((sfgb['Max angle'] - sfgb['Min angle']).values.ravel())).astype(int)

    # Next, at image resolution 
    stats_frame = binned_frame.copy()
    lat_grid = np.arange(-90.5, 90.5 + delta_lat_res, delta_lat_res)
    angle_grid = np.arange(0, 90 + delta_angle_res, delta_angle_res)

    stats_frame['Center latitude'] = pd.cut(stats_frame['Center latitude'], lat_grid).apply(lambda x: x.mid.astype(float)).astype(float)
    stats_frame['Center angle'] = pd.cut(stats_frame['Center angle'], angle_grid).apply(lambda x: x.mid.astype(float)).astype(float)
    sfgb = stats_frame.groupby(['Center latitude', 'Center angle']).mean()
    sfgb_beamcount = stats_frame.groupby(['Center latitude', 'Center angle']).count() / (np.sum(np.ravel(beam_mask)))

    sfgb = sfgb.reset_index()
    sfgb_beamcount = sfgb_beamcount.reset_index()
    mask = sfgb['Center angle'] <= cutoff_angle
    sfgb[~mask] = np.nan
    sfgb_beamcount[~mask] = np.nan
    sfgb = sfgb.set_index(['Center latitude', 'Center angle']).dropna()
    sfgb_beamcount = sfgb_beamcount.set_index(['Center latitude', 'Center angle']).dropna()

    sfgb['Beam count'] = sfgb_beamcount['Median image']
    sfgb['Error estimate'] = np.sqrt(std**2 / sfgb['Beam count'])

    return sfgb_hires, sfgb 


def fit_binned_limbdarken(stats_frame, min_angle_range=10., plots=False, use_errors=True): 

    def limb_dark_cost(angle, Io, p): 
        # Return and fix, check the Wikipedia for limb-darkening
        return Io * np.cos(np.radians(angle))**p 
    
    data_frame = stats_frame['Center image'].unstack() 
    error_frame = stats_frame['Error estimate'].unstack() 

    if plots: 
        fig, axs = plt.subplots(1, 2)
        angcolors = plt.cm.Blues(np.linspace(0.25, 1, len(data_frame.index)))
        ldcolors = plt.cm.Greens(np.linspace(0.25, 1, len(data_frame.index)))

    angle_res = np.mean(np.gradient(data_frame.columns.values))

    # Fit at image resolution
    Io_list = [] 
    Io_std_list = [] 
    p_list = [] 
    p_std_list = [] 

    for i, ind in enumerate(tqdm(data_frame.index)): 
        intens = data_frame.loc[ind].values
        sigma_intens = error_frame.loc[ind].values 
        angles = data_frame.loc[ind].index.values
        mask = ~np.isnan(intens)
        intens = intens[mask]
        angles = angles[mask]
        sigma_intens = sigma_intens[mask]

        if len(intens) > np.ceil(min_angle_range / angle_res).astype(int): 
            if use_errors: 
                out, cov, *_ = spo.curve_fit(limb_dark_cost, angles, intens, check_finite=True, p0=[1, 0.05], sigma=sigma_intens, absolute_sigma=True, bounds=(0, np.inf), max_nfev=1e6, x_scale='jac')
            else: 
                out, cov, *_ = spo.curve_fit(limb_dark_cost, angles, intens, check_finite=True, p0=[1, 0.05], bounds=(0, np.inf), x_scale='jac', max_nfev=1e6)
            Io_list.append(out[0])
            Io_std_list.append(np.sqrt(cov[0, 0]))
            p_list.append(out[1])
            p_std_list.append(np.sqrt(cov[1, 1]))
            if plots: 
                axs[0].plot(angles, limb_dark_cost(angles, out[0], out[1]), color=angcolors[i])
                axs[0].plot(angles, intens, color=ldcolors[i])
                axs[1].plot(angles, intens - limb_dark_cost(angles, out[0], out[1]), color=angcolors[i])
        else: 
            Io_list.append(np.nan)
            Io_std_list.append(np.nan)
            p_list.append(np.nan)
            p_std_list.append(np.nan)
     
    lat_array = data_frame.index.values
    Io_array = np.array(Io_list)
    Io_std_array = np.array(Io_std_list)
    p_array = np.array(p_list)
    p_std_array = np.array(p_std_list)

    output_dict = {'latitude_bins': lat_array, 
                   'nadir_intensity': Io_array,
                   'limb_darkening_coeff': p_array, 
                   'intensity_error': Io_std_array, 
                   'limb_darkening_error': p_std_array}

    if plots: 
        axs[0].set_xlabel('Emission angle (degrees)')
        axs[0].set_ylabel('Intensity')
        axs[1].set_xlabel('Emission angle (degrees)')
        axs[1].set_ylabel('Difference from model')
        fig.tight_layout()

    return output_dict

# Rationale for this function 
# See Oyafuso et al. 2020 r.e. limb-darkening and giant planets 
# Their approach is model based. I'm trying to avoid that. 
# Half integer and integer powers of cos(angle) don't work well, or at least give divergent answers 
# After some experimentation, I'm going with the following form: 
# TB(theta) = A * (exp(B / cos(theta)^C) + exp(D / cos(theta)^E) + ... 
# This is based on the structure of the microwave radiative transfer equation. 

def fit_binned_limbdarken_exps(stats_frame, n_exps=2, min_angle_range=10., plots=False, use_errors=True): 
    # Number of terms (n_exps) is the number of exponentials above. Two should work fine

    def limb_dark_cost(angval, *p):
        rv = 0.
        for i in range(1, len(p), 2): 
            rv = rv + p[0] * np.exp(p[i] / np.cos(np.radians(angval))**p[i+1])
        return rv 

    def ldwrap(p, angle): 
        return limb_dark_cost(angle, *p)

    data_frame = stats_frame['Center image'].unstack() 
    error_frame = stats_frame['Error estimate'].unstack() 

    if plots: 
        fig, axs = plt.subplots(1, 2)
        angcolors = plt.cm.Blues(np.linspace(0.25, 1, len(data_frame.index)))
        ldcolors = plt.cm.Greens(np.linspace(0.25, 1, len(data_frame.index)))

    angle_res = np.mean(np.gradient(data_frame.columns.values))

    nadir_list = []
    nadir_std_list = []  
    coefs_list = [] 
    covar_list = [] 

    for i, ind in enumerate(tqdm(data_frame.index)): 
        intens = data_frame.loc[ind].values
        sigma_intens = error_frame.loc[ind].values 
        angles = data_frame.loc[ind].index.values
        mask = ~np.isnan(intens)
        intens = intens[mask]
        angles = angles[mask]
        sigma_intens = sigma_intens[mask]

        if len(intens) > np.ceil(min_angle_range / angle_res).astype(int): 

            pguess = np.concatenate([[-0.2, 0.2] for x in range(n_exps)])
            pguess = np.insert(pguess, 0, np.nanmax(data_frame.values))  # Seed a rough estimate of the nadir brightness 
            bnds = [(0, np.inf)]
            for j in range(n_exps): 
                bnds = bnds + [(-np.inf, 0), (0, 5)]
            bnds = ([x[0] for x in bnds], [x[1] for x in bnds])
            
            if use_errors: 
                out, cov, *_ = spo.curve_fit(limb_dark_cost, angles, intens, check_finite=True, p0=pguess, sigma=sigma_intens, absolute_sigma=True, bounds=bnds, max_nfev=1e6, x_scale='jac')
            else: 
                out, cov, *_ = spo.curve_fit(limb_dark_cost, angles, intens, check_finite=True, p0=pguess, bounds=bnds, max_nfev=1e6, x_scale='jac')
            nadir_list.append(float(limb_dark_cost(0., *out)))
            jac = spo.approx_fprime(out, ldwrap, np.float64(1e-4), 0.)
            sig = np.sqrt(jac.T @ (cov @ jac))
            nadir_std_list.append(float(sig))
            coefs_list.append(out)
            covar_list.append(cov)
            if plots: 
                axs[0].plot(angles, limb_dark_cost(angles, *out), color=angcolors[i])
                axs[0].plot(angles, intens, color=ldcolors[i])
                axs[1].plot(angles, intens - limb_dark_cost(angles, *out), color=angcolors[i])
        else: 
            nadir_list.append(np.nan)
            nadir_std_list.append(np.nan)
            coefs_list.append(np.nan * np.zeros(1 + n_exps*2))
            covar_list.append(np.nan * np.diag(np.zeros(1 + n_exps*2)))

    lat_array = data_frame.index.values
    nadir_array = np.array(nadir_list)
    nadir_std_array = np.array(nadir_std_list)
    coefs_array = np.stack(coefs_list)
    covar_array = np.stack(covar_list)

    output_dict = {'latitude_bins': lat_array, 
                   'nadir_intensity': nadir_array,
                   'intensity_error': nadir_std_array, 
                   'limb_darkening_coeffs': coefs_array, 
                   'limb_darkening_error': covar_array}

    if plots: 
        axs[0].set_xlabel('Emission angle (degrees)')
        axs[0].set_ylabel('Intensity')
        axs[1].set_xlabel('Emission angle (degrees)')
        axs[1].set_ylabel('Difference from model')
        fig.tight_layout()

    return output_dict

def plot_planet(image, header, beam, ghe_dict=None, msp_dict=None):
    """ General plotter for planets
        Only plots a single image 

        Inputs: 
        image - FITS data, hdu.data
        header - FITS header, hdu.header
        beam - radio_beam Beam object
        ghe_dict - Optional, output dictionary returned by get_horizons_ephemeris
        msp_dict - Optional, output dictionary returned by map_sky_projection
    """ 
    
    image = np.squeeze(image)
    wcs = WCS(header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs.celestial), layout='constrained')
    im = ax.imshow(image, origin='lower')
    cbar = fig.colorbar(im)
    cbar.set_label('Intensity ({})'.format(header['BUNIT']))
    ax.set_xlabel('Right Ascension ({}, degrees)'.format(wcs.wcs.radesys))
    ax.set_ylabel('Declination ({}, degrees)'.format(wcs.wcs.radesys))

    # Beam 
    elps = Ellipse((0.1, 0.1), width=beam.major.value, height=beam.minor.value, angle=beam.pa.value, color='white', fill=True, lw=0.1, transform=ax.transAxes)
    ax.add_patch(elps)

    # Draw outer circle
    if ghe_dict is not None: 
        ang_diam = ghe_dict['angular_diameter']
        b_eff = ghe_dict['apparent_equator_radius'].value / ghe_dict['apparent_pole_radius'].value
        np_angle = ghe_dict['north_pole_angle']
        center = (header['CRVAL1'] * u.deg, header['CRVAL2'] * u.deg)
        circle = SphericalCircle((center[0], center[1]), ang_diam/2, color='white', fill=False, lw=1, transform=ax.get_transform('world'))
        ax.add_patch(circle)

        # Set limits 
        world_xmin = SkyCoord(center[0]+ang_diam, center[1]-ang_diam, frame=wcs.wcs.radesys.lower())
        world_xmax = SkyCoord(center[0]-ang_diam, center[1]+ang_diam, frame=wcs.wcs.radesys.lower())
        xmin_pixel, ymin_pixel = wcs.celestial.world_to_pixel(world_xmin)
        xmax_pixel, ymax_pixel = wcs.celestial.world_to_pixel(world_xmax)
        ax.set_xlim(xmin_pixel, xmax_pixel)
        ax.set_ylim(ymin_pixel, ymax_pixel)
            
    # Draw lat/lon grid 
    if msp_dict is not None: 
        grid_lons = msp_dict['planetocentric_longitude'].value
        grid_lats = msp_dict['planetodetic_latitude'].value
        CS = ax.contour(np.round(grid_lats, 1), levels=[0, 30, 45, 60, 75], colors='w', linewidths=1.)
        CL = plt.clabel(CS, manual=False, fontsize=12, rightside_up=True, use_clabeltext=False)
        CS = ax.contour(np.round(grid_lons, 1), levels=[-180, -90, 0, 90, 180], colors='w', linewidths=1.)

    return fig, ax 


def resample_image(image, wcs, sample=1.):
    """ Image resampling
    
        Inputs: 
        image: Sky image array
        wcs: Image WCS 
        sample: > 1 is higher resolution, fractional is decimating

        Returns: 
        Resampled image 
        Dictionary containing resampled intermediate projection tuple, 
            resampled SkyCoord 
        """
    
    image = np.squeeze(image) 

    # Infer boundary behavior 
    if np.isnan(image).any(): 
        f_v = np.nan
    else: 
        f_v = 0.

    ygrid, xgrid = np.mgrid[:image.shape[0], :image.shape[1]]
    y2grid, x2grid = np.mgrid[:int(image.shape[0] * sample), :int(image.shape[1] * sample)]
    y3grid = y2grid / sample
    x3grid = x2grid / sample

    resample_image = spi.interpn((xgrid[0], ygrid[:, 0]), image, np.vstack([x3grid.ravel(), y3grid.ravel()]).T, 
                                 bounds_error=False, fill_value=f_v)
    resample_image = resample_image.reshape(x3grid.shape).T 

    # Resample the coordinates 
    # Projected coordinates 
    l_grid = x3grid - wcs.wcs.crpix[0]
    m_grid = y3grid - wcs.wcs.crpix[1]
    l_grid = l_grid * abs(wcs.wcs.cdelt[0])
    m_grid = m_grid * abs(wcs.wcs.cdelt[1])
    projcoord = (l_grid * wcs.wcs.cunit[0], m_grid * wcs.wcs.cunit[1])
    skycoord = wcs.celestial.array_index_to_world(x3grid, y3grid).icrs

    output_dict = {'projection_coordinates': projcoord, 
                   'sky_coordinates': skycoord}

    return resample_image, output_dict

def bindown_image(image, wcs, sample=1.):
    """ Image downsampling via histogram binning
    
        Inputs: 
        image: Sky image array
        wcs: Image WCS 
        sample: Must be less than 1 

        Returns: 
        Resampled image 
        Dictionary containing resampled intermediate projection tuple, 
            resampled SkyCoord 
        """
    if sample > 1: 
        raise ValueError('This is for downsampling only, sample must be < 1') 

    image = np.squeeze(image) 

    # Infer boundary behavior 
    if np.isnan(image).any(): 
        f_v = np.nan
    else: 
        f_v = 0.

    ygrid, xgrid = np.mgrid[:image.shape[0], :image.shape[1]]
    y2grid, x2grid = np.mgrid[:int(image.shape[0] * sample), :int(image.shape[1] * sample)]
    y3grid = y2grid / sample
    x3grid = x2grid / sample

    # Make bin edges
    # Using array indices so these will be gridded and consistent
    xlanes = x3grid[0] 
    xde = xlanes[1] - xlanes[0]
    ylanes = y3grid[:, 0] 
    yde = ylanes[1] - ylanes[0]
    xedges = xlanes - xde/2 
    xedges = np.append(xedges, xedges[-1] + xde)
    yedges = ylanes - yde/2 
    yedges = np.append(yedges, yedges[-1] + yde)

    counts, *_ = np.histogram2d(xgrid.ravel(), ygrid.ravel(), bins=(xedges, yedges))
    weights, *_ = np.histogram2d(xgrid.ravel(), ygrid.ravel(), bins=(xedges, yedges), weights=image.ravel())
    resample_image = (weights / counts).T 
    
    # Resample the coordinates 
    # Projected coordinates 
    l_grid = x3grid - wcs.wcs.crpix[0]
    m_grid = y3grid - wcs.wcs.crpix[1]
    l_grid = l_grid * abs(wcs.wcs.cdelt[0])
    m_grid = m_grid * abs(wcs.wcs.cdelt[1])
    projcoord = (l_grid * wcs.wcs.cunit[0], m_grid * wcs.wcs.cunit[1])
    skycoord = wcs.celestial.array_index_to_world(x3grid, y3grid).icrs

    output_dict = {'projection_coordinates': projcoord, 
                   'sky_coordinates': skycoord}

    return resample_image, output_dict


def make_limbdarkened_disk_model(disk_avg, ldp, emang_map): 
    """ Makes a limb-darkened disk from a map of emission angle in the sky plane

        Inputs: 
        disk_avg: Disk averaged intensity in arbitrary units 
        ldp: Limb-darkening parameter (0 is a uniform disk)
        emang_map: Map of incidence angle in the sky plane (degrees)
    """

    lddisk = disk_avg * np.cos(np.radians(emang_map))**ldp
    lddisk[np.isnan(emang_map)] = np.nan 
    # Ensure disk average quantity is consistent  
    lddisk = lddisk * disk_avg / np.nanmean(lddisk)
    lddisk[np.isnan(lddisk)] = 0.   
    
    return lddisk

def make_zonal_model(lat_bins, disk_intens, ldp, lat_map, emang_map): 
    """ Makes a zonally resolved limb-darkened disk model

        Inputs: 
        lat_bins: Latitude bins for disk_intens and ldp
        disk_intens: Disk intensity 
        ldp: Limb-darkening parameter (0 is a uniform disk)
        lat_map: Map of latitude in the sky plane (degrees)        
        emang_map: Map of incidence angle in the sky plane (degrees)
    """

    nm = ~np.isnan(disk_intens)
    iot = spi.interp1d(lat_bins[nm], disk_intens[nm], bounds_error=False, fill_value=(disk_intens[nm][0], disk_intens[nm][-1]))
    pt = spi.interp1d(lat_bins[nm], ldp[nm], bounds_error=False, fill_value=(ldp[nm][0], ldp[nm][-1]))
    zonedisk = iot(lat_map.ravel()) * np.cos(np.radians(emang_map.ravel()))**pt(lat_map.ravel())
    zonedisk = zonedisk.reshape(emang_map.shape)
    zonedisk[np.isnan(zonedisk)] = 0.

    return zonedisk 

def make_zonal_model_exps(lat_bins, coeffs, lat_map, emang_map): 
    """ Makes a zonally resolved limb-darkened disk model

        Inputs: 
        lat_bins: Latitude bins 
        lat_map: Map of latitude in the sky plane (degrees)        
        emang_map: Map of incidence angle in the sky plane (degrees)
    """

    # Compute brightness curves, the extrapolate in brightness 

    # angles = np.arange(0, 90, 0.5)     
    angles = np.arange(0, np.nanmax(emang_map), 0.5)     
    nm = ~np.isnan(coeffs[:, 0])
    use_bins = lat_bins[nm]
    use_coeffs = coeffs[nm, :]  
    brights = [] 

    for j, uc in enumerate(use_coeffs): 
        out = 0.
        for i in range(1, coeffs.shape[-1], 2): 
            out = out +  uc[0] * np.exp( uc[i] / np.cos(np.radians(angles))** uc[i+1])
        brights.append(out)
    brights = np.stack(brights) 

    rgi = spi.RegularGridInterpolator((use_bins, angles), brights, method='linear', bounds_error=False, fill_value=None)
    expdisk = rgi((lat_map, emang_map))
    expdisk[np.isnan(expdisk)] = 0.
    expdisk[expdisk < 0] = 0.

    # nm = ~np.isnan(coeffs[:, 0])
    # pt = spi.interp1d(lat_bins[nm], coeffs[nm, :], axis=0, bounds_error=False, fill_value=(coeffs[nm][0, :], coeffs[nm][-1, :]), kind='nearest')
    # map_p = pt(lat_map.ravel())
    # out = 0. 
    # for i in range(1, coeffs.shape[-1], 2): 
    #     out = out + map_p[:, 0] * np.exp(map_p[:, i] / np.cos(np.radians(emang_map.ravel()))**map_p[:, i+1])
    # expdisk = out.reshape(emang_map.shape)
    # expdisk[np.isnan(expdisk)] = 0.
    # expdisk[expdisk < 0] = 0.

    return expdisk 

def return_relative_coords(skycoord): 
    """ Simple routine to compute relative coordinates 
    """
    # Convert to Cartesian representation (unit vector on sphere)
    cartesian = skycoord.cartesian
    mean_x = np.mean(cartesian.x)
    mean_y = np.mean(cartesian.y)
    mean_z = np.mean(cartesian.z)
    mean_cart = CartesianRepresentation(mean_x, mean_y, mean_z)
    center = SkyCoord(mean_cart, frame=skycoord.frame)
    relative_ra = skycoord.ra - center.ra
    relative_dec = skycoord.dec - center.dec
    return center, relative_ra, relative_dec

def convolve_with_beam(image, synthesized_beam, res): 
    """ Convolves image with a Gaussian beam using astropy 
    """

    beam_kernel = synthesized_beam.as_kernel(res)
    conv_image = convolve_fft(image, beam_kernel)
    return conv_image

def planck_function(freq, intens): 
    """ Frequency in Hz units 
        Intensity in Jy
    """
    Bv = intens * 1e-26 
    log_arg = 2.0 * spc.h * freq**3. / (spc.c**2. * Bv) + 1.0 
    T = (spc.h * freq / spc.k) * 1.0 / (np.log(log_arg))
    return T 

def rj_function(freq, intens): 
    """ Frequency in Hz units 
        Intensity in Jy
    """
    wave = spc.c / freq
    Bv = intens * 1e-26
    T = Bv / 2 / spc.k / wave**2 
    return T 

def inv_planck_function(freq, T): 
    """ Frequency in Hz units 
        T in Kelvins
    """
    Bv = 2 * spc.h * freq**3 / spc.c**2 / (np.exp(spc.h * freq / spc.k / T) - 1)
    return Bv * 1e26 

def inv_rj_function(freq, T): 
    """ Frequency in Hz units 
        T in Kelvins
    """
    wave = spc.c / freq
    Bv = T * 2 * spc.k * wave**2 
    return Bv * 1e26

