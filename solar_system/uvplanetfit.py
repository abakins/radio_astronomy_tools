
import numpy as np 
import scipy.optimize as spo 
import scipy.integrate as sint 
from scipy.special import gamma, jv
from scipy.stats import binned_statistic
import scipy.ndimage as sni 

from astropy.io import fits 
from astropy.wcs import WCS 
from astropy import units 
import matplotlib.pyplot as plt 

def limb_disk(Io, radius, u, v, phi, b, p,):
    """ Generates a limb darkened disk model for co-polarized visibilities

        :param Io: Disk intensity (Jy * Distance^2 / Area)
        :param radius: Disk radius in radians
        :param u: U baseline coordinates in wavelengths
        :param v: V baseline coordinates in wavelengths
        :param phi: counterclockwise rotation (in degrees) of the long axis 
                    relative to the image "l" coordinate (generally the horizontal axis)
        :param b: ratio of long to short axis of disk
        :param p: limb darkening parameter

        Defined by Butler and Bastian's chapter in SIRA II 
    """

    phi = np.radians(phi)
    u_p = u * np.cos(phi) - v * np.sin(phi)
    v_p = b * (u * np.sin(phi) + v * np.cos(phi))
    beta = radius * np.sqrt(u_p**2 + v_p**2)
    arg = 2 * np.pi * beta
    q = 1 + p / 2
    lambdafunc = gamma(q + 1) * (0.5 * arg)**(-q) * jv(q, arg) 
    vis = Io * np.pi * radius**2 * lambdafunc

    return vis

def disk_crosspol(radius, u, v, phi, b, er, ei): 
    """ Generates a normalized disk model for cross-polarized visibilities

    :param radius: Disk radius in radians
    :param u: U baseline coordinates in wavelengths
    :param v: V baseline coordinates in wavelengths
    :param phi: counterclockwise rotation (in degrees) of the long axis 
                relative to the image "l" coordinate (generally the horizontal axis)
    :param b: ratio of long to short axis of disk
    :param er: Dielectric constant, real part 
    :param ei: Dielectric constant, imag part

    Defined by Butler and Bastian's chapter in SIRA II 
    """

    phi = np.radians(phi)
    u_p = u * np.cos(phi) - v * np.sin(phi)
    v_p = b * (u * np.sin(phi) + v * np.cos(phi))
    beta = radius * np.sqrt(u_p**2 + v_p**2)
    eps = er + 1j * ei 

    def integrand(p): 
        theta = np.arcsin(p)
        Rpl = (eps * np.cos(theta) - np.sqrt(eps - np.sin(theta)**2)) / \
              (eps * np.cos(theta) + np.sqrt(eps - np.sin(theta)**2))
        Rpr = (np.cos(theta) - np.sqrt(eps - np.sin(theta)**2)) / \
              (np.cos(theta) + np.sqrt(eps - np.sin(theta)**2))
        Rpl = abs(Rpl**2)
        Rpr = abs(Rpr**2)
        return (Rpl-Rpr) * jv(2, 2 * np.pi * p * beta[:, np.newaxis]) * p 

    res = sint.fixed_quad(integrand, 0, 1, n=100)  # I think n=100 gives sufficient accuracy 
    return res[0] 

def get_pol_vis(cvis, u, v, rot_angle, pol_type='linear'): 
    """ Computes polarized visibility quantity from polarimetric visibilities, used in uvplanetfit_polarized
        Definitions are from a memo shared by B. Butler. 
        
        Inputs: 
        :param cvis: Array of complex visibility samples (real + 1j imag),
                    should be 4xN with polarizations in alphabetical order
        :param u: U baseline coordinates in wavelengths
        :param v: V baseline coordinates in wavelengths
        :param rot_angle: Polarization rotation offset angle in degrees 
        :param pol_type: Either 'linear' or 'circular'
    """

    phi = np.arctan(u / v) 
    psi = np.radians(rot_angle) 
    cosf = np.cos(2 * phi - 2 * psi)
    sinf = np.sin(2 * phi - 2 * psi) 

    if pol_type == 'linear': 
        Vp = (cvis[3] - cvis[0]) * cosf - (cvis[1] - cvis[2]) * sinf
    elif pol_type == 'circular': 
        Vp = (cvis[2] - cvis[1]) * cosf - 1j * (cvis[2] - cvis[1]) * sinf
    else: 
        raise ValueError('Invalid poltype') 
    
    return Vp 
            
def uvplanetfit(cvis, uwave, vwave, fit_params={'Io': 1., 'r': 1.}, static_params={}, fit_reals=False, 
                solmode='L2', solver_kwargs={}, compute_sigma=False, sigma_bin=101, 
                make_plots=False, savefig='fit_result.png'): 
    """ uvplanetfit is a replacement for CASA's uvmodelfit task. 
        In addition to generating a uniform disk model, it can also accomodate limb-darkening 
        and ellipsoid shapes. 
        This function can only be used for co-polarized visibilities, see uvplanetfit_polarized for cross-polarized data

        Parameters which can be fit (added via fit_params as a dictionary including starting guesses)
        Other values are either assumed by default or can be added via static_params
        Valid fit_params include: 
        'Io': Disk intensity (Jy * Distance^2 / Area)
        'r': Disk radius in radians 
        'p': Limb darkening parameter 
        'b': Ratio of long to short axis of disk
        'phi': Counterclockwise rotation (in degrees) of the long axis relative to the image "l" coordinate (generally the horizontal axis)

        The user must extract input quantites from their CASA ms using ms.getdata 
        and associated routines. Make sure also to apply any flags 
        
        Inputs: 
        :param cvis: Array of co-polarized complex visibility samples (real + 1j imag)
        :param uwave: Array of spatial frequencies along the U axis, in units of wavelengths 
        :param vwave: Array of spatial frequencies along the V axis, in units of wavelengths
        :param fit_params: See above 
        :param static_params: See above 
        :param fit_reals: If True, fit to the real part of the visibility measurement 
                            if False (default), fit to the visibility amplitudes 
        :param solmode: Cost function norm, either 'L2' (default) or 'L1'
        :param solver_kwargs: Dictionary of keyword arguments to pass to solver 
                              see scipy.optimize.minimize
        :param compute_sigma: If True (default is False), bin visibilities and compute standard deviation to weight fitting 
        :param sigma_bin: Bin for visibility standard deviation calculations 
        :param make_plots: If True (default False), make plots to illustrate the fit 
        :param savefig: If None, don't save. Otherwise, save to a filename which is specified in this argument

    This routine will eventually be updated to provide uncertainties for the fit parameters 

    """

    static_params.setdefault('Io', 1.)
    static_params.setdefault('r', 1.)
    static_params.setdefault('p', 0.)
    static_params.setdefault('b', 1.)
    static_params.setdefault('phi', 0.)

    solver_kwargs.setdefault('method', 'Nelder-Mead')
    solver_kwargs.setdefault('options', {'disp': True, 'maxiter': 1e4, 'xatol': 1e-3})

    if fit_reals: 
        visdata = np.real(cvis)
    else: 
        visdata = np.abs(cvis) 

    uvwave = np.sqrt(uwave**2 + vwave**2)
    if compute_sigma:         
        use_bins = np.linspace(np.min(uvwave), np.max(uvwave), sigma_bin)
        statistic, bin_edges, binnumber = binned_statistic(uvwave, visdata, statistic='std', bins=use_bins)
        statistic[statistic == 0] = np.min(statistic[statistic > 0])
        sigma = statistic[binnumber - 1]
    else: 
        sigma = None 

    fit_keys = list(fit_params.keys())
    args = (visdata, uwave, vwave, fit_keys, static_params, fit_reals, solmode, sigma)
    result = spo.minimize(fun=cost_function, x0=np.array(list(fit_params.values())), args=args, **solver_kwargs)

    all_params = static_params
    for i, k in enumerate(fit_keys): 
        all_params[k] = result.x[i]

    if make_plots: 
        uvmin = np.min(uvwave)
        uvmax = np.max(uvwave)
        uv_input = np.linspace(uvmin, uvmax, 10000)
        u_in = np.sqrt((uv_input**2) / 2)
        v_in = u_in

        model_disk = limb_disk(all_params['Io'], all_params['r'], u_in, v_in, all_params['phi'], all_params['b'], all_params['p'])
        sub_disk = limb_disk(all_params['Io'], all_params['r'], uwave, vwave, all_params['phi'], all_params['b'], all_params['p'])

        if not fit_reals: 
            model_disk = abs(model_disk)
            sub_disk = abs(sub_disk) 
            ylabel = 'Abs(V) (Jy)'
        else: 
            ylabel='Re(V) (Jy)'


        diff = visdata - sub_disk

        fig = plt.figure() 
        ax = fig.add_subplot(111) 
        ax.plot(uvwave / 1e3, visdata, linestyle='none', marker = '.', markersize=1, color='darkcyan', label='Calibrated Visibilities')   
        ax.plot(uv_input / 1e3, model_disk, color='black', linestyle='--', linewidth=0.8, label='Fit Model')
        
        ax.set_xlabel(r'UV Distance (k$\lambda$)')
        ax.set_ylabel(ylabel)
        ins = ax.inset_axes([0.45, 0.45, 0.5, 0.5])
        ins.plot(uvwave / 1e3, diff, linestyle='none', marker='.', markersize=1, color='darkcyan', label='Calibrated Visibilities')
        ins.set_xlabel(r'UV Distance (k$\lambda$)')
        ins.set_ylabel(ylabel)

        if compute_sigma:
            ubin = np.sqrt((use_bins**2)/2) 
            vbin = ubin 
            error_disk = limb_disk(all_params['Io'], all_params['r'], ubin, vbin, all_params['phi'], all_params['b'], all_params['p'])
            if not fit_reals: 
                error_disk = abs(error_disk) 
            ax.fill_between(use_bins[:-1] / 1e3, error_disk[:-1] + statistic, error_disk[:-1] - statistic, alpha=0.25, color='darkcyan')
            ins.fill_between(use_bins[:-1] / 1e3, statistic, -statistic, alpha=0.25, color='darkcyan')
        
        plt.tight_layout()
        if savefig: 
            plt.savefig(savefig, dpi=300)

    print('Final fit parameters')
    for k, v in all_params.items(): 
        print('{}: {}'.format(k, v))


    return all_params

def uvplanetfit_polarized(cvis, uwave, vwave, fit_params={'eps': 1., 'r': 1.}, static_params={}, fit_reals=False,
                          solmode='L2', solver_kwargs={}, compute_sigma=False, sigma_bin=101, 
                          make_plots=False, savefig='fit_result_xpol.png', feed_type=['LL', 'LR', 'RL', 'RR']): 
    """ uvplanetfit_polarized is the equivalent of uvplanetfit for cross-polarized measurements 

        Parameters which can be fit (added via fit_params as a dictionary including starting guesses)
        Other values are either assumed by default or can be added via static_params
        Valid fit_params include: 
        'eps': Real part of the dielectric constant (if static, can be specified as complex)
        'r': Disk radius in radians 
        'b': Ratio of long to short axis of disk
        'phi': Counterclockwise rotation (in degrees) of the long axis relative to the image "l" coordinate (generally the horizontal axis)
        'rot_angle': Polarization angle offset (in degrees), usually a systematic term which must be corrected for. 

        Inputs: 
        :param cvis: Array of complex visibility samples (real + 1j imag),
                    should be 4xN with first index being feed polarization (sequence in feed_type)
        :param uwave: Array of spatial frequencies along the U axis, in units of wavelengths 
        :param vwave: Array of spatial frequencies along the V axis, in units of wavelengths 
        :param fit_params: See above 
        :param static_params: See above 
        :param fit_reals: If True, fit to the real part of the visibility measurement 
                            if False (default), fit to the visibility amplitudes 
        :param solmode: Cost function norm, either 'L2' (default) or 'L1'
        :param solver_kwargs: Dictionary of keyword arguments to pass to solver 
                              see scipy.optimize.minimize
        :param compute_sigma: If True (default is False), bin visibilities and compute standard deviation to weight fitting 
        :param sigma_bin: Bin for visibility standard deviation calculations 
        :param make_plots: If True (default False), make plots to illustrate the fit 
        :param savefig: If None, don't save. Otherwise, save to a filename which is specified in this argument
        :param feed_type: Length four list of feed polarization strings corresponding to the rows of cvis (i.e. ('LL', 'LR', 'RL', 'RR'))

    """

    static_params.setdefault('eps', 1.)
    static_params.setdefault('r', 1.)
    static_params.setdefault('b', 1.)
    static_params.setdefault('phi', 0.)
    static_params.setdefault('rot_angle', 0.)

    solver_kwargs.setdefault('method', 'Nelder-Mead')
    solver_kwargs.setdefault('options', {'disp': True, 'maxiter': 1e4, 'xatol': 1e-3})

    # Sort out polarization
    mask = np.argsort(feed_type)
    feed_type = list(np.array(feed_type)[mask])
    if feed_type == ['LL', 'LR', 'RL', 'RR']: 
        pol_type = 'circular' 
    elif feed_type == ['XX', 'XY', 'YX', 'YY']: 
        pol_type = 'linear'
    else: 
        raise ValueError('Specify appropriate linear or circular polarization feeds')
    cvis = cvis[mask]  # Sort visibilities by polarization 
    
    uvwave = np.sqrt(uwave**2 + vwave**2)

    if compute_sigma:         
        # Get firstpass polarized visibilities 
        all_params = static_params
        for k, v in fit_params.items(): 
            all_params[k] = v
        visdata = abs(get_pol_vis(cvis, uwave, vwave, all_params['rot_angle'], pol_type=pol_type))
        use_bins = np.linspace(np.min(uvwave), np.max(uvwave), sigma_bin)
        statistic, bin_edges, binnumber = binned_statistic(uvwave, visdata, statistic='std', bins=use_bins)
        statistic[statistic == 0] = np.min(statistic[statistic > 0])
        sigma = statistic[binnumber - 1]
    else: 
        sigma = None 

    fit_keys = list(fit_params.keys())
    args = (cvis, uwave, vwave, fit_keys, static_params, fit_reals, solmode, sigma, pol_type)
    result = spo.minimize(fun=cost_function_polarized, x0=np.array(list(fit_params.values())), args=args, **solver_kwargs)

    all_params = static_params
    for i, k in enumerate(fit_keys): 
        all_params[k] = result.x[i]

    if make_plots: 
        uvmin = np.min(uvwave)
        uvmax = np.max(uvwave)
        uv_input = np.linspace(uvmin, uvmax, 10000)
        u_in = np.sqrt((uv_input**2) / 2)
        v_in = u_in

        visdata = get_pol_vis(cvis, uwave, vwave, all_params['rot_angle'], pol_type=pol_type)
        model_disk = disk_crosspol(all_params['r'], u_in, v_in, all_params['phi'], all_params['b'], np.real(all_params['eps']), np.imag(all_params['eps']))
        sub_disk = disk_crosspol(all_params['r'], uwave, vwave, all_params['phi'], all_params['b'], np.real(all_params['eps']), np.imag(all_params['eps']))
        
        if not fit_reals: 
            visdata = abs(visdata)
            model_disk = abs(model_disk)
            sub_disk = abs(sub_disk) 
            ylabel = 'Abs(V) (Jy)'
        else: 
            ylabel='Re(V) (Jy)'

        diff = visdata - sub_disk

        fig = plt.figure() 
        ax = fig.add_subplot(111) 
        ax.plot(uvwave / 1e3, visdata, linestyle='none', marker = '.', markersize=1, color='darkcyan', label='Calibrated Visibilities')   
        ax.plot(uv_input / 1e3, model_disk, color='black', linestyle='--', linewidth=0.8, label='Fit Model')
        
        ax.set_xlabel(r'UV Distance (k$\lambda$)')
        ax.set_ylabel(ylabel)
        ins = ax.inset_axes([0.45, 0.45, 0.5, 0.5])
        ins.plot(uvwave / 1e3, diff, linestyle='none', marker='.', markersize=1, color='darkcyan', label='Calibrated Visibilities')
        ins.set_xlabel(r'UV Distance (k$\lambda$)')
        ins.set_ylabel(ylabel)

        if compute_sigma:
            ubin = np.sqrt((use_bins**2)/2) 
            vbin = ubin 
            error_disk = disk_crosspol(all_params['r'], ubin, vbin, all_params['phi'], all_params['b'], np.real(all_params['eps']), np.imag(all_params['eps']))
            ax.fill_between(use_bins[:-1] / 1e3, error_disk[:-1] + statistic, error_disk[:-1] - statistic, alpha=0.25, color='darkcyan')
            ins.fill_between(use_bins[:-1] / 1e3, statistic, -statistic, alpha=0.25, color='darkcyan')
        
        plt.tight_layout()
        if savefig: 
            plt.savefig(savefig, dpi=300)

    print('Final fit parameters')
    for k, v in all_params.items(): 
        print('{}: {}'.format(k, v))

    return all_params


def cost_function(params, visdata, uwave, vwave, fit_keys, static_params, 
                  fit_reals, solmode, sigma, disp_iter=True): 
    """ Cost function to minimize for fitting disk to data

        :param params: Current iteration of best fit parameters
        :param visdata: Visibility data, either amplitude or reals
        :param uwave: U baselines
        :param vwave: V baselines
        :param fit_keys: Dictionary keys for fit_params, since minimize guesses aren't dictionaries 
        :param static_params: Dictionary of static parameters 
        :param fit_reals: If True, visdata is reals, if False, visdata is amplitudes 
        :param solmode: Either 'L2' or 'L1'
        :param sigma: Standard deviation for the values of amp
        :param disp_iter: If True (default), display iteration cost function 

    """
    all_params = static_params
    for i, k in enumerate(fit_keys): 
        all_params[k] = params[i]
    
    model_disk = limb_disk(all_params['Io'], all_params['r'], 
                           uwave, vwave, all_params['phi'], all_params['b'], 
                           all_params['p'])
    if not fit_reals: 
        model_disk = abs(model_disk) 

    if sigma is None: sigma = np.ones(model_disk.shape)

    if solmode == 'L1': 
        out = np.sum(abs(visdata - model_disk) / sigma)
    elif solmode == 'L2': 
        out = np.sum((visdata - model_disk)**2 / sigma**2)
    else:
        raise ValueError('Invalid solmode argument') 
    
    if disp_iter: print(out)
    return out

def cost_function_polarized(params, visdata, uwave, vwave, fit_keys, static_params, fit_reals, 
                  solmode, sigma, pol_type, disp_iter=True): 
    """ Cost function to minimize for fitting disk to data, used in uvplanetfit_polarized

        :param params: Current iteration of best fit parameters
        :param visdata: Visibility data, either amplitude or reals
        :param uwave: U baselines
        :param vwave: V baselines
        :param fit_keys: Dictionary keys for fit_params, since minimize guesses aren't dictionaries 
        :param static_params: Dictionary of static parameters 
        :param fit_reals: If True, visdata is reals, if False, visdata is amplitudes 
        :param solmode: Either 'L2' or 'L1'
        :param sigma: Standard deviation for the values of amp
        :param pol_type: Either 'linear' or 'circular' for get_pol_vis
        :param disp_iter: If True (default), display iteration cost function 

    """
    all_params = static_params
    for i, k in enumerate(fit_keys): 
        all_params[k] = params[i]

    Vp = get_pol_vis(visdata, uwave, vwave, all_params['rot_angle'], pol_type)
    
    model_disk = disk_crosspol(all_params['r'], uwave, vwave, all_params['phi'], all_params['b'], 
                                np.real(all_params['eps']), np.imag(all_params['eps']))
    
    if not fit_reals: 
        Vp = abs(Vp)
        model_disk = abs(model_disk) 
    
    if sigma is None: sigma = np.ones(model_disk.shape)

    if solmode == 'L1': 
        out = np.sum(abs(Vp - model_disk) / sigma)
    elif solmode == 'L2': 
        out = np.sum(abs((Vp - model_disk)**2) / sigma**2)
    else:
        raise ValueError('Invalid solmode argument') 
    
    if disp_iter: print(out)
    return out


def uvpf_to_sm(uvpf_params, template_fits, upsample=5, savefilename='startmodel.fits'): 
    
    """ Converts the results of uvplanetfit into a starting model for tclean 
        Requires a template fits image, and it is assumed that the spatial axes 
        of this image are both equal and have even length.

        Inputs: 
        :param uvpf_params: Outputs of the uvfitplanet routine, see documentation for that function 
        :param template_fits: Template fits image to which the model will be mapped 
                              Values will be assumed uniform along the spectral axis, and non-Stokes-I images
                              will be assumed to be zero 
        :param upsample: Factor by which to upsample the image (to avoid sharp corners)
    """


    hdulist = fits.open(template_fits)
    hdu = hdulist[0] 
    this_wcs = WCS(hdu.header)
    
    
    # This isn't needed here, but I'm keeping it since I will need it for other routines later... 
    # Get a grid of world coordinates for the image
    # arraysize = this_wcs.celestial.array_shape
    # l_index, m_index = np.mgrid[:arraysize[0], :arraysize[1]]
    # lgrid, mgrid = this_wcs.celestial.array_index_to_world_values(l_index, m_index)
    # center_l, center_m = this_wcs.celestial.array_index_to_world_values(hdu.header['CRPIX1']-1, hdu.header['CRPIX2']-1)
    # lgrid = lgrid - center_l
    # mgrid = mgrid - center_m

    # Noting that the constraints imposed here (square image, symmetric rotated ellipsoid) 
    # are allowing me to get away with some imprecision r.e. coordinate systems

    imagesize = this_wcs.celestial.pixel_shape[0]   
    pixel_grid = np.linspace(-imagesize / 2, imagesize / 2, imagesize * upsample)  # Assumes square image 
    lgrid, mgrid = np.meshgrid(pixel_grid, pixel_grid)    
    # Set up a grid defining distance from the center 
    squeeze_radgrid = np.sqrt((1 / uvpf_params['b'])**2 * lgrid**2 + mgrid**2)
    # This numerical rotation approach below is not ideal, but not high enough priority to be refined right now 
    if uvpf_params['phi'] != 0: 
        rot_radgrid = sni.rotate(squeeze_radgrid, uvpf_params['phi'], reshape=False, mode='nearest')
    else: 
        rot_radgrid = squeeze_radgrid 

    cellsize = this_wcs.celestial.proj_plane_pixel_scales()[0].to(units.rad).value  # In radians 
    cellradius = uvpf_params['r'] / cellsize
        
    quot = (1 - (rot_radgrid / cellradius)**2)
    model_image = np.zeros((imagesize * upsample, imagesize * upsample))
    model_image[rot_radgrid < cellradius] = uvpf_params['Io'] * cellsize**2
    image = model_image * quot.astype(complex)**(uvpf_params['p'] / 2) * (1 + uvpf_params['p'] / 2)
    image = abs(image)
    # Downsample 
    image_grid = np.linspace(-imagesize / 2, imagesize / 2, imagesize + 1)
    bincount, *_ = np.histogram2d(lgrid.ravel(), mgrid.ravel(), bins=image_grid)
    binsum, *_ = np.histogram2d(lgrid.ravel(), mgrid.ravel(), weights=image.ravel(), bins=image_grid) 
    image = binsum / bincount

    # Write the image, using dynamic slicing 
    write_image = np.zeros(this_wcs.pixel_shape)
    dyslice = [slice(None)] * this_wcs.pixel_n_dim
    stokes_axis = this_wcs.axis_type_names.index('STOKES')
    dyslice[stokes_axis] = slice(0, 1) 
    write_image[tuple(dyslice)] = image[..., np.newaxis, np.newaxis]

    # And write 
    hdu.data = write_image.T 
    hdulist[0] = hdu
    hdulist.writeto(savefilename, overwrite=True) 




def print_RJ_TB(all_params, wavelength): 
    """ Prints Rayleigh Jeans brightness temperature from uvplanetfit outputs """
    datb = np.mean(all_params['Io'] * 1e-26 * wavelength**2 / (2 * 1.380649e-23))
    ldp_correction = (1 + all_params['p'] / 2)
    print('Disk Center Rayleigh Jeans Brightness Temperature: %f' % (datb * ldp_correction))
    print('Disk Average Rayleigh Jeans Brightness Temperature: %f' % datb)















