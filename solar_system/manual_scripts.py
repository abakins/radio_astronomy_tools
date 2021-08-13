import os
import numpy as np 
from numpy import sin, cos, sqrt, pi
from scipy.special import gamma, jv
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import pyfits as fits
import pickle

# Manual Changes 

# This script includes a collection of tasks that will need to be performed following processing of all Uranus VLA data with the scripted VLA pipeline.
# To select which tasks to perform, include them in the tasks list 
tasks = ['ImageUranus']

# Include the names of the relevant measurement set name and the fields associated with each target 
msname = 'merged.ms'
bpcal_field = '0,2'
flux_field = '0,2'
gain_field = '4'
target_field = '0~61'

# This task can be used to manually set the flux density of the flux calibrator if the VLA did not observe a standard calibrator. 
# The fluxes used will need to be modified for different datasets
# Hopefully, this won't be used very often, as the VLA should observe well characterized calibrators 
if 'SetFluxModel' in tasks: 
	# These are in spw order 
	fluxes = [0.7280, 0.7257, 0.7235, 0.7213, 0.7191, 0.7169, 0.7148, 0.7126, 0.8560, 0.8515, 0.8485, 0.8451, 0.8420, 0.8389, 0.8358, 0.8327]
	for i in range(0, len(fluxes)): 
		setjy(vis=msname, field=flux_field, spw=str(i), standard='manual', fluxdensity=[fluxes[i], 0, 0, 0], usescratch=True)

# This task is used to make CLEAN images of the calibrators to check the calibration solutions. 
# The cellsize and imsize parameters will need to be modified for different datasets depending on the desired resolution for imaging
# The CLEAN mask will need to be applied manually
if 'ImageCalibrators' in tasks: 
	cellsize = '0.025arcsec'
	imsize = [300, 300]
	os.system('rm -rf bpcal_clean.*')
	tclean(vis=msname, field=bpcal_field, imagename='bpcal_clean', imsize=imsize, cell=cellsize, specmode='mfs', gridder='standard', 
			deconvolver='hogbom', pbcor=True, weighting='briggs', robust=0., niter=2000, threshold='3mJy', cycleniter=500, interactive=True)

	os.system('rm -rf flux_clean.*')
	tclean(vis=msname, field=flux_field, imagename='flux_clean', imsize=imsize, cell=cellsize, specmode='mfs', gridder='standard', 
			deconvolver='hogbom', pbcor=True, weighting='briggs', robust=0., niter=2000, threshold='3mJy', cycleniter=500, interactive=True)

	os.system('rm -rf gain_clean.*')
	tclean(vis=msname, field=gain_field, imagename='gain_clean', imsize=imsize, cell=cellsize, specmode='mfs', gridder='standard', 
			deconvolver='hogbom', pbcor=True, weighting='briggs', robust=0., niter=2000, threshold='3mJy', cycleniter=500, interactive=True)

# This task averages together (in frequency) the data from the sidebands and places the USB and LSB averages in a new ms
# Spws and channel bins will need to be changed for each dataset
if 'ChannelAverage' in tasks: 
	usb_spw = '0~7'
	lsb_spw = '8~15'
	chanbin = 512  # Channels per spw * number of spws per sideband

	# Average to multi-channel
	mstransform(vis=msname, outputvis='average_set.ms', field='3~33', datacolumn='corrected', chanaverage=True, chanbin=64)
	# Average to single channel 
	mstransform(vis=msname, outputvis='average_set_USB.ms', field=target_field, spw=usb_spw, combinespws=True, datacolumn='corrected', keepflags=False)
	mstransform(vis=msname, outputvis='average_set_LSB.ms', field=target_field, spw=lsb_spw, combinespws=True, datacolumn='corrected', keepflags=False)
	mstransform(vis='average_set_USB.ms', outputvis='average_set_USB_single.ms', chanaverage=True, chanbin=chanbin, datacolumn='data')
	mstransform(vis='average_set_LSB.ms', outputvis='average_set_LSB_single.ms', chanaverage=True, chanbin=chanbin, datacolumn='data')
	concat(vis=['average_set_LSB_single.ms', 'average_set_USB_single.ms'], concatvis='average_set.ms')

# This task performs a phase-only self-calibration with a variable solution interval
if 'SelfCalibrate' in tasks: 
	solint = '1min'
	refant = 'ea05,ea12,ea21,ea15'
	os.system('rm -rf selfcal.cal')
	gaincal(vis=msname, field=target_field, spw='0,1,5~15', caltable='selfcal.cal', solint=solint, calmode='p', refant=refant, minsnr=3, gaintype='T')
	applycal(vis=msname, field=target_field, gaintable='selfcal.cal')


if 'ImageUranus' in tasks: 
	# For every factor of 2 increase in resolution, the model brightness must be decreased by 4 (square law)
	os.system('rm -rf cleaned_image_mfs.*')
	cellsize='0.05arcsec'  
	imsize=[300,300]
	tclean(vis=msname, field=target_field, imagename='cleaned_image_mfs', imsize=imsize, cell=cellsize, specmode='mfs',
			gridder='standard', deconvolver='multiscale', scales=[0, 6, 12, 24], pbcor=True, weighting='briggs', robust=0., niter=10000, threshold='0.0mJy',
			cycleniter=5000, interactive=True, phasecenter='URANUS', spw='5~15', startmodel='StartModel.im', datacolumn='corrected', mask='circle[[139pix,150pix],45pix]')

# The following functions are necessary for the task of generating a limb-darkened disk image that is fit to the visibility data 


def limb_disk(Io, radius, u, v, phi, b, p):
	""" Generates a limb darkened disk in visibility-space
	:param Io: Disk intensity
	:param radius: disk radius in radians
	:param u: U baseline coordinates in wavelengths
	:param v: V baseline coordinates in wavelengths
	:param phi: counterclockwise rotation off axis
	:param b: ratio of long to short axis of disk
	:param p: limb darkening parameter
	"""

	phi = np.radians(phi)

	u_p = u * cos(phi) - v * sin(phi)
	v_p = b * (u * sin(phi) + v * cos(phi))

	beta = radius * sqrt(u_p**2 + v_p**2)

	arg = 2 * pi * beta
	q = 1 + p / 2
	lambdafunc = gamma(q + 1) * (0.5 * arg)**(-q) * jv(q, arg)  # From Solar System Objects paper in NRAO Synth Imaging
	vis = Io * pi * radius**2 * lambdafunc

	return vis


def make_disk(imagesize, cellsize, Io, radius, p, offset_tuple): 
	""" Creates a limb darkened disk brightness image 
		:param imagesize: number of pixels in the image 
		:param cellsize: length of cell in arcseconds
		:param Io: disk peak intensity 
		:param radius: Radius of the disk in radians 
		:param p: limb darkening parameter
	"""
	cellsize = cellsize * 4.8481e-6  # Conversion to radians
	delu = 1 / (imagesize * cellsize)
	uv_max = delu * imagesize / 2
	uv = np.linspace(-uv_max, uv_max, imagesize)
	umesh, vmesh = np.meshgrid(uv, uv)
	visibilities = limb_disk(Io, radius, umesh, vmesh, 0, 1, p)
	image = np.fft.fft2(visibilities) / imagesize**2
	image = np.fft.fftshift(image)
	imsave = abs(image)
	imsave = np.roll(imsave, offset_tuple[0], axis=0)
	imsave = np.roll(imsave, offset_tuple[1], axis=1)
	# Image storage
	hdu = fits.PrimaryHDU(imsave)
	hdul = fits.HDUList([hdu])
	os.system('rm -rf StartModel.fits')
	hdul.writeto('StartModel.fits')
	hdul.close()


def radius_function(params, u, v, real, mode): 
	""" Cost function to minimize for fitting disk to data
		See the function description for 'fit_disk'
		:param params: Current iteration of best fit parameters
		:param u: U baselines
		:param v: V baselines
		:param real: Real part of visibilities
	"""
	Io = params[0]
	p = params[1]
	radius = params[2]
	if mode == 'Uniform': 
		p = 0
	modelreal = limb_disk(Io, radius, u, v, 0, 1, p)
	out = np.sum((real - modelreal)**2)
	# No negative fluxes should result
	if Io < 0: 
		out += 1e20
	print(out)
	return out


def define_disk(imagesize, cellsize, Io, radius): 
	# June 2012: 3.448 arcsec or 1.671638e-5 radian diameter
	# b = 1.023 for Uranus 
	# Intens is 2.0735 mJy/beam, total flux dens is 5.6921e-1 
	# Io is 2487231593 (whatever units)

	image = np.zeros((imagesize, imagesize))
	pixrad = pi / (180 * 3600) * cellsize
	cellradius = radius / pixrad
	shape = np.shape(image)
	check = np.linspace(-shape[0] / 2, shape[0] / 2, shape[0])  # Assumes square image 
	c_lr, c_ud = np.meshgrid(check, check)
	rad = np.sqrt(c_lr**2 + c_ud**2)
	# Empirical shift
	rad = np.sqrt(c_lr**2 + (c_ud - 3)**2)
	image[rad < cellradius] = Io * (pi / (180 * 3600) * cellsize)**2
	plt.imshow(image)
	hdu = fits.PrimaryHDU(image)
	hdul = fits.HDUList([hdu])
	os.system('rm -rf StartModel.fits')
	hdul.writeto('StartModel.fits')
	hdul.close()


def fit_disk(filename, field, spw, imagesize, cellsize, guess, offset_tuple, mode='LimbDarken', plots=False, putuvmodel=False):
	""" Fits a uniform or limb-darkened disk to the visibilities
		and generates an image-space starting model in Jy/pixel units """ 

	fitfield = field[::3]

	ms.open(filename)
	ms.selectinit(datadescid=spw)  
	ms.select({'field_id': fitfield})
	ms.selectpolarization(['LL'])
	data = ms.getdata(['real', 'u', 'v', 'uvdist', 'flag', 'axis_info'])
	pickle.dump(data, open('raw_data.p', 'wb'))
	ms.close()
	wavelength = 3e8 / data['axis_info']['freq_axis']['chan_freq'][0]
	wavelength = np.mean(wavelength)  # Is this valid?
	real = data['real'][0][0]
	u = data['u'] / wavelength
	v = data['v'] / wavelength
	flag = data['flag'][0][0]
	new_u = u[flag == False]
	new_v = v[flag == False]
	new_real = real[flag == False]

	args = (new_u, new_v, new_real, mode)

	out = minimize(fun=radius_function, x0=guess, args=args, method='Nelder-Mead', options={'disp': True, 'maxiter': 1e4, 'fatol': 1e-6})
	if mode == 'Uniform': 
		out.x[1] = 0
	print("Io: %f" % out.x[0])
	print("p: %f" % out.x[1])
	print("radius: %f" % out.x[2])
	print('Disk Center Brightness Temperature: %f' % (out.x[0] * 1e-26 * wavelength**2 / (2 * 1.380649e-23) * (1 + out.x[1] / 2)))
	print('Disk Average Brightness Temperature: %f' % (out.x[0] * 1e-26 * wavelength**2 / (2 * 1.380649e-23)))
	Jypixel_center = out.x[0] * (1 + out.x[1] / 2) * (pi / (180 * 3600) * cellsize)**2
	Jypixel_average = out.x[0] * (pi / (180 * 3600) * cellsize)**2
	print('Disk Center Intensity per Pixel: %f Jy' % Jypixel_center)
	print('Disk Average Intensity per Pixel: %f Jy' % Jypixel_average)

	make_disk(imagesize, cellsize, out.x[0], out.x[2], out.x[1], offset_tuple)

	if plots: 
		uvdist = sqrt(new_u**2 + new_v**2)
		uvmax = max(uvdist)
		uvmin = min(uvdist)
		uvinput = np.linspace(uvmin, uvmax, 1000)
		u_in = sqrt((uvinput**2) / 2)
		v_in = u_in
		model_real = limb_disk(out.x[0], out.x[2], u_in, v_in, 0, 1, out.x[1])
		fig, ax = plt.subplots(1, 1)
		plt.plot(uvdist / 1e3, new_real, linestyle='none', marker = '.', markersize=1, color='darkcyan', label='Calibrated Visibilities')	
		plt.plot(uvinput / 1e3, model_real, color='black', linestyle='--', linewidth=0.8, label='Limb Darkened Disk Model')
		plt.xlabel(r'UV Distance (k$\lambda$)')
		plt.ylabel('Real(V) (Jy)')
		plt.grid(b=False)
		# sub_real = limb_disk(out.x[0], out.x[2], new_u, new_v, 0, 1, out.x[1])
		# diff = new_real - sub_real
		# ins = ax.inset_axes([0.45, 0.45, 0.5, 0.5])
		# ins.plot(uvdist / 1e3, diff, linestyle='none', marker='.', markersize=0.3, color='darkcyan', label='Calibrated Visibilities')
		# ins.set_xlabel(r'UV Distance (k$\lambda$)')
		# ins.set_ylabel('Real(V) (Jy)')
		plt.tight_layout()
		plt.savefig('VisibilityFit.pdf', format='pdf', dpi=600, transparent=True)
		plt.close()
	if putuvmodel: 
		ms.open(filename, nomodify=False)
		ms.selectinit(datadescid=spw)  
		ms.select({'field_id': field})
		ms.selectpolarization(['LL', 'RR'])
		data = ms.getdata(['model_data', 'u', 'v', 'flag', 'axis_info'])
		u = data['u'] / wavelength
		v = data['v'] / wavelength
		flag = data['flag'][0][0]
		new_u = u[flag == False]
		new_v = v[flag == False]
		fitted_real = limb_disk(out.x[0], out.x[2], new_u, new_v, 0, 1, out.x[1])
		data['model_data'][:, :, flag == False] = fitted_real.astype(np.complex)
		ms.putdata(data)
		ms.close()


if 'FitDisk' in tasks: 
	imagesize = 300
	cellsize = 0.05
	field = np.arange(0, 40, 1)
	spw = 0
	guess = np.array([1e12, 0, pi / (180 * 3600) * 1.5])  # Zero spacing visibility, limb-darkening, radius in radians 
	offset_tuple = (0, -11)  # Vertical, horizontal in pixels
	fit_disk(msname, field, spw, imagesize, cellsize, guess, offset_tuple, mode='LimbDarken', putuvmodel=False)


# Function for inputting correct ephemeris into a starting model
def load_model(filename, template, imagesize, cellsize): 

	imagefile = filename[:-5] + '.im'
	os.system('rm -rf %s' % imagefile)
	apply_params = ['cdelt1', 'cdelt2', 'crpix1', 'crpix2' ]
	copy_params = ['bunit', 'cunit1', 'cunit2', 'cunit4', 'cdelt4', 'crpix4', 'crval1', 'crval2', 'crval4', 'ctype1', 'ctype2', 'ctype4', 'date-obs', 'equinox', 'imtype', 'object', 'projection', 'reffreqtype', 'telescope']

	# Get the base image from which parameters are copied
	model_base = imhead(imagename=template, mode='list')
	importfits(fitsimage=filename, imagename=imagefile, defaultaxes=True, defaultaxesvalues=[model_base['crval1'], model_base['crval2'], model_base['crval4'], 'I'],
				beam=[str(model_base['beammajor']['value']) + model_base['beammajor']['unit'], str(model_base['beamminor']['value']) + model_base['beamminor']['unit'], str(model_base['beampa']['value']) + model_base['beampa']['unit']])

	# Change axes names
	imhead(imagename=imagefile, mode='put', hdkey='ctype1', hdvalue='Right Ascension')
	imhead(imagename=imagefile ,mode='put', hdkey='ctype2', hdvalue='Declination')
	# Delete degenerate axes
	imsubimage(imagename=imagefile, outfile=imagefile + '.sub', dropdeg=True, keepaxes=[4, 5])
	os.system('rm -rf %s' % (imagefile))
	# Reorder the axes for consistency
	imtrans(imagename=imagefile + '.sub', outfile=imagefile, order='0132')
	os.system('rm -rf %s.sub' % (imagefile))

	# Digging in with the CASA toolkit
	# The default axes for the model image are Linear, where CASA needs them to be Direction axes. 
	ia.open(template)
	image_cs = ia.coordsys()
	ia.close()
	ia.open(imagefile)
	model_cs = ia.coordsys()
	# Replace the coordinate axes
	model_cs.replace(image_cs.torecord(), 0, 0)
	ia.setcoordsys(model_cs.torecord())
	# Done
	ia.close()

	# Iterate through and copy the relevant parameters from the initially calibrated image into the model image 
	for index2 in copy_params: 
		imhead(imagename=imagefile, mode='put', hdkey=index2, hdvalue=str(model_base[index2]))
	# Changing some of the default parameters based on the actual image dimensions
	imhead(imagename=imagefile, mode='put', hdkey='cdelt1', hdvalue=str(-cellsize * np.pi / (180 * 3600)))
	imhead(imagename=imagefile, mode='put', hdkey='cdelt2', hdvalue=str(cellsize * np.pi / (180 * 3600)))
	imhead(imagename=imagefile, mode='put', hdkey='crpix1', hdvalue=str(imagesize / 2))
	imhead(imagename=imagefile, mode='put', hdkey='crpix2', hdvalue=str(imagesize / 2))


if 'LoadModel' in tasks: 
	imagesize = 300
	cellsize = 0.05 
	template = 'cleaned_image.image'
	model = 'StartModel.fits'
	load_model(model, template, imagesize, cellsize)


