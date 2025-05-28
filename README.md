# radio_astronomy_tools
A collection of scripts for reduction and imaging of radio astronomy data. These are designed for use in Python with the CASA packages https://casa.nrao.edu/.
Everything is provided as-is, please e-mail alexakins@gmail.com with any issues. 

## solar_system 
Utilities for solar system objects. 
- uvplanetfit: A routine for fitting limb-darkened disks to visibility samples
-uvplanetfit_polarized: A routine for fitting polarized disk models to cross-polarized visibility samples 
- uvpf_to_sm: Takes the results of uvplanetfit and generates an image which can be passed to tclean's "startmodel" (polarimetric mode in progress)
