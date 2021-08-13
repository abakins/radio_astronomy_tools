# radio_astronomy_tools
A collection of scripts for reduction and imaging of radio astronomy data. These are designed for use in Python with the CASA packages https://casa.nrao.edu/.
Everything is provided as-is, please e-mail alexander.akins@jpl.nasa.gov with any issues. 

## solar_system 

Utilities for imaging solar system objects. 
Includes routines for fitting a limb-darkened disk model to visibilities in a CASA MS 

## pipelines

Contains a modified version of the EVLA scripted pipeline validated for CASA 5.7 and a modified version of the scripted pipeline that can be used for older VLA observations

## analysis_scripts
A clone of CASA Analysis Utilities developed by the NRAO. Partial documentation available at https://safe.nrao.edu/wiki/bin/view/Main/CasaExtensions. 
This can be imported from within CASA by adding the analysis_scripts/ folder to the path and running 
```
import analysisUtils
```
