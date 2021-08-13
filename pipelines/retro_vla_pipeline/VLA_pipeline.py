######################################################################
#
# Copyright (C) 2013
# Associated Universities, Inc. Washington DC, USA,
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
# License for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
#
# Correspondence concerning VLA Pipelines should be addressed as follows:
#    Please register and submit helpdesk tickets via: https://help.nrao.edu
#    Postal address:
#              National Radio Astronomy Observatory
#              VLA Pipeline Support Office
#              PO Box O
#              Socorro, NM,  USA
#
######################################################################
# EVLA pipeline
# For continuum modes (contiguous spws within a baseband)
# May work for other modes as well
#
# 06/13/12 C. Chandler
# 07/20/12 B. Kent
# 02/05/13 C. Chandler initial release for CASA 4.1
# 09/23/14 C. Chandler modified to work on CASA 4.2.2, and updated to
#          use Perley-Butler 2013 flux density scale
# 06/08/15 C. Chandler modified for CASA 4.3.1
# 06/08/15 C. Chandler modified for CASA 4.4.0
# 07/13/15 E. Momjian for CASA 4.4.0 split target rflag and final uv
#          plots into two scripts
#          Separate calibrators and targets in final rflag
#          Separate calibrators and targets in statwt
# 07/29/15 E. Momjian force SETJY to create the model column, otherwise
#          calibration will not be correct
# 08/25/15 C. Chandler moved plots to after statwt
# 10/13/15 E. Momjian modified for CASA 4.5.0
#          The use of the real vs. virtual model in setjy is a choice (y/n)
#          at the start of the pipeline
# 02/20/16 E. Momjian modified for CASA 4.5.2
#          Using mstransform based split2 instead of split      
# 04/12/16 E. Momjian modified for CASA 4.5.3
# 04/12/16 E. Momjian modified for CASA 4.6.0
#          Mstransform based split2 has been renamed as split.
#          Also using Mstransform based hanningsmooth. 
# 10/10/16 E. Momjian modified for CASA 4.7.0
#          Added Amp & phase vs. frequency plots in weblog
# 03/08/17 E. Momjian modified for CASA 5.0.0 
# 05/01/18 E. Momjian modified for CASA 5.3.0
# 05/23/19 E. Momjian modified to use the Perley-Butler 2017 flux density scale
######################################################################

# Change version and date below with each svn commit.  Note changes in the
# .../trunk/doc/CHANGELOG.txt and .../trunk/doc/bugs_features.txt files

version = "1.5.0"
svnrevision = '12nnn'
date = "2018May01"

print("Pipeline version modified for old VLA Uranus reductions")
import sys

plots = False

# Define location of pipeline
pipepath = os.getcwd() + '/'

#This is the default time-stamped casa log file, in case we
#    need to return to it at any point in the script
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

maincasalog = casalogger.func_globals['casa']['files']['logfile']


def logprint(msg, logfileout=maincasalog):
    print(msg)
    casalog.setlogfile(logfileout)
    casalog.post(msg)
    casalog.setlogfile(maincasalog)
    casalog.post(msg)
    return

#Create timing profile list and file if they don't already exist
if 'time_list' not in globals():
    time_list = []

timing_file = 'logs/timing.log'

if not os.path.exists(timing_file):
    timelog = open(timing_file, 'w')
else:
    timelog = open(timing_file, 'a')
    
def runtiming(pipestate, status):
    '''Determine profile for a given state/stage of the pipeline
    '''
    time_list.append({'pipestate':pipestate, 'time':time.time(), 'status':status})
#    
    if (status == "end"):
        timelog=open(timing_file,'a')
        timelog.write(pipestate+': '+str(time_list[-1]['time'] - time_list[-2]['time'])+' sec \n')
        timelog.flush()
        timelog.close()
        #with open(maincasalog, 'a') as casalogfile:
        #    tempfile = open('logs/'+pipestate+'.log','r')
        #    casalogfile.write(tempfile.read())
        #    tempfile.close()
        #casalogfile.close()
#        
    return time_list

######################################################################

# The following script includes all the definitions and functions and
# prior inputs needed by a run of the pipeline.

time_list=runtiming('startup', 'start')
execfile(pipepath+'VLA_pipe_startup.py')
time_list=runtiming('startup', 'end')
pipeline_save()

######################################################################

try:

# Outstanding issues
# It is unclear if calibrations are being generated and applied correctly
# as preliminary results look poorly calibrated and are incorrectly flux scale
# More work needed, but it's functional enough now to at least run 
######################################################################

# IMPORT THE DATA TO CASA
    execfile(pipepath+'VLA_pipe_import.py')

######################################################################

# HANNING SMOOTH (OPTIONAL, MAY BE IMPORTANT IF THERE IS NARROWBAND RFI)

    # execfile(pipepath+'VLA_pipe_hanning.py')

######################################################################

# GET SOME INFORMATION FROM THE MS THAT WILL BE NEEDED LATER, LIST
# THE DATA, AND MAKE SOME PLOTS
    # Notes: Works for now, but go back and write further calibrator logic 
    # and combine redundant fields
    execfile(pipepath+'VLA_pipe_msinfo.py')

######################################################################

# DETERMINISTIC FLAGGING:
# TIME-BASED: online flags, shadowed data, zeroes, pointing scans, quacking
    # Notes: Works, removed all band-dependent flagging. 
    execfile(pipepath+'VLA_pipe_flagall.py')

######################################################################

# PREPARE FOR CALIBRATIONS
# Fill model columns for primary calibrators
    # Notes: Works if standard calibrator is observed 
    # and can be detected (proper reference frame)
    # Still need to add manual entry 
    execfile(pipepath+'VLA_pipe_calprep.py')

######################################################################

# PRIOR CALIBRATIONS
# Gain curves, opacities, antenna position corrections, 
# requantizer gains (NB: requires CASA 4.1 or later!).  Also plots switched
# power tables, but these are not currently used in the calibration
    # Notes: Works, although unsure if accurate
    # Weather is modelled
    execfile(pipepath+'VLA_pipe_priorcals.py')

#*********************************************************************


# DETERMINE SOLINT FOR SCAN-AVERAGE EQUIVALENT
    # Notes: appears to work 
    execfile(pipepath+'VLA_pipe_solint.py')

######################################################################

# DO TEST GAIN CALIBRATIONS TO ESTABLISH SHORT SOLINT
    # Notes: Appears to work 
    execfile(pipepath+'VLA_pipe_testgains.py')

#*********************************************************************

# MAKE GAIN TABLE FOR FLUX DENSITY BOOTSTRAPPING
# Make a gain table that includes gain and opacity corrections for final
# amp cal, for flux density bootstrapping
    # Notes: appears to work 
    execfile(pipepath+'VLA_pipe_fluxgains.py')

######################################################################

# FLAG GAIN TABLE PRIOR TO FLUX DENSITY BOOTSTRAPPING
# NB: need to break here to flag the gain table interatively, if
# desired; not included in real-time pipeline

#    execfile(pipepath+'VLA_pipe_fluxflag.py')

#*********************************************************************

# DO THE FLUX DENSITY BOOTSTRAPPING -- fits spectral index of
# calibrators with a power-law and puts fit in model column
    # Notes: appears to work 
    execfile(pipepath+'VLA_pipe_fluxboot.py')

######################################################################

# MAKE FINAL CALIBRATION TABLES
    # Appears to work 
    execfile(pipepath+'VLA_pipe_finalcals.py')

######################################################################

# APPLY ALL CALIBRATIONS AND CHECK CALIBRATED DATA
    # Appears to work 
    # Setting cal weights here instead of separately. good? 
    execfile(pipepath+'VLA_pipe_applycals.py')

######################################################################

# NOW RUN ALL CALIBRATED DATA (INCLUDING TARGET) THROUGH rflag
    # Appears to work 
    execfile(pipepath+'VLA_pipe_targetflag.py')

######################################################################

# CALCULATE DATA WEIGHTS BASED ON ST. DEV. WITHIN EACH SPW
    # Notes: Flags all data. Weights are set at the apply stage
    # execfile(pipepath+'EVLA_pipe_statwt.py')

######################################################################

# MAKE FINAL UV PLOTS

    # execfile(pipepath+'EVLA_pipe_plotsummary.py')

######################################################################

# COLLECT RELEVANT PLOTS AND TABLES

    # execfile(pipepath+'EVLA_pipe_filecollect.py')

######################################################################

# WRITE WEBLOG

    # execfile(pipepath+'EVLA_pipe_weblog.py')

######################################################################

# Quit if there have been any exceptions caught:

except KeyboardInterrupt, keyboardException:
    logprint ("Keyboard Interrupt: " + str(keyboardException))
except Exception, generalException:
    logprint ("Exiting script: " + str(generalException))
