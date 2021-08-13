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

# PREPARE FOR CALIBRATIONS

# Fill models for all primary calibrators
# NB: in CASA 3.4.0 can only set models based on field ID and spw, not
# by intents or scans

logprint ("Starting VLA_pipe_calprep.py", logfileout='logs/calprep.log')
time_list=runtiming('calprep', 'start')
QA2_calprep='Pass'

logprint ("Setting models for standard primary calibrators", logfileout='logs/calprep.log')

tb.open(ms_active)

positions = []

for ii in range(0,len(field_positions[0][0])):
    positions.append([field_positions[0][0][ii], field_positions[1][0][ii]])

standard_source_names = [ '3C48', '3C138', '3C147', '3C286' ]
standard_source_fields = find_standards(positions)

standard_source_found = False
for standard_source_field in standard_source_fields:
    if standard_source_field:
        standard_source_found = True
if not standard_source_found:
    standard_source_found = False
    logprint ("ERROR: No standard flux density calibrator observed, switching to manual input", logfileout='logs/calprep.log')
    print('Switching to manual input. WARNING: This currently does nothing. Flux scale will be arbitrary until fixed')
    print('Source Catalog Tool: https://science.nrao.edu/facilities/vla/observing/callist')
    for ids in flux_state_IDs: 
        ra = np.degrees(positions[ids][0])
        ra_h = ra / 15 
        rh = np.floor(ra_h)
        rm = np.floor((ra_h - rh) * 60)
        rs = (ra_h - rh - rm / 60) * 3600
        if np.sign(rh) == -1: 
            rh = 24 + rh
        print('')
        dec = np.degrees(positions[ids][1])
        d = np.floor(dec)
        m = np.floor((dec - d) * 60)
        s = (dec - d - m / 60) * 3600
        print('Flux field {0}, RA:{1},{2},{3}, DEC: {4}, {5}, {6}'.format(field_names[ids], rh, rm, rs, d, m, s))



ii=0
for fields in standard_source_fields:
    for myfield in fields:
        spws = field_spws[myfield]
        for myspw in spws:
            reference_frequency = center_frequencies[myspw]
            EVLA_band = find_EVLA_band(reference_frequency)
            logprint ("Center freq for spw "+str(myspw)+" = "+str(reference_frequency)+", observing band = "+EVLA_band, logfileout='logs/calprep.log')

            model_image = standard_source_names[ii]+'_'+EVLA_band+'.im'

            logprint ("Setting model for field "+str(myfield)+" spw "+str(myspw)+" using "+model_image, logfileout='logs/calprep.log')
            try:
            	default('setjy')
            	vis=ms_active
            	field=str(myfield)
            	spw=str(myspw)
            	selectdata=False
            	scalebychan=True
            	standard='Perley-Butler 2017'
            	model=model_image
            	listmodels=False
            	usescratch=scratch
            	setjy()
            except:
               logprint('no data found for field ' + str(myfield)+" spw "+str(myspw), logfileout='logs/calprep.log')    
    ii=ii+1

tb.close()

logprint("Finished setting models for known calibrators", logfileout='logs/calprep.log')

logprint ("Finished VLA_pipe_calprep.py", logfileout='logs/calprep.log')
logprint ("QA2 score: "+QA2_calprep, logfileout='logs/calprep.log')
time_list=runtiming('calprep', 'end')

pipeline_save()
