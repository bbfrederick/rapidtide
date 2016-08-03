#!/bin/csh
#
#       $Author: frederic $
#       $Date: 2013/12/17 17:54:46 $
#       $Id: gensimbold,v 1.14 2013/12/17 17:54:46 frederic Exp $
#

set thesourcedir=.
set thedestdir=.
set meanval=1000.0
set noisepct=0.005
set noiselevel=`echo $meanval $noisepct | awk '{print $1*$2}'`
set samplerate=12.5
set starttime=656.4
set thbfile=$thesourcedir'/lf_tHb'
set hbofile=$thesourcedir'/lf_HbO'
set hbrfile=$thesourcedir'/lf_HbR'
set fmrifile=$thesourcedir'/fmri.nii.gz'
set numskip=30
set sliceordertype=0	# no slice delay
echo 'simulated NIRS timecourse has a sample rate of '$samplerate' and starts at '$starttime

set makesimsources='True'

if ( $makesimsources == 'True' ) then
    # pull one volume out of our example bold file to serve as a basis for constructing our dataset
    fslroi $fmrifile $thesourcedir'/sim_lags' 0 1

    # do simsubject1
    # make a 'head' with a mean value of 1000.0 and a matching mask
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add 1000.0 -roi 16 32 16 32 5 20 0 1 $thesourcedir'/sim_mean'
    fslmaths $thesourcedir'/sim_mean' -thr 1.0 -bin $thesourcedir'/sim_mask'

    # define an area within the 'head' with a BOLD signal
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add 1.0 -roi 16 32 16 32 5  5 0 1 $thesourcedir'/boldpc1'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add 2.0 -roi 16 32 16 32 10 5 0 1 $thesourcedir'/boldpc2'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add 3.5 -roi 16 32 16 32 15 5 0 1 $thesourcedir'/boldpc3'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add 4.0 -roi 16 32 16 32 20 5 0 1 $thesourcedir'/boldpc4'
    fslmaths $thesourcedir'/boldpc1' \
	-add $thesourcedir'/boldpc2' \
	-add $thesourcedir'/boldpc3' \
	-add $thesourcedir'/boldpc4' \
        $thesourcedir'/sim_boldpc'
    
    # define 9 columns of various delay values
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add -3.5 -roi 20 6 20 6 5 20 0 1 $thesourcedir'/lag1'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add -2.5 -roi 20 6 29 6 5 20 0 1 $thesourcedir'/lag2'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add -1.5 -roi 20 6 38 6 5 20 0 1 $thesourcedir'/lag3'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add -0.5 -roi 29 6 20 6 5 20 0 1 $thesourcedir'/lag4'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add  0.0 -roi 29 6 29 6 5 20 0 1 $thesourcedir'/lag5'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add  0.5 -roi 29 6 38 6 5 20 0 1 $thesourcedir'/lag6'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add  1.5 -roi 38 6 20 6 5 20 0 1 $thesourcedir'/lag7'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add  2.5 -roi 38 6 29 6 5 20 0 1 $thesourcedir'/lag8'
    fslmaths $thesourcedir'/sim_lags' -mul 0.0 -add  3.5 -roi 38 6 38 6 5 20 0 1 $thesourcedir'/lag9'
    fslmaths $thesourcedir'/lag1' \
	-add $thesourcedir'/lag2' \
	-add $thesourcedir'/lag3' \
	-add $thesourcedir'/lag4' \
	-add $thesourcedir'/lag5' \
	-add $thesourcedir'/lag6' \
	-add $thesourcedir'/lag7' \
	-add $thesourcedir'/lag8' \
	-add $thesourcedir'/lag9' \
	$thesourcedir'/sim_lags'
    
    echo $starttime > $thesourcedir'/sim_starttime.txt'
    echo $samplerate > $thesourcedir'/sim_samplerate.txt'
    
    # clean up
    rm $thesourcedir'/boldpc'[1-9]'.nii.gz' $thesourcedir'/lag'[1-9]'.nii.gz'
    endif

# generate the simulated datasets from the source files
simdata $fmrifile $thesourcedir'/sim_mean.nii.gz' \
    $thesourcedir'/sim_boldpc.nii.gz' \
    $thesourcedir'/sim_lags.nii.gz' \
    $hbrfile $samplerate $numskip $starttime $thesourcedir'/sim_boldtemp_HbR' $noiselevel $sliceordertype
simdata $fmrifile $thesourcedir'/sim_mean.nii.gz' \
    $thesourcedir'/sim_boldpc.nii.gz' \
    $thesourcedir'/sim_lags.nii.gz' \
    $thbfile $samplerate $numskip $starttime $thesourcedir'/sim_boldtemp_tHb' $noiselevel $sliceordertype
fslmaths $thesourcedir'/sim_boldtemp_tHb' -add $thesourcedir'/sim_boldtemp_HbR' \
    -div 2 $thesourcedir'/sim_boldtemp'
fslmaths $thesourcedir'/sim_boldtemp' -mul 1.0 $thesourcedir'/sim_bolddata' -odt short

simdata $fmrifile $thesourcedir'/sim_mean.nii.gz' \
    $thesourcedir'/sim_boldpc.nii.gz' \
    $thesourcedir'/sim_lags.nii.gz' \
    $thbfile $samplerate $numskip $starttime $thesourcedir'/sim_boldtemp' 0.0 $sliceordertype
fslmaths $thesourcedir'/sim_boldtemp' -mul 1.0 $thesourcedir'/sim_bolddata_nonoise' -odt short

rm $thesourcedir'/sim_boldtemp'*'.nii.gz'
