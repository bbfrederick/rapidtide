showxcorr_legacy
----------------

Description:
^^^^^^^^^^^^

	Like rapidtide, but for single time courses.  Takes two text files as input, calculates and displays
	the time lagged crosscorrelation between them, fits the maximum time lag, and estimates
	the significance of the correlation.  It has a range of filtering,
	windowing, and correlation options.  This is the old interface - for new analyses you should use showxcorrx.

Inputs:
^^^^^^^
	showxcorr requires two text files containing timecourses with the same sample rate, one timepoint per line, which are to be correlated, and the sample rate.

Outputs:
^^^^^^^^
	showxcorr outputs everything to standard out, including the Pearson correlation, the maximum cross correlation, the time of maximum cross correlation, and estimates of the significance levels (if specified).  There are no output files.

Usage:
^^^^^^

	::

		usage: showxcorr timecourse1 timecourse2 samplerate [-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [-t] [-w] [-f] [-z FILENAME] [-N TRIALS]

		required arguments:
			timcoursefile1:	text file containing a timeseries
			timcoursefile2:	text file containing a timeseries
			samplerate:	the sample rate of the timecourses, in Hz

		optional arguments:
		    -t            - detrend the data
		    -w            - prewindow the data
		    -l LABEL      - label for the delay value
		    -s STARTTIME  - time of first datapoint to use in seconds in the first file
		    -D DURATION   - amount of data to use in seconds
		    -r RANGE      - restrict peak search range to +/- RANGE seconds (default is
				    +/-15)
		    -d            - turns off display of graph
		    -F            - filter data and regressors from LOWERFREQ to UPPERFREQ.
				    LOWERSTOP and UPPERSTOP can be specified, or will be
				    calculated automatically
		    -V            - filter data and regressors to VLF band
		    -L            - filter data and regressors to LFO band
		    -R            - filter data and regressors to respiratory band
		    -C            - filter data and regressors to cardiac band
		    -T            - trim data to match
		    -A            - print data on a single summary line
		    -a            - if summary mode is on, add a header line showing what values
				    mean
		    -f            - negate (flip) second regressor
		    -z FILENAME   - use the columns of FILENAME as controlling variables and
				    return the partial correlation
		    -N TRIALS     - estimate significance thresholds by Monte Carlo with TRIALS
				    repetition


