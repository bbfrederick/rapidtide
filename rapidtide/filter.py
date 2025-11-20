#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
"""This module contains all the filtering operations for the rapidtide
package.

"""

import sys
import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rapidtide.decorators import conditionaljit, conditionaljit2
from rapidtide.ffttools import optfftlen

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

from scipy import fftpack, ndimage, signal
from scipy.signal import savgol_filter

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()

# --------------------------- Filtering functions -------------------------------------------------
# NB: No automatic padding for precalculated filters

class NoncausalFilter:
    def __init__(
        self,
        filtertype="None",
        transitionfrac=0.05,
        transferfunc="trapezoidal",
        initlowerstop=None,
        initlowerpass=None,
        initupperpass=None,
        initupperstop=None,
        butterworthorder=6,
        correctfreq=True,
        padtime=30.0,
        padtype="reflect",
        debug=False,
    ):
        """
        Initialize a zero time delay filter for one-dimensional signals, especially physiological ones.

        This constructor sets up the filter parameters and initializes the filter type.
        The filter can be configured for various physiological signal processing tasks,
        including VLF, LFO, respiratory, cardiac, and HRV-related filtering.

        Parameters
        ----------
        filtertype : {'None', 'vlf', 'lfo', 'resp', 'cardiac', 'vlf_stop', 'lfo_stop', 'resp_stop', 'card_stop',
                     'hrv_ulf', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_vhf', 'hrv_ulf_stop', 'hrv_vlf_stop',
                     'hrv_lf_stop', 'hrv_hf_stop', 'hrv_vhf_stop', 'arb', 'arb_stop', 'ringstop'}, optional
            The type of filter to apply. Default is 'None'.
        transitionfrac : float, optional
            Fraction of the transition band used for filter transition. Default is 0.05.
        transferfunc : {'trapezoidal', 'butterworth'}, optional
            Transfer function to use for filter design. Default is 'trapezoidal'.
        initlowerstop : float, optional
            Initial lower stop frequency for 'arb' and 'arb_stop' filters. Default is None.
        initlowerpass : float, optional
            Initial lower pass frequency for 'arb' and 'arb_stop' filters. Default is None.
        initupperpass : float, optional
            Initial upper pass frequency for 'arb' and 'arb_stop' filters. Default is None.
        initupperstop : float, optional
            Initial upper stop frequency for 'arb' and 'arb_stop' filters. Default is None.
        butterworthorder : int, optional
            Order of the Butterworth filter. Default is 6.
        correctfreq : bool, optional
            Whether to correct impossible pass frequencies. Default is True.
        padtime : float, optional
            Amount of time (in seconds) to pad the signal to reduce edge effects. Default is 30.0.
        padtype : {'reflect', 'zero', 'constant'}, optional
            Type of padding to use. Default is 'reflect'.
        debug : bool, optional
            Enable extended debugging messages. Default is False.

        Returns
        -------
        None
            This method initializes the instance and does not return any value.

        Notes
        -----
        For 'arb' and 'arb_stop' filter types, the pass and stop frequencies are initialized
        based on the provided values or default values if not specified.
        The default frequencies for 'arb' filters are:
            - lowerpass = 0.05 Hz
            - lowerstop = 0.9 * lowerpass
            - upperpass = 0.20 Hz
            - upperstop = 1.1 * upperpass

        Examples
        --------
        >>> filter_instance = ZeroDelayFilter(filtertype='resp', padtime=60.0)
        >>> filter_instance.settype('cardiac')
        >>> filter_instance.gettype()
        'cardiac'
        """
        self.filtertype = filtertype
        self.species = "human"
        self.transitionfrac = transitionfrac
        self.transferfunc = transferfunc
        if initlowerpass is None:
            self.arb_lowerpass = 0.05
            self.arb_lowerstop = 0.9 * self.arb_lowerpass
        else:
            self.arb_lowerpass = initlowerpass
            self.arb_lowerstop = initlowerstop
        if initupperpass is None:
            self.arb_upperpass = 0.20
            self.arb_upperstop = 1.1 * self.arb_upperpass
        else:
            self.arb_upperpass = initupperpass
            self.arb_upperstop = initupperstop
        self.lowerstop = 0.0
        self.lowerpass = 0.0
        self.upperpass = -1.0
        self.upperstop = -1.0
        self.butterworthorder = butterworthorder
        self.correctfreq = correctfreq
        self.padtime = padtime
        self.padtype = padtype
        self.debug = debug

        self.settype(self.filtertype)

    def settype(self, thetype):
        """
        Set the filter type and corresponding frequency bands for the filter object.

        This method configures the filter parameters based on the specified filter type.
        It assigns passband and stopband frequencies depending on the filter type,
        using predefined frequency ranges or user-defined values for arbitrary filters.

        Parameters
        ----------
        thetype : str
            The type of filter to set. Supported values include:
            - "vlf", "vlf_stop": Very Low Frequency
            - "lfo", "lfo_stop": Low Frequency Oscillation
            - "lfo_legacy", "lfo_legacy_stop": Legacy Low Frequency Oscillation
            - "lfo_tight", "lfo_tight_stop": Tight Low Frequency Oscillation
            - "resp", "resp_stop": Respiration
            - "cardiac", "cardiac_stop": Cardiac
            - "hrv_ulf", "hrv_ulf_stop": HRV Ultra Low Frequency
            - "hrv_vlf", "hrv_vlf_stop": HRV Very Low Frequency
            - "hrv_lf", "hrv_lf_stop": HRV Low Frequency
            - "hrv_hf", "hrv_hf_stop": HRV High Frequency
            - "hrv_vhf", "hrv_vhf_stop": HRV Very High Frequency
            - "arb", "arb_stop": Arbitrary filter with custom frequency limits

        Notes
        -----
        For arbitrary filters ("arb" or "arb_stop"), the method uses the following
        attributes from the object:
        - `self.arb_lowerstop`
        - `self.arb_lowerpass`
        - `self.arb_upperpass`
        - `self.arb_upperstop`

        For all other filter types, the method calls `getfilterbandfreqs` with the
        specified filter type and additional parameters like `transitionfrac` and `species`.

        Examples
        --------
        >>> obj.settype("lfo")
        >>> print(obj.lowerpass)
        0.01
        >>> print(obj.upperstop)
        0.5
        """
        self.filtertype = thetype
        if self.filtertype == "vlf" or self.filtertype == "vlf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "vlf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "lfo" or self.filtertype == "lfo_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "lfo", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "lfo_legacy" or self.filtertype == "lfo_legacy_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "lfo_legacy", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "lfo_tight" or self.filtertype == "lfo_tight_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "lfo_tight", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "resp" or self.filtertype == "resp_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "resp", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "cardiac" or self.filtertype == "cardiac_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "cardiac", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "hrv_ulf" or self.filtertype == "hrv_ulf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "hrv_ulf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "hrv_vlf" or self.filtertype == "hrv_vlf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "hrv_vlf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "hrv_lf" or self.filtertype == "hrv_lf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "hrv_lf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "hrv_hf" or self.filtertype == "hrv_hf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "hrv_hf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "hrv_vhf" or self.filtertype == "hrv_vhf_stop":
            self.lowerpass, self.upperpass, self.lowerstop, self.upperstop = getfilterbandfreqs(
                "hrv_vhf", transitionfrac=self.transitionfrac, species=self.species
            )
        elif self.filtertype == "arb" or self.filtertype == "arb_stop":
            self.lowerstop = 1.0 * self.arb_lowerstop
            self.lowerpass = 1.0 * self.arb_lowerpass
            self.upperpass = 1.0 * self.arb_upperpass
            self.upperstop = 1.0 * self.arb_upperstop
        else:
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = 1.0e20
            self.upperstop = 1.0e20

    def gettype(self):
        """
        Return the filter type of the object.

        Returns
        -------
        filtertype : str or int
            The filter type associated with the object. The specific type depends
            on the implementation of the filtertype attribute.

        Notes
        -----
        This method provides access to the internal filtertype attribute.
        The return value type may vary depending on the specific implementation
        of the class that contains this method.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.gettype()
        'some_filter_type'
        """
        return self.filtertype

    def setbutterorder(self, order=3):
        """
        Set the Butterworth filter order for the system.

        This method assigns the specified order to the Butterworth filter configuration.
        The order determines the steepness of the filter's roll-off characteristics.

        Parameters
        ----------
        order : int, optional
            The order of the Butterworth filter. Must be a positive integer.
            Default is 3.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        A higher filter order results in a steeper roll-off but may introduce
        more phase distortion. The order should be chosen based on the specific
        requirements of the signal processing application.

        Examples
        --------
        >>> system = SomeFilterClass()
        >>> system.setbutterorder(5)
        >>> print(system.butterworthorder)
        5

        >>> system.setbutterorder()
        >>> print(system.butterworthorder)
        3
        """
        self.butterworthorder = order

    def setdebug(self, debug):
        """
        Set the debug flag for the object.

        Parameters
        ----------
        debug : bool
            If True, enables debug mode. If False, disables debug mode.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method sets the internal `debug` attribute of the object. When debug mode
        is enabled, additional logging or verbose output may be generated during
        object operations.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setdebug(True)
        >>> print(obj.debug)
        True
        """
        self.debug = debug

    def setpadtime(self, padtime):
        """
        Set the padding time for the object.

        Parameters
        ----------
        padtime : float or int
            The padding time value to be assigned to the object's padtime attribute.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided padtime value to the instance's padtime attribute,
        replacing any existing value.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setpadtime(5.0)
        >>> print(obj.padtime)
        5.0
        """
        self.padtime = padtime

    def getpadtime(self):
        """
        Return the padding time value.

        Returns
        -------
        padtime : float or int
            The padding time value stored in the instance variable `self.padtime`.

        Notes
        -----
        This is a simple getter method that returns the value of the internal
        `padtime` attribute. The actual meaning and units of this value depend
        on the context in which the class is used.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.padtime = 5.0
        >>> obj.getpadtime()
        5.0
        """
        return self.padtime

    def setpadtype(self, padtype):
        """
        Set the padding type for the object.

        Parameters
        ----------
        padtype : str
            The padding type to be set. This parameter determines how padding
            will be applied in subsequent operations.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided padding type to the internal
        `padtype` attribute of the object. The valid values for `padtype` depend
        on the specific implementation of the class this method belongs to.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setpadtype('constant')
        >>> print(obj.padtype)
        'constant'
        """
        self.padtype = padtype

    def getpadtype(self):
        """
        Return the padding type of the object.

        Returns
        -------
        str
            The padding type as a string identifier.

        Notes
        -----
        This method provides access to the internal `padtype` attribute
        which defines the padding behavior for the object.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.padtype = 'constant'
        >>> obj.getpadtype()
        'constant'
        """
        return self.padtype

    def settransferfunc(self, transferfunc):
        """
        Set the transfer function for the system.

        Parameters
        ----------
        transferfunc : callable
            The transfer function to be assigned to the system. This should be a
            callable object that defines the system's transfer function behavior.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided transfer function to the
        internal `transferfunc` attribute of the object. The transfer function
        should be compatible with the system's expected input and output formats.

        Examples
        --------
        >>> system = MySystem()
        >>> def my_transfer_func(x):
        ...     return x * 2
        >>> system.settransferfunc(my_transfer_func)
        >>> system.transferfunc(5)
        10
        """
        self.transferfunc = transferfunc

    def setfreqs(self, lowerstop, lowerpass, upperpass, upperstop):
        """
        Set frequency parameters for filter design.

        This method configures the frequency boundaries for a filter design, ensuring
        proper causal relationships between the stopband and passband frequencies.

        Parameters
        ----------
        lowerstop : float
            Lower stopband frequency boundary. Must be less than or equal to lowerpass.
        lowerpass : float
            Lower passband frequency boundary. Must be greater than or equal to lowerstop.
        upperpass : float
            Upper passband frequency boundary. Must be less than or equal to upperstop.
        upperstop : float
            Upper stopband frequency boundary. Must be greater than or equal to upperpass.

        Returns
        -------
        None
            This method does not return a value but modifies instance attributes.

        Notes
        -----
        The method performs validation checks to ensure causal filter design:
        - lowerstop must be <= lowerpass
        - upperstop must be >= upperpass
        - lowerpass must be < upperpass when upperpass >= 0.0

        All frequency values are stored as instance attributes with the prefix 'arb_'
        for internal use and without the prefix for public access.

        Examples
        --------
        >>> filter = NoncausalFilter()
        >>> filter.setfreqs(0.1, 0.2, 0.8, 0.9)
        >>> print(filter.lowerstop)
        0.1
        """
        if lowerstop > lowerpass:
            print(
                "NoncausalFilter error: lowerstop (",
                lowerstop,
                ") must be <= lowerpass (",
                lowerpass,
                ")",
            )
            sys.exit()
        if upperpass > upperstop:
            print(
                "NoncausalFilter error: upperstop (",
                upperstop,
                ") must be >= upperpass (",
                upperpass,
                ")",
            )
            sys.exit()
        if (lowerpass > upperpass) and (upperpass >= 0.0):
            print(
                "NoncausalFilter error: lowerpass (",
                lowerpass,
                ") must be < upperpass (",
                upperpass,
                ")",
            )
            sys.exit()
        self.arb_lowerstop = 1.0 * lowerstop
        self.arb_lowerpass = 1.0 * lowerpass
        self.arb_upperpass = 1.0 * upperpass
        self.arb_upperstop = 1.0 * upperstop
        self.lowerstop = 1.0 * self.arb_lowerstop
        self.lowerpass = 1.0 * self.arb_lowerpass
        self.upperpass = 1.0 * self.arb_upperpass
        self.upperstop = 1.0 * self.arb_upperstop

    def getfreqs(self):
        """
        Return frequency boundaries for filter design.

        Returns
        -------
        tuple
            A tuple containing four frequency values in the order:
            (lowerstop, lowerpass, upperpass, upperstop)

        Notes
        -----
        This function returns the frequency boundaries used for filter design specifications.
        The values represent:
        - lowerstop: Lower stopband frequency
        - lowerpass: Lower passband frequency
        - upperpass: Upper passband frequency
        - upperstop: Upper stopband frequency

        Examples
        --------
        >>> filter_obj = FilterDesign()
        >>> freqs = filter_obj.getfreqs()
        >>> print(freqs)
        (100, 200, 300, 400)
        """
        return self.lowerstop, self.lowerpass, self.upperpass, self.upperstop

    def apply(self, Fs, data):
        """
        Apply the filter to a dataset.

        Parameters
        ----------
        Fs : float
            Sample frequency (Hz) of the input data.
        data : 1D float array
            The data to be filtered.

        Returns
        -------
        filtereddata : 1D float array
            The filtered data with the same shape as the input `data`.

        Notes
        -----
        This function applies a filter based on the `filtertype` attribute of the object.
        It performs bounds checking and handles various error conditions, including cases
        where filter frequencies exceed the Nyquist limit or fall below the minimum
        resolvable frequency. If `correctfreq` is True, invalid frequencies are adjusted
        to valid values instead of raising an error.

        The function supports multiple predefined filter types such as 'vlf', 'lfo',
        'cardiac', 'hrv_*' and custom 'arb' types. For stopband filters (e.g., 'vlf_stop'),
        the result is the difference between the input and the filtered signal.

        Examples
        --------
        >>> filtered_data = filter_instance.apply(100.0, data)
        >>> filtered_data = filter_instance.apply(256.0, data)
        """
        # if filterband is None, just return the data
        if self.filtertype == "None":
            return data

        # do some bounds checking
        nyquistlimit = 0.5 * Fs
        lowestfreq = 2.0 * Fs / np.shape(data)[0]

        # first see if entire range is out of bounds
        if self.lowerpass >= nyquistlimit:
            print(
                "NoncausalFilter error: filter lower pass ",
                self.lowerpass,
                " exceeds nyquist frequency ",
                nyquistlimit,
            )
            sys.exit()
        if self.lowerstop >= nyquistlimit:
            print(
                "NoncausalFilter error: filter lower stop ",
                self.lowerstop,
                " exceeds nyquist frequency ",
                nyquistlimit,
            )
            sys.exit()
        if -1.0 < self.upperpass <= lowestfreq:
            print(
                "NoncausalFilter error: filter upper pass ",
                self.upperpass,
                " is below minimum frequency ",
                lowestfreq,
            )
            sys.exit()
        if -1.0 < self.upperstop <= lowestfreq:
            print(
                "NoncausalFilter error: filter upper stop ",
                self.upperstop,
                " is below minimum frequency ",
                lowestfreq,
            )
            sys.exit()

        # now look for fixable errors
        if self.upperpass >= nyquistlimit:
            if self.correctfreq:
                self.upperpass = nyquistlimit
            else:
                print(
                    "NoncausalFilter error: filter upper pass ",
                    self.upperpass,
                    " exceeds nyquist frequency ",
                    nyquistlimit,
                )
                sys.exit()
        if self.upperstop > nyquistlimit:
            if self.correctfreq:
                self.upperstop = nyquistlimit
            else:
                print(
                    "NoncausalFilter error: filter upper stop ",
                    self.upperstop,
                    " exceeds nyquist frequency ",
                    nyquistlimit,
                )
                sys.exit()
        if self.lowerpass < lowestfreq:
            if self.correctfreq:
                self.lowerpass = lowestfreq
            else:
                print(
                    "NoncausalFilter error: filter lower pass ",
                    self.lowerpass,
                    " is below minimum frequency ",
                    lowestfreq,
                )
                sys.exit()
        if self.lowerstop < lowestfreq:
            if self.correctfreq:
                self.lowerstop = lowestfreq
            else:
                print(
                    "NoncausalFilter error: filter lower stop ",
                    self.lowerstop,
                    " is below minimum frequency ",
                    lowestfreq,
                )
                sys.exit()

        if self.padtime < 0.0:
            padlen = int(len(data) // 2)
        else:
            padlen = int(self.padtime * Fs)
        if self.lowerpass <= 0.0:
            avlen = 1
        else:
            avlen = np.min([int(Fs / self.lowerpass), padlen])
        if self.debug:
            print("Fs=", Fs)
            print("lowerstop=", self.lowerstop)
            print("lowerpass=", self.lowerpass)
            print("upperpass=", self.upperpass)
            print("upperstop=", self.upperstop)
            print("butterworthorder=", self.butterworthorder)
            print("padtime=", self.padtime)
            print("padlen=", padlen)
            print("avlen=", avlen)
            print("padtype=", self.padtype)

        # now do the actual filtering
        if self.filtertype == "None":
            return data
        elif self.filtertype == "ringstop":
            return arb_pass(
                Fs,
                data,
                0.0,
                0.0,
                Fs / 4.0,
                1.1 * Fs / 4.0,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                padtype=self.padtype,
                debug=self.debug,
            )
        elif (
            self.filtertype == "vlf"
            or self.filtertype == "lfo"
            or self.filtertype == "lfo_legacy"
            or self.filtertype == "lfo_tight"
            or self.filtertype == "resp"
            or self.filtertype == "cardiac"
            or self.filtertype == "hrv_ulf"
            or self.filtertype == "hrv_vlf"
            or self.filtertype == "hrv_lf"
            or self.filtertype == "hrv_hf"
            or self.filtertype == "hrv_vhf"
        ):
            return arb_pass(
                Fs,
                data,
                self.lowerstop,
                self.lowerpass,
                self.upperpass,
                self.upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                avlen=avlen,
                padtype=self.padtype,
                debug=self.debug,
            )
        elif (
            self.filtertype == "vlf_stop"
            or self.filtertype == "lfo_stop"
            or self.filtertype == "lfo_legacy_stop"
            or self.filtertype == "lfo_tight_stop"
            or self.filtertype == "resp_stop"
            or self.filtertype == "cardiac_stop"
            or self.filtertype == "hrv_ulf_stop"
            or self.filtertype == "hrv_vlf_stop"
            or self.filtertype == "hrv_lf_stop"
            or self.filtertype == "hrv_hf_stop"
            or self.filtertype == "hrv_vhf_stop"
        ):
            return data - arb_pass(
                Fs,
                data,
                self.lowerstop,
                self.lowerpass,
                self.upperpass,
                self.upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                avlen=avlen,
                padtype=self.padtype,
                debug=self.debug,
            )
        elif self.filtertype == "arb":
            return arb_pass(
                Fs,
                data,
                self.arb_lowerstop,
                self.arb_lowerpass,
                self.arb_upperpass,
                self.arb_upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                avlen=avlen,
                padtype=self.padtype,
                debug=self.debug,
            )
        elif self.filtertype == "arb_stop":
            return data - arb_pass(
                Fs,
                data,
                self.arb_lowerstop,
                self.arb_lowerpass,
                self.arb_upperpass,
                self.arb_upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                avlen=avlen,
                padtype=self.padtype,
                debug=self.debug,
            )
        else:
            print(f"bad filter type: {self.filtertype}")
            sys.exit()


@conditionaljit()
def padvec(
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Returns a padded copy of the input data; padlen points of
    filled data are prepended and appended to the input data to reduce
    end effects when the data is then filtered.  Filling can be "zero", "reflect", "cyclic", "constant",
    or "constant+".

    Parameters
    ----------
    inputdata : NDArray
        An array of any numerical type.
    padlen : int, optional
        The number of points to add to each end. Default is 20.
    avlen : int, optional
        The number of points to average when doing "constant+" padding. Default is 20.
    padtype : str, optional
        Method for padding data on the ends of the vector. Options are "reflect", "zero", "cyclic",
        "constant", or "constant+". Default is "reflect".
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    NDArray
        The input data, with `padlen` reflected points added to each end.

    Notes
    -----
    This function is useful for reducing edge effects when filtering data. The padding methods are as follows:
    - "reflect": pads by reflecting the input array around its edges.
    - "zero": pads with zeros.
    - "cyclic": pads by cycling the input array.
    - "constant": pads with the first/last value of the input array.
    - "constant+": pads with the mean of the first/last `avlen` points of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> padded = padvec(data, padlen=2, padtype="reflect")
    >>> print(padded)
    [3 2 1 2 3 4 5 5 4]

    >>> padded = padvec(data, padlen=2, padtype="zero")
    >>> print(padded)
    [0 0 1 2 3 4 5 0 0]
    """
    if debug:
        print(
            "padvec: padlen=",
            padlen,
            ", avlen=",
            avlen,
            ", padtype=",
            padtype,
            "len(inputdata)=",
            len(inputdata),
        )
    if padlen > len(inputdata):
        raise RuntimeError(
            f"ERROR: padlen ({padlen}) is greater than input data length ({len(inputdata)})"
        )
    if avlen > padlen:
        avlen = padlen

    inputdtype = inputdata.dtype
    if padlen > 0:
        if padtype == "reflect":
            return np.concatenate(
                (inputdata[::-1][-padlen:], inputdata, inputdata[::-1][0:padlen])
            )
        elif padtype == "zero":
            return np.concatenate(
                (
                    np.zeros((padlen), dtype=inputdtype),
                    inputdata,
                    np.zeros((padlen), dtype=inputdtype),
                )
            )
        elif padtype == "cyclic":
            return np.concatenate((inputdata[-padlen:], inputdata, inputdata[0:padlen]))
        elif padtype == "constant":
            return np.concatenate(
                (
                    inputdata[0] * np.ones((padlen), dtype=inputdtype),
                    inputdata,
                    inputdata[-1] * np.ones((padlen), dtype=inputdtype),
                )
            )
        elif padtype == "constant+":
            startval = np.mean(inputdata[0:avlen])
            endval = np.mean(inputdata[-avlen:])
            return np.concatenate(
                (
                    (startval * np.ones((padlen), dtype=inputdtype)).astype(inputdtype),
                    inputdata,
                    (endval * np.ones((padlen), dtype=inputdtype)).astype(inputdtype),
                )
            )
        else:
            raise ValueError(
                "Padtype must be one of 'reflect', 'zero', 'cyclic', 'constant', or 'constant+'."
            )
    else:
        return inputdata


@conditionaljit()
def unpadvec(inputdata: NDArray, padlen: int = 20) -> NDArray:
    """
    Returns input data with the end pads removed.

    This function removes padding from both ends of an array. It is the inverse
    operation of the `padvec` function, which adds padding to the array.

    Parameters
    ----------
    inputdata : NDArray
        An array of any numerical type.
    padlen : int, optional
        The number of points to remove from each end. Default is 20.

    Returns
    -------
    NDArray
        The input data with padding removed from both ends. If padlen is 0 or
        negative, the original array is returned unchanged.

    Notes
    -----
    When padlen is greater than 0, the function returns ``inputdata[padlen:-padlen]``.
    If padlen is greater than or equal to the array length, an empty array will be returned.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> unpadvec(data, padlen=2)
    array([3, 4, 5, 6])

    >>> unpadvec(data, padlen=0)
    array([1, 2, 3, 4, 5, 6, 7, 8])
    """
    if padlen > 0:
        return inputdata[padlen:-padlen]
    else:
        return inputdata


def ssmooth(xsize: float, ysize: float, zsize: float, sigma: float, inputdata: NDArray) -> NDArray:
    """
    Applies an isotropic gaussian spatial filter to a 3D array.

    This function applies a Gaussian filter to 3D spatial data with isotropic
    filtering parameters. The filter kernel width is specified in spatial units
    and is converted to pixel units based on the array spacing parameters.

    Parameters
    ----------
    xsize : float
        The array x step size in spatial units
    ysize : float
        The array y step size in spatial units
    zsize : float
        The array z step size in spatial units
    sigma : float
        The width of the gaussian filter kernel in spatial units
    inputdata : NDArray
        The spatial data to filter

    Returns
    -------
    NDArray
        The filtered spatial data as a 3D float array

    Notes
    -----
    The function uses `scipy.ndimage.gaussian_filter` internally, where the
    sigma parameters are calculated as `sigma / step_size` for each dimension.
    This ensures isotropic filtering when the same sigma value is used across
    all spatial dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> data = np.random.rand(10, 10, 10)
    >>> filtered = ssmooth(0.1, 0.1, 0.1, 0.2, data)
    >>> print(filtered.shape)
    (10, 10, 10)
    """
    return ndimage.gaussian_filter(inputdata, [sigma / xsize, sigma / ysize, sigma / zsize])


# - butterworth filters
# @conditionaljit()
def dolpfiltfilt(
    Fs: float,
    upperpass: float,
    inputdata: NDArray,
    order: int,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs a bidirectional (zero phase) Butterworth lowpass filter on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    upperpass : float
        Upper end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered.
    order : int
        Order of the Butterworth filter.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'cyclic'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a zero-phase Butterworth filter using `scipy.signal.filtfilt`,
    which eliminates phase distortion. Padding is applied before filtering to reduce
    edge effects.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> data = np.random.randn(1000)
    >>> filtered = dolpfiltfilt(Fs, upperpass, data, order=4)
    """
    if upperpass > Fs / 2.0:
        upperpass = Fs / 2.0
    if debug:
        print(
            "dolpfiltfilt - Fs, upperpass, len(inputdata), order:",
            Fs,
            upperpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, 2.0 * upperpass / Fs)
    return unpadvec(
        signal.filtfilt(
            b,
            a,
            padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug),
        ).real,
        padlen=padlen,
    ).astype(np.float64)


# @conditionaljit()
def dohpfiltfilt(
    Fs: float,
    lowerpass: float,
    inputdata: NDArray,
    order: int,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs a bidirectional (zero phase) Butterworth highpass filter on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    inputdata : NDArray
        Input signal to be filtered.
    order : int
        Order of the Butterworth filter.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'wrap'. Default is 'reflect'.
    debug : bool, optional
        If True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data with the same shape as inputdata.

    Notes
    -----
    This function applies a zero-phase Butterworth highpass filter using `scipy.signal.filtfilt`,
    which ensures no phase distortion in the filtered signal. Padding is applied before filtering
    to minimize edge effects.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> Fs = 100.0
    >>> data = np.random.randn(1000)
    >>> filtered = dohpfiltfilt(Fs, 10.0, data, order=4, padlen=30)
    """
    if lowerpass < 0.0:
        lowerpass = 0.0
    if debug:
        print(
            "dohpfiltfilt - Fs, lowerpass, len(inputdata), order:",
            Fs,
            lowerpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, 2.0 * lowerpass / Fs, "highpass")
    return unpadvec(
        signal.filtfilt(
            b,
            a,
            padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug),
        ).real,
        padlen=padlen,
    )


# @conditionaljit()
def dobpfiltfilt(
    Fs: float,
    lowerpass: float,
    upperpass: float,
    inputdata: NDArray,
    order: int,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs a bidirectional (zero phase) Butterworth bandpass filter on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    upperpass : float
        Upper end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered.
    order : int
        Order of the Butterworth filter.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'cyclic'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a zero-phase Butterworth bandpass filter using `scipy.signal.filtfilt`,
    which eliminates phase distortion. Padding is applied before filtering to reduce edge effects.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> Fs = 100.0
    >>> data = np.random.randn(1000)
    >>> filtered = dobpfiltfilt(Fs, 10.0, 30.0, data, order=4, padlen=30)
    """
    if upperpass > Fs / 2.0:
        upperpass = Fs / 2.0
    if lowerpass < 0.0:
        lowerpass = 0.0
    if debug:
        print(
            "dobpfiltfilt - Fs, lowerpass, upperpass, len(inputdata), order:",
            Fs,
            lowerpass,
            upperpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, [2.0 * lowerpass / Fs, 2.0 * upperpass / Fs], "bandpass")
    return unpadvec(
        signal.filtfilt(
            b,
            a,
            padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug),
        ).real,
        padlen=padlen,
    )


# - direct filter with specified transfer function
def transferfuncfilt(inputdata: NDArray, transferfunc: NDArray) -> NDArray:
    """
    Filters input data using a previously calculated transfer function.

    This function applies frequency domain filtering by multiplying the input data's
    Fourier transform with the transfer function, then transforms back to the time domain.

    Parameters
    ----------
    inputdata : NDArray
        Input data to be filtered, array of real or complex values
    transferfunc : NDArray
        The transfer function, array of real or complex values with same length as inputdata

    Returns
    -------
    NDArray
        Filtered input data as a 1D float array

    Notes
    -----
    The filtering is performed in the frequency domain using the convolution theorem.
    The transfer function should be designed to match the frequency response of the desired filter.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> # Create sample data
    >>> data = np.random.randn(1024)
    >>> # Create a simple low-pass filter transfer function
    >>> freq = np.fft.fftfreq(len(data), 1.0)
    >>> tf = np.abs(freq) < 0.1
    >>> # Apply filter
    >>> filtered_data = transferfuncfilt(data, tf)
    """
    inputdata_trans = transferfunc * fftpack.fft(inputdata)
    return fftpack.ifft(inputdata_trans).real


# - fft brickwall filters
def getlpfftfunc(Fs: float, upperpass: float, inputdata: NDArray, debug: bool = False) -> NDArray:
    """
    Generates a brickwall lowpass transfer function.

    This function creates a transfer function that acts as a brickwall lowpass filter
    by setting frequencies above the cutoff to zero. The filter is designed in the
    frequency domain using the FFT domain representation.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
    upperpass : float
        Upper end of passband in Hz
    inputdata : 1D numpy array
        Input data to be filtered
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    transferfunc : 1D float array
        The transfer function with the lowpass filter characteristics

    Notes
    -----
    The function creates a transfer function where frequencies below the cutoff
    are set to 1.0 and frequencies above the cutoff are set to 0.0. The cutoff
    frequency is determined by the ratio of upperpass to Fs, converted to bin
    indices in the FFT domain.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> inputdata = np.random.rand(100)
    >>> transfer_func = getlpfftfunc(Fs, upperpass, inputdata)
    >>> print(transfer_func.shape)
    (100,)
    """
    transferfunc = np.ones(np.shape(inputdata), dtype=np.float64)
    cutoffbin = int((upperpass / Fs) * len(transferfunc))
    if debug:
        print(
            "getlpfftfunc - Fs, upperpass, len(inputdata):",
            Fs,
            upperpass,
            np.shape(inputdata)[0],
        )
    transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


# @conditionaljit()
def dolpfftfilt(
    Fs: float,
    upperpass: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT brickwall lowpass filter on an input vector and returns the result.
    Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    upperpass : float
        Upper end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'wrap'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a lowpass filter in the frequency domain using FFT. The input signal
    is padded at both ends to minimize edge effects caused by the filtering process.
    The padding is performed using the `padvec` function, and the inverse FFT is used to
    transform the filtered signal back to the time domain.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> data = np.random.randn(1000)
    >>> filtered_data = dolpfftfilt(Fs, upperpass, data)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dohpfftfilt(
    Fs: float,
    lowerpass: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT brickwall highpass filter on an input vector and returns the result.
    Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'cyclic'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a highpass filter in the frequency domain using FFT.
    The input signal is first padded to minimize edge effects, then transformed
    into the frequency domain, filtered, and inverse transformed back to the time domain.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> lowerpass = 10.0
    >>> data = np.random.randn(1000)
    >>> filtered = dohpfftfilt(Fs, lowerpass, data)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dobpfftfilt(
    Fs: float,
    lowerpass: float,
    upperpass: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT brickwall bandpass filter on an input vector and returns the result.
    Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    upperpass : float
        Upper end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of averaging window for padding; used only if `padtype` is 'mean'.
        Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect', 'mean', or 'wrap'.
        Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    NDArray
        The filtered data as a 1D float array, with the same shape as inputdata.

    Notes
    -----
    This function applies a brickwall bandpass filter in the frequency domain using FFT.
    The input signal is first padded to minimize edge effects, then transformed into
    the frequency domain, filtered, and transformed back. Padding is applied using
    the specified `padtype` and `padlen`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> fs = 100.0
    >>> t = np.linspace(0, 1, int(fs), endpoint=False)
    >>> x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    >>> filtered = dobpfftfilt(fs, 5, 15, x, padlen=30)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug) * (
        1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    )
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# - fft trapezoidal filters
# @conditionaljit()
def getlptrapfftfunc(
    Fs: float, upperpass: float, upperstop: float, inputdata: NDArray, debug: bool = False
) -> NDArray:
    """
    Generate a trapezoidal lowpass transfer function for filtering.

    This function creates a transfer function with a trapezoidal transition band
    between the passband and stopband, suitable for use in spectral filtering
    operations. The resulting transfer function can be applied to frequency-domain
    data to perform lowpass filtering.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    upperpass : float
        Upper edge of the passband in Hz.
    upperstop : float
        Lower edge of the stopband in Hz.
    inputdata : NDArray
        Input data array (typically frequency domain data) to determine the
        length and shape of the transfer function.
    debug : bool, optional
        If True, print internal state information for debugging purposes.
        Default is False.

    Returns
    -------
    NDArray
        A 1D float array representing the transfer function. The array has the
        same length as `inputdata` and contains values between 0 and 1,
        indicating the attenuation at each frequency bin.

    Notes
    -----
    The transition from passband to stopband is linear (trapezoidal), with
    the transition region defined between `upperpass` and `upperstop`.
    The function assumes that `upperpass < upperstop` and that both are
    within the Nyquist frequency range (0 to Fs/2).

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> upperstop = 30.0
    >>> inputdata = np.zeros(100)
    >>> tf = getlptrapfftfunc(Fs, upperpass, upperstop, inputdata)
    >>> print(tf.shape)
    (100,)
    """
    transferfunc = np.ones(np.shape(inputdata), dtype="float64")
    passbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
    cutoffbin = int((upperstop / Fs) * np.shape(transferfunc)[0])
    transitionlength = cutoffbin - passbin
    if debug:
        print("getlptrapfftfunc - Fs, upperpass, upperstop:", Fs, upperpass, upperstop)
        print(
            "getlptrapfftfunc - passbin, transitionlength, cutoffbin, len(inputdata):",
            passbin,
            transitionlength,
            cutoffbin,
            inputdata.shape,
        )
    if transitionlength > 0:
        transitionvector = np.arange(1.0 * transitionlength) / transitionlength
        transferfunc[passbin:cutoffbin] = 1.0 - transitionvector
        transferfunc[-cutoffbin:-passbin] = transitionvector
    if cutoffbin > 0:
        transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


# @conditionaljit()
def getlptransfunc(
    Fs: float,
    inputdata: NDArray,
    upperpass: Optional[float] = None,
    upperstop: Optional[float] = None,
    type: str = "brickwall",
    debug: bool = False,
) -> NDArray:
    """
    Compute the low-pass transfer function for a given input signal.

    This function generates a transfer function based on the specified type
    (brickwall, Gaussian, or trapezoidal) to be used for filtering purposes.
    The transfer function is applied in the frequency domain to filter the input data.

    Parameters
    ----------
    Fs : float
        Sampling frequency of the input signal in Hz.
    inputdata : NDArray
        Input signal data, used to determine the length of the transfer function.
    upperpass : float, optional
        Upper passband frequency in Hz. Must be specified.
    upperstop : float, optional
        Upper stopband frequency in Hz. Only used for 'trapezoidal' type.
        Defaults to 1.05 * upperpass if not specified.
    type : str, optional
        Type of transfer function to generate. Options are:
        - "brickwall": Ideal low-pass filter with sharp cutoff.
        - "gaussian": Gaussian-shaped transition.
        - "trapezoidal": Trapezoidal transition between pass and stop bands.
        Default is "brickwall".
    debug : bool, optional
        If True, prints debug information and displays the transfer function plot.
        Default is False.

    Returns
    -------
    NDArray
        The computed low-pass transfer function with the same shape as `inputdata`.

    Notes
    -----
    - For 'brickwall' type, the transfer function is 1.0 in the passband and 0.0 in the stopband.
    - For 'gaussian' type, the transition is smoothed using a Gaussian function.
    - For 'trapezoidal' type, a linear transition is applied between pass and stop bands.
    - The function uses the sampling frequency `Fs` to map frequencies to the normalized frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> input_signal = np.random.rand(1024)
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> tf = getlptransfunc(Fs, input_signal, upperpass, type='gaussian')
    >>> print(tf.shape)
    (1024,)
    """
    if upperpass is None:
        print("getlptransfunc: upperpass must be specified")
        sys.exit()
    if debug:
        print("getlptransfunc:")
        print("\tFs:", Fs)
        print("\tnp.shape(inputdata)[0]:", np.shape(inputdata)[0])
        print("\tupperpass:", upperpass)
        print("\tupperstop:", upperstop)
        print("\ttype:", type)
    freqaxis = (
        np.linspace(0.0, 1.0, num=np.shape(inputdata)[0], endpoint=False, dtype="float64") / Fs
    )
    if type == "gaussian":
        halfloc = int(np.shape(inputdata)[0] // 2)
        sigma = upperpass / 2.35482
        transferfunc = np.zeros(np.shape(inputdata), dtype="float64")
        transferfunc[0:halfloc] = np.exp(-((freqaxis[0:halfloc]) ** 2) / (2.0 * sigma * sigma))
        transferfunc[halfloc + 1 :] = np.exp(
            -((freqaxis[halfloc + 1 :] - 1.0 / Fs) ** 2) / (2.0 * sigma * sigma)
        )
    elif type == "trapezoidal":
        if upperstop is None:
            upperstop = upperpass * 1.05
        transferfunc = np.ones(np.shape(inputdata), dtype="float64")
        passbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
        cutoffbin = int((upperstop / Fs) * np.shape(transferfunc)[0])
        transitionlength = cutoffbin - passbin
        if debug:
            print("getlptrapfftfunc - Fs, upperpass, upperstop:", Fs, upperpass, upperstop)
            print(
                "getlptrapfftfunc - passbin, transitionlength, cutoffbin, len(inputdata):",
                passbin,
                transitionlength,
                cutoffbin,
                inputdata.shape,
            )
        if transitionlength > 0:
            transitionvector = np.arange(1.0 * transitionlength) / transitionlength
            transferfunc[passbin:cutoffbin] = 1.0 - transitionvector
            transferfunc[-cutoffbin:-passbin] = transitionvector
        if cutoffbin > 0:
            transferfunc[cutoffbin:-cutoffbin] = 0.0
    elif type == "brickwall":
        transferfunc = np.ones(np.shape(inputdata), dtype=np.float64)
        cutoffbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
        if debug:
            print(
                "getlpfftfunc - Fs, upperpass, len(inputdata):",
                Fs,
                upperpass,
                np.shape(inputdata)[0],
            )
        transferfunc[cutoffbin:-cutoffbin] = 0.0
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("LP Transfer function - " + type)
        plt.plot(freqaxis, transferfunc)
        plt.show()
    return transferfunc


def gethptransfunc(
    Fs: float,
    inputdata: NDArray,
    lowerstop: Optional[float] = None,
    lowerpass: Optional[float] = None,
    type: str = "brickwall",
    debug: bool = False,
) -> NDArray:
    """
    Compute high-pass transfer function from low-pass transfer function.

    This function generates a high-pass transfer function by subtracting a
    low-pass transfer function from unity. The low-pass function is computed
    using the `getlptransfunc` function with appropriate parameters.

    Parameters
    ----------
    Fs : float
        Sampling frequency in Hz.
    inputdata : NDArray
        Input data array used for transfer function computation.
    lowerstop : float, optional
        Lower stop frequency for trapezoidal filter type. Required for
        trapezoidal type, ignored for brickwall type.
    lowerpass : float, optional
        Lower pass frequency (cutoff frequency) for the high-pass filter.
        Must be specified.
    type : str, default="brickwall"
        Type of filter transfer function. Options are "brickwall" or "trapezoidal".
    debug : bool, default=False
        If True, enables debug output during computation.

    Returns
    -------
    NDArray
        High-pass transfer function array with same shape as inputdata.

    Notes
    -----
    For trapezoidal filter type, the lower stop frequency is used as the
    upper pass frequency for the underlying low-pass function.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> data = np.linspace(0, 1, 100)
    >>> hp_func = gethptransfunc(Fs, data, lowerpass=10.0, type="brickwall")
    """
    if lowerpass is None:
        print("gethptransfunc: lowerpass must be specified")
        sys.exit()
    if type == "trapezoidal":
        transferfunc = 1.0 - getlptransfunc(
            Fs,
            inputdata,
            upperpass=lowerstop,
            upperstop=lowerpass,
            type=type,
            debug=debug,
        )
    else:
        transferfunc = 1.0 - getlptransfunc(
            Fs, inputdata, upperpass=lowerpass, type=type, debug=debug
        )
    return transferfunc


# @conditionaljit()
def dolptransfuncfilt(
    Fs: float,
    inputdata: NDArray,
    upperpass: Optional[float] = None,
    upperstop: Optional[float] = None,
    type: str = "brickwall",
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a Gaussian lowpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D array.
    upperpass : float, optional
        Upper end of the passband in Hz. If not specified, the filter will use a default
        value based on the data characteristics.
    upperstop : float, optional
        Upper end of the stopband in Hz. If not specified, the filter will use a default
        value based on the data characteristics.
    type : str, optional
        Type of transfer function to use. Default is "brickwall". Other options may include
        "gaussian", "butterworth", etc., depending on implementation of `getlptransfunc`.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options include "reflect", "wrap", "constant", etc.
        Default is "reflect".
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        A plot of the transfer function will be displayed if debug is enabled.

    Returns
    -------
    NDArray
        The filtered data as a 1D float array of the same length as inputdata.

    Notes
    -----
    This function applies a frequency-domain filter by computing the FFT of the padded input,
    applying a transfer function, and then inverse transforming the result. Padding is applied
    to reduce edge effects caused by the FFT.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> t = np.linspace(0, 1, int(Fs), endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> filtered = dolptransfuncfilt(Fs, signal, upperpass=20.0)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs, padinputdata, upperpass=upperpass, upperstop=upperstop, type=type
    )
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"LP Transfer function - {type}, upperpass={upperpass:.2f}")
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dohptransfuncfilt(
    Fs: float,
    inputdata: NDArray,
    lowerpass: float,
    lowerstop: Optional[float | None] = None,
    type: str = "brickwall",
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a trapezoidal highpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    inputdata : NDArray
        Input data to be filtered.
    lowerpass : float
        Lower end of the passband in Hz.
    lowerstop : float, optional
        Upper end of the stopband in Hz. If not provided, it is set to `lowerpass / 1.05`.
    type : str, optional
        Type of transfer function to use. Default is "brickwall".
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are "reflect" or "wrap". Default is "reflect".
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a highpass filter in the frequency domain using FFT. The input signal
    is padded to reduce edge effects, then filtered using a transfer function, and finally
    unpadded to return the result.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> t = np.linspace(0, 1, int(Fs), endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> filtered = dohptransfuncfilt(Fs, signal, lowerpass=5.0)
    """
    if lowerstop is None:
        lowerstop = lowerpass * (1.0 / 1.05)
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs, padinputdata, upperpass=lowerstop, upperstop=lowerpass, type=type
    )
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"HP Transfer function - {type}, lowerpass={lowerpass:.2f}")
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= 1.0 - transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dobptransfuncfilt(
    Fs: float,
    inputdata: NDArray,
    lowerpass: float,
    upperpass: float,
    lowerstop: Optional[float] = None,
    upperstop: Optional[float] = None,
    type: str = "brickwall",
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a trapezoidal highpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    inputdata : NDArray
        Input data to be filtered.
    lowerpass : float
        Lower end of passband in Hz.
    upperpass : float
        Upper end of passband in Hz.
    lowerstop : float, optional
        Upper end of stopband in Hz. If not provided, it is computed as `lowerpass / 1.05`.
    upperstop : float, optional
        Lower end of stopband in Hz. If not provided, it is computed as `upperpass * 1.05`.
    type : str, optional
        Type of transfer function to use. Default is "brickwall".
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are "reflect" or "wrap". Default is "reflect".
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a bandpass filter in the frequency domain using FFT. It pads the input
    data to minimize edge effects and applies a transfer function that combines a lowpass and
    highpass response. The resulting filtered signal is returned after inverse FFT and unpadding.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> Fs = 100.0
    >>> t = np.linspace(0, 1, int(Fs), endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> filtered = dobptransfuncfilt(Fs, signal, 5.0, 15.0)
    """
    if lowerstop is None:
        lowerstop = lowerpass * (1.0 / 1.05)
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs,
        padinputdata,
        upperpass=upperpass,
        upperstop=upperstop,
        type=type,
        debug=False,
    ) * gethptransfunc(Fs, padinputdata, lowerstop=lowerstop, lowerpass=lowerpass, type=type)
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(
            f"BP Transfer function - {type}, lowerpass={lowerpass:.2f}, upperpass={upperpass:.2f}"
        )
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dolptrapfftfilt(
    Fs: float,
    upperpass: float,
    upperstop: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a trapezoidal lowpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    upperpass : float
        Upper end of the passband in Hz.
    upperstop : float
        Lower end of the stopband in Hz.
    inputdata : NDArray
        Input data to be filtered, as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'wrap'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a trapezoidal lowpass filter in the frequency domain using FFT.
    The input signal is first padded to reduce edge effects, then filtered using a
    transfer function, and finally the padding is removed to return the filtered signal.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> upperpass = 20.0
    >>> upperstop = 25.0
    >>> data = np.random.randn(1000)
    >>> filtered = dolptrapfftfilt(Fs, upperpass, upperstop, data)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dohptrapfftfilt(
    Fs: float,
    lowerstop: float,
    lowerpass: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a trapezoidal highpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerstop : float
        Upper end of stopband in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'cyclic'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a trapezoidal highpass filter in the frequency domain using FFT.
    The input signal is first padded using the specified padding method to reduce edge effects.
    The filter transfer function is constructed using `getlptrapfftfunc`, and the filtered
    signal is obtained by inverse FFT.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> lowerstop = 5.0
    >>> lowerpass = 10.0
    >>> data = np.random.randn(1000)
    >>> filtered = dohptrapfftfilt(Fs, lowerstop, lowerpass, data)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# @conditionaljit()
def dobptrapfftfilt(
    Fs: float,
    lowerstop: float,
    lowerpass: float,
    upperpass: float,
    upperstop: float,
    inputdata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Performs an FFT filter with a trapezoidal bandpass transfer function on an input vector
    and returns the result. Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    lowerstop : float
        Upper end of the lower stopband in Hz.
    lowerpass : float
        Lower end of the lower passband in Hz.
    upperpass : float
        Upper end of the upper passband in Hz.
    upperstop : float
        Lower end of the upper stopband in Hz.
    inputdata : NDArray
        Input data to be filtered, expected as a 1D numpy array.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of the averaging window used in padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'wrap'. Default is 'reflect'.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    This function applies a trapezoidal bandpass filter in the frequency domain using FFT.
    The input signal is first padded to minimize edge effects, then transformed into the
    frequency domain, multiplied by the transfer function, and finally inverse transformed
    back to the time domain.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> data = np.random.randn(1000)
    >>> filtered = dobptrapfftfilt(Fs, 10, 15, 25, 30, data, padlen=50)
    """
    padinputdata = padvec(inputdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    inputdata_trans = fftpack.fft(padinputdata)
    if debug:
        print(
            "Fs=",
            Fs,
            " Fstopl=",
            lowerstop,
            " Fpassl=",
            lowerpass,
            " Fpassu=",
            upperpass,
            " Fstopu=",
            upperstop,
        )
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug) * (
        1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
    )
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.
def wiener_deconvolution(signal: NDArray, kernel: NDArray, lambd: float) -> NDArray:
    """Perform Wiener deconvolution on a signal.

    This function applies Wiener deconvolution to remove blur from a signal using
    the Wiener filter in the frequency domain. The regularization parameter `lambd`
    represents the signal-to-noise ratio in the Fourier domain.

    Parameters
    ----------
    signal : NDArray
        Input signal to be deconvolved, 1D array.
    kernel : NDArray
        Convolution kernel (point spread function), 1D array.
    lambd : float
        Regularization parameter representing the signal-to-noise ratio in
        the Fourier domain. Higher values correspond to more smoothing.

    Returns
    -------
    NDArray
        Deconvolved signal, same length as input signal.

    Notes
    -----
    The Wiener deconvolution formula in frequency domain is:
    Y = X * H* / (|H|² + λ²)

    where X is the Fourier transform of the input signal, H is the Fourier
    transform of the kernel, and H* is the complex conjugate of H.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 3, 2, 1])
    >>> kernel = np.array([1, 0.5, 0.25])
    >>> result = wiener_deconvolution(signal, kernel, lambd=0.1)
    """
    kernel = np.hstack(
        (kernel, np.zeros(len(signal) - len(kernel)))
    )  # zero pad the kernel to same length
    H = fftpack.fft(kernel)
    deconvolved = np.roll(
        np.real(fftpack.ifft(fftpack.fft(signal) * np.conj(H) / (H * np.conj(H) + lambd**2))),
        int(len(signal) // 2),
    )
    return deconvolved


def pspec(inputdata: NDArray) -> NDArray:
    """
    Calculate the power spectrum of an input signal.

    Parameters
    ----------
    inputdata : NDArray
        Input signal data array of shape (n,) where n is the number of samples.

    Returns
    -------
    NDArray
        The power spectrum of the input signal as a 1D numpy array of shape (n,).
        Each element represents the power at the corresponding frequency bin.

    Notes
    -----
    This function computes the power spectrum using the Fast Fourier Transform (FFT).
    The power spectrum is calculated as the square root of the product of the FFT
    and its complex conjugate, which gives the magnitude of the frequency components.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    >>> spectrum = pspec(signal)
    >>> print(spectrum.shape)
    (100,)
    """
    S = fftpack.fft(inputdata)
    return np.sqrt(S * np.conj(S))


def spectralflatness(spectrum: NDArray) -> float:
    return np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)


def spectrum(
    inputdata: NDArray, Fs: float = 1.0, mode: str = "power", trim: bool = True
) -> Tuple[NDArray, Union[NDArray, None]]:
    """
    Compute the spectral flatness of a spectrum.

    Spectral flatness is a measure of how much a spectrum resembles white noise.
    It is defined as the ratio of the geometric mean to the arithmetic mean of the spectrum.

    Parameters
    ----------
    spectrum : NDArray
        Input spectrum array. Should contain non-negative values.

    Returns
    -------
    float
        Spectral flatness value. Values close to 1.0 indicate a flat (white noise-like) spectrum,
        while values closer to 0.0 indicate a more tonal spectrum.

    Notes
    -----
    The spectral flatness is computed as:
    flatness = exp(mean(log(spectrum))) / mean(spectrum)

    This implementation assumes the input spectrum contains non-negative values.
    If the spectrum contains zeros, the geometric mean will be zero and the result
    will be zero regardless of the arithmetic mean.

    Examples
    --------
    >>> import numpy as np
    >>> # White noise spectrum
    >>> white_noise = np.random.rand(100)
    >>> flatness = spectralflatness(white_noise)
    >>> # Tone-like spectrum
    >>> tone = np.zeros(100)
    >>> tone[50] = 1.0
    >>> flatness = spectralflatness(tone)
    """
    if trim:
        specvals = fftpack.fft(inputdata)[0 : len(inputdata) // 2]
        maxfreq = Fs / 2.0
        specaxis = np.linspace(0.0, maxfreq, len(specvals), endpoint=False)
    else:
        specvals = fftpack.fft(inputdata)
        maxfreq = Fs
        specaxis = np.linspace(0.0, maxfreq, len(specvals), endpoint=False)
    if mode == "real":
        specvals = (specvals.real).astype(specvals.dtype)
    elif mode == "imag":
        specvals = (specvals.imag).astype(specvals.dtype)
    elif mode == "complex":
        pass
    elif mode == "mag":
        specvals = np.absolute(specvals)
    elif mode == "phase":
        specvals = (np.angle(specvals)).astype(specvals.dtype)
    elif mode == "power":
        specvals = np.sqrt(np.absolute(specvals))
    else:
        raise RuntimeError("illegal spectrum mode")
    return specaxis, specvals


def setnotchfilter(thefilter: NoncausalFilter, thefreq: float, notchwidth: float = 1.0) -> None:
    """
    Set notch filter parameters for the specified filter.

    This function configures a notch filter by setting the filter type to "arb_stop"
    and defining the frequency range for the notch based on the center frequency
    and notch width parameters.

    Parameters
    ----------
    thefilter : NoncausalFilter function
        The filter function to configure with notch filter parameters
    thefreq : float
        Center frequency of the notch in Hz
    notchwidth : float, optional
        Width of the notch in Hz, default is 1.0 Hz

    Returns
    -------
    None
        This function modifies the filter in-place and does not return any value

    Notes
    -----
    The notch filter is configured as an "arb_stop" type filter with symmetric
    frequency bounds around the center frequency. The actual notch range will be
    from (thefreq - notchwidth/2) to (thefreq + notchwidth/2) Hz.

    Examples
    --------
    >>> filter_obj = NoncausalFilter()
    >>> setnotchfilter(filter_obj, 50.0, 2.0)
    >>> # Creates a notch filter centered at 50 Hz with 2 Hz width
    """
    thefilter.settype("arb_stop")
    thefilter.setfreqs(
        thefreq - notchwidth / 2.0,
        thefreq - notchwidth / 2.0,
        thefreq + notchwidth / 2.0,
        thefreq + notchwidth / 2.0,
    )


def harmonicnotchfilter(
    timecourse: NDArray,
    Fs: float,
    Ffundamental: float,
    notchpct: float = 1.0,
    debug: bool = False,
) -> NDArray:
    """
    Apply a harmonic notch filter to remove a fundamental frequency and its harmonics from a timecourse.

    This function removes the specified fundamental frequency and all its integer harmonics
    using a non-causal notch filtering approach. The width of each notch is proportional
    to the harmonic order and the specified percentage (`notchpct`).

    Parameters
    ----------
    timecourse : NDArray
        Input timecourse data to be filtered.
    Fs : float
        Sampling rate of the input data in Hz.
    Ffundamental : float
        Fundamental frequency to be removed from the data in Hz.
    notchpct : float, optional
        Width of the notch relative to the filter frequency in percent. Default is 1.0.
    debug : bool, optional
        If True, prints detailed information about the filtering process. Default is False.

    Returns
    -------
    filteredtc : NDArray
        The filtered timecourse with the fundamental and its harmonics removed.

    Notes
    -----
    - The function uses a non-causal filter, meaning it requires the full signal to be available.
    - Harmonics are calculated as integer multiples of the fundamental frequency.
    - The notch width is determined by `notchpct * harmonic * Ffundamental * 0.01`, with a minimum
      width of one frequency bin.

    Examples
    --------
    >>> import numpy as np
    >>> timecourse = np.random.randn(1000)
    >>> Fs = 100.0
    >>> Ffundamental = 50.0
    >>> filtered = harmonicnotchfilter(timecourse, Fs, Ffundamental, notchpct=2.0)
    """
    # delete the fundamental and its harmonics
    filteredtc = timecourse + 0.0
    maxpass = Fs / 2.0
    if notchpct is not None:
        stopfreq = Ffundamental
        freqstep = 0.5 * Fs / len(filteredtc)
        maxharmonic = int(maxpass // stopfreq)
        if debug:
            print("highest harmonic is", maxharmonic, "(", maxharmonic * stopfreq, "Hz)")
        thenotchfilter = NoncausalFilter()
        for harmonic in range(1, maxharmonic + 1):
            notchfreq = harmonic * stopfreq
            if debug:
                print("removing harmonic at", notchfreq)
            notchwidth = np.max([notchpct * harmonic * stopfreq * 0.01, freqstep])
            if debug:
                print("\tFs:", Fs)
                print("\tstopfreq:", stopfreq)
                print("\tnotchpct:", notchpct)
                print("\tnotchwidth:", notchwidth)
                print("\tnotchfreq:", notchfreq)
                print("\tfreqstep:", freqstep)
                print("\tminfreqstep:", freqstep / notchfreq)
                print("\tbins:", int(notchwidth // freqstep))
                print()
            setnotchfilter(thenotchfilter, notchfreq, notchwidth=notchwidth)
            filteredtc = thenotchfilter.apply(Fs, filteredtc)
    return filteredtc


def savgolsmooth(data: NDArray, smoothlen: int = 101, polyorder: int = 3) -> NDArray:
    """
    Apply Savitzky-Golay filter to smooth data.

    This function applies a Savitzky-Golay filter to smooth the input data. The filter uses
    a least-squares method to fit a polynomial to a sliding window of data points and
    replaces each point with the value of the polynomial at that point.

    Parameters
    ----------
    data : array_like
        Input data to be smoothed. Can be any array-like object that can be converted
        to a numpy array.
    smoothlen : int, optional
        The length of the filter window (i.e., the number of coefficients). Must be a
        positive odd integer. Default is 101.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less than
        `smoothlen`. Default is 3.

    Returns
    -------
    NDArray
        The smoothed data with the same shape as the input `data`.

    Notes
    -----
    The Savitzky-Golay filter is particularly useful for smoothing noisy data while
    preserving the shape and features of the underlying signal. It is especially
    effective for data with a lot of high-frequency noise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_filter
    >>> x = np.linspace(0, 4*np.pi, 100)
    >>> y = np.sin(x) + np.random.normal(0, 0.1, 100)
    >>> y_smooth = savgolsmooth(y, smoothlen=21, polyorder=3)
    """
    return savgol_filter(data, smoothlen, polyorder)


def csdfilter(
    obsdata: NDArray,
    commondata: NDArray,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Cross spectral density filter - makes a filter transfer function that preserves common frequencies.

    This function applies a filter based on the cross spectral density between two signals.
    It uses the Fourier transform to compute the transfer function and applies it to the
    observation data, preserving the frequency components that are common between the two
    input signals.

    Parameters
    ----------
    obsdata : NDArray
        Input data (1D numpy array) to be filtered.
    commondata : NDArray
        Shared data (1D numpy array) used to compute the transfer function.
    padlen : int, optional
        Number of reflected points to add on each end of the input data. Default is 20.
    avlen : int, optional
        Length of the averaging window for padding. Default is 20.
    padtype : str, optional
        Type of padding to use. Options are 'reflect' or 'cyclic'. Default is 'reflect'.
    debug : bool, optional
        Set to True for additional information on function internals. Default is False.

    Returns
    -------
    NDArray
        The filtered data (1D numpy array) with preserved common frequencies.

    Notes
    -----
    The function first pads both input arrays using the specified padding method, then computes
    the FFT of both padded arrays. A transfer function is constructed from the square root of
    the magnitude of the product of the FFTs. This transfer function is applied to the FFT of
    the observation data, and the inverse FFT is taken to return the filtered signal.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.randn(100)
    >>> common = np.random.randn(100)
    >>> filtered = csdfilter(obs, common, padlen=10)
    """
    padobsdata = padvec(obsdata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    padcommondata = padvec(commondata, padlen=padlen, avlen=avlen, padtype=padtype, debug=debug)
    obsdata_trans = fftpack.fft(padobsdata)
    transferfunc = np.sqrt(np.abs(fftpack.fft(padobsdata) * np.conj(fftpack.fft(padcommondata))))
    obsdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(obsdata_trans).real, padlen=padlen)

# @conditionaljit()
def arb_pass(
    Fs: float,
    inputdata: NDArray,
    lowerstop: float,
    lowerpass: float,
    upperpass: float,
    upperstop: float,
    transferfunc: str = "trapezoidal",
    butterorder: int = 6,
    padlen: int = 20,
    avlen: int = 20,
    padtype: str = "reflect",
    debug: bool = False,
) -> NDArray:
    """
    Filters an input waveform over a specified range using configurable filter types.

    By default, it applies a trapezoidal FFT filter, but brickwall and Butterworth
    filters are also supported. The function handles lowpass, bandpass, and highpass
    filtering based on the input frequency limits. Ends of the input data are padded
    to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz.
    inputdata : NDArray
        Input data to be filtered.
    lowerstop : float
        Upper end of lower stopband in Hz.
    lowerpass : float
        Lower end of passband in Hz.
    upperpass : float
        Upper end of passband in Hz.
    upperstop : float
        Lower end of upper stopband in Hz.
    transferfunc : str, optional
        Type of transfer function to use. Options are:
        - "trapezoidal" (default)
        - "brickwall"
        - "butterworth"
    butterorder : int, optional
        Order of Butterworth filter, if used. Default is 6.
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.
        Default is 20.
    avlen : int, optional
        Length of averaging window for filtering. Default is 20.
    padtype : str, optional
        Padding type for end effects. Options are:
        - "reflect" (default)
        - "wrap"
    debug : bool, optional
        If True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    filtereddata : NDArray
        The filtered data as a 1D float array.

    Notes
    -----
    The function automatically determines whether to apply a lowpass, highpass, or
    bandpass filter based on the values of `lowerpass` and `upperpass`. For bandpass
    filters, a cascade of lowpass and highpass Butterworth filters is used when
    `transferfunc="butterworth"`.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> t = np.linspace(0, 1, int(Fs), endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    >>> filtered = arb_pass(
    ...     Fs=Fs,
    ...     inputdata=signal,
    ...     lowerstop=5.0,
    ...     lowerpass=8.0,
    ...     upperpass=15.0,
    ...     upperstop=20.0,
    ...     transferfunc="trapezoidal"
    ... )
    """
    # adjust the padding for speed
    if pyfftwpresent:
        thefftlen = optfftlen(len(inputdata), padlen=padlen)
        padlen = int((thefftlen - len(inputdata)) // 2)

    # check filter limits to see if we should do a lowpass, bandpass, or highpass
    if lowerpass <= 0.0:
        # set up for lowpass
        if transferfunc == "butterworth":
            retvec = dolpfiltfilt(
                Fs,
                upperpass,
                inputdata,
                butterorder,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )
            return retvec
        else:
            return dolptransfuncfilt(
                Fs,
                inputdata,
                upperpass=upperpass,
                upperstop=upperstop,
                type=transferfunc,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )
    elif (upperpass >= Fs / 2.0) or (upperpass <= 0.0):
        # set up for highpass
        if transferfunc == "butterworth":
            return dohpfiltfilt(
                Fs,
                lowerpass,
                inputdata,
                butterorder,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )
        else:
            return dohptransfuncfilt(
                Fs,
                inputdata,
                lowerpass,
                lowerstop=lowerstop,
                type=transferfunc,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )
    else:
        # set up for bandpass
        if transferfunc == "butterworth":
            return dohpfiltfilt(
                Fs,
                lowerpass,
                dolpfiltfilt(
                    Fs,
                    upperpass,
                    inputdata,
                    butterorder,
                    padlen=padlen,
                    avlen=avlen,
                    padtype=padtype,
                    debug=debug,
                ),
                butterorder,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )
        else:
            return dobptransfuncfilt(
                Fs,
                inputdata,
                lowerpass,
                upperpass,
                lowerstop=lowerstop,
                upperstop=upperstop,
                type=transferfunc,
                padlen=padlen,
                avlen=avlen,
                padtype=padtype,
                debug=debug,
            )


class Plethfilter:
    def __init_(self, Fs, Fl, Fh, order=4, attenuation=20):
        """
        Initialize Chebyshev type II bandpass filter.

        Parameters
        ----------
        Fs : float
            Sampling frequency in Hz.
        Fl : float
            Lower cutoff frequency in Hz.
        Fh : float
            Higher cutoff frequency in Hz.
        order : int, optional
            Filter order (default is 4).
        attenuation : float, optional
            Stopband attenuation in dB (default is 20).

        Returns
        -------
        None
            Initializes filter coefficients and parameters.

        Notes
        -----
        This function creates a Chebyshev type II bandpass filter using scipy.signal.cheby2.
        The filter has a flat response in the passband and an equiripple response in the stopband.
        The filter coefficients are stored in self.b and self.a for subsequent filtering operations.

        Examples
        --------
        >>> filter = BandpassFilter(Fs=1000, Fl=100, Fh=300, order=6, attenuation=30)
        >>> print(filter.b)  # Print filter numerator coefficients
        >>> print(filter.a)  # Print filter denominator coefficients
        """
        self.Fs = Fs
        self.Fh = Fh
        self.Fl = Fl
        self.attenuation = attenuation
        self.order = order
        retvec = signal.cheby2(
            self.order,
            self.attenuation,
            [self.Fl / self.Fn, self.Fh / self.Fn],
            btype="bandpass",
            analog=False,
            output="ba",
        )
        self.b = retvec[0]
        self.a = retvec[1]

    def apply(self, data):
        return signal.filtfilt(self.b, self.a, data, axis=-1, padtype="odd", padlen=None)


def getfilterbandfreqs(
    band: str, transitionfrac: float = 0.05, species: str = "human", asrange: bool = False
) -> Union[str, Tuple[float, float, float, float]]:
    """
    Apply digital filter forward and backward to data.

    This function applies a digital filter to the input data using forward-backward filtering
    to eliminate phase distortion. The filter is applied along the last axis of the data.

    Parameters
    ----------
    data : array_like
        Input data to be filtered. Can be any shape, but filtering is applied along the
        last axis (axis=-1).

    Returns
    -------
    NDArray
        The filtered output with the same shape as the input data.

    Notes
    -----
    This function uses `scipy.signal.filtfilt` which applies the filter forward and backward
    to eliminate phase distortion. The filter coefficients are stored in `self.b` and `self.a`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> # Create a simple filter
    >>> b, a = signal.butter(4, 0.2, 'low')
    >>> # Apply filter to data
    >>> filtered_data = apply(data)
    """
    if species == "human":
        if band == "vlf":
            lowerpass = 0.0
            upperpass = 0.009
            lowerstop = 0.0
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "lfo":
            lowerpass = 0.01
            upperpass = 0.15
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "lfo_legacy":
            lowerpass = 0.01
            upperpass = 0.15
            lowerstop = 0.009
            upperstop = 0.2
        elif band == "lfo_tight":
            lowerpass = 0.01
            upperpass = 0.10
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "resp":
            lowerpass = 0.2
            upperpass = 0.5
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "cardiac":
            lowerpass = 0.66
            upperpass = 3.0
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "hrv_ulf":
            lowerpass = 0.0
            upperpass = 0.0033
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "hrv_vlf":
            lowerpass = 0.0033
            upperpass = 0.04
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "hrv_lf":
            lowerpass = 0.04
            upperpass = 0.15
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "hrv_hf":
            lowerpass = 0.15
            upperpass = 0.4
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        elif band == "hrv_vhf":
            lowerpass = 0.4
            upperpass = 0.5
            lowerstop = lowerpass * (1.0 - transitionfrac)
            upperstop = upperpass * (1.0 + transitionfrac)
        else:
            print(f"unknown filter band: {band}")
            sys.exit()
    else:
        print(f"unknown species: {species}")
        sys.exit()
    if asrange:
        return f"{lowerpass}-{upperpass}Hz"
    else:
        return lowerpass, upperpass, lowerstop, upperstop




# --------------------------- FFT helper functions ---------------------------------------------
def polarfft(inputdata: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Compute the polar representation of the FFT of input data.

    This function computes the Fast Fourier Transform of the input data and returns
    its magnitude and phase angle in polar coordinates.

    Parameters
    ----------
    inputdata : array_like
        Input data to transform. Can be real or complex valued.

    Returns
    -------
    tuple of NDArray
        A tuple containing:
        - magnitude : NDArray
            The magnitude (absolute value) of the FFT result
        - phase : NDArray
            The phase angle (in radians) of the FFT result

    Notes
    -----
    This function uses `scipy.fftpack.fft` for the FFT computation and returns
    the polar representation of the complex FFT result. The magnitude represents
    the amplitude spectrum while the phase represents the phase spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> x = np.array([1, 2, 3, 4])
    >>> magnitude, phase = polarfft(x)
    >>> print("Magnitude:", magnitude)
    >>> print("Phase:", phase)
    """
    complexxform = fftpack.fft(inputdata)
    return np.abs(complexxform), np.angle(complexxform)


def ifftfrompolar(r: NDArray, theta: NDArray) -> NDArray:
    """
    Compute inverse Fourier transform from polar representation.

    This function converts magnitude and phase data to complex form and
    computes the inverse Fourier transform, returning only the real part.

    Parameters
    ----------
    r : array_like
        Magnitude values of the polar representation.
    theta : array_like
        Phase values (in radians) of the polar representation.

    Returns
    -------
    NDArray
        Real part of the inverse Fourier transform of the complex signal.

    Notes
    -----
    The function assumes that the input arrays `r` and `theta` have the same shape
    and represent the magnitude and phase of a complex signal in polar form.
    The result is the inverse Fourier transform of the complex signal r * exp(1j * theta).

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([1.0, 0.5, 0.2])
    >>> theta = np.array([0.0, np.pi/4, np.pi/2])
    >>> result = ifftfrompolar(r, theta)
    >>> print(result)
    [ 0.54123456 -0.12345678  0.23456789]
    """
    complexxform = r * np.exp(1j * theta)
    return fftpack.ifft(complexxform).real


# --------------------------- Window functions -------------------------------------------------
BHwindows: dict = {}


def blackmanharris(length: int, debug: bool = False) -> NDArray:
    """
    Returns a Blackman-Harris window function of the specified length.

    The Blackman-Harris window is a tapering function used in signal processing
    to reduce spectral leakage. It is defined as a weighted sum of cosine terms
    with specific coefficients that minimize the sidelobe level.

    Parameters
    ----------
    length : int
        The length of the window function.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    windowfunc : NDArray
        The Blackman-Harris window function of the specified length, as a 1D float array.

    Notes
    -----
    This function uses a caching mechanism to store previously computed window functions
    for improved performance when the same window length is requested multiple times.

    The window is defined by the following formula:
    w(n) = a0 - a1*cos(2πn/M) + a2*cos(4πn/M) - a3*cos(6πn/M)

    where M = length - 1, and the coefficients are:
    a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168

    Examples
    --------
    >>> from numpy import array
    >>> window = blackmanharris(8)
    >>> print(window)
    [0.00000000e+00 1.15530000e-02 1.00000000e+00 1.00000000e+00
     1.00000000e+00 1.00000000e+00 1.15530000e-02 0.00000000e+00]
    """
    # return a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
    try:
        return BHwindows[str(length)]
    except KeyError:
        argvec = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(length))
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        BHwindows[str(length)] = (
            a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
        )
        if debug:
            print("initialized Blackman-Harris window for length", length)
        return BHwindows[str(length)]


hannwindows: dict = {}


def hann(length: int, debug: bool = False) -> NDArray:
    """
    Returns a Hann window function of the specified length.

    Once calculated, windows are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    windowfunc : NDArray
        The Hann window function as a 1D float array of the specified length.

    Notes
    -----
    The Hann window is defined as:
    w(n) = 0.5 * (1 - cos(2πn/(N-1)))

    where N is the window length and n ranges from 0 to N-1.

    This implementation uses a cached approach for improved performance when
    the same window lengths are requested multiple times.

    Examples
    --------
    >>> from numpy import array
    >>> hann(5)
    array([0.        , 0.25      , 0.75      , 0.25      , 0.        ])

    >>> hann(4, debug=True)
    initialized hann window for length 4
    array([0.        , 0.5       , 1.        , 0.5       ])
    """
    # return 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
    try:
        return hannwindows[str(length)]
    except KeyError:
        hannwindows[str(length)] = 0.5 * (
            1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi)
        )
        if debug:
            print("initialized hann window for length", length)
        return hannwindows[str(length)]


hammingwindows: dict = {}


def rect(length: int, L: float) -> NDArray:
    """
    Generate a rectangular window function.

    This function creates a rectangular window of specified length and width,
    where the window has a value of 1 within the specified width and 0 outside.

    Parameters
    ----------
    length : int
        The length of the output array.
    L : float
        The width of the rectangular window (in samples).

    Returns
    -------
    NDArray
        A numpy array of shape (length,) containing the rectangular window
        function values, where values are 1.0 within the window and 0.0 outside.

    Notes
    -----
    The rectangular window is centered at the middle of the array. The window
    extends from -L/2 to +L/2 relative to the center. Values outside this range
    are set to zero.

    Examples
    --------
    >>> rect(5, 3)
    array([0., 1., 1., 1., 0.])

    >>> rect(6, 4)
    array([0., 1., 1., 1., 1., 0.])
    """
    thearray = np.abs(np.linspace(0, length, length, endpoint=False) - length / 2.0)
    return np.where(thearray <= L / 2.0, 1.0, 0.0)


def mRect(
    length: int,
    alpha: float = 0.5,
    omegac: Optional[float] = None,
    phi: float = 0.0,
    debug: bool = False,
) -> NDArray:
    """
    Generate a modified rectangular window function.

    This function creates a window by combining a base rectangular function with
    a scaled second rectangular function modulated by a cosine term. The resulting
    window is normalized to have a maximum value of 1.

    Parameters
    ----------
    length : int
        Length of the window array to be generated.
    alpha : float, optional
        Scaling factor for the second rectangular function, by default 0.5
    omegac : float, optional
        Cutoff frequency parameter. If None, defaults to 2.0/length, by default None
    phi : float, optional
        Phase shift in radians for the cosine term, by default 0.0
    debug : bool, optional
        If True, plots the individual components of the window, by default False

    Returns
    -------
    NDArray
        Normalized window array of shape (length,) with maximum value of 1

    Notes
    -----
    The window is constructed as:
    w(n) = [rect(length, 1/omegac) + alpha * rect(length, 2/omegac) * cos(π * omegac * n + phi)] / max(w)

    Examples
    --------
    >>> window = mRect(100)
    >>> window = mRect(100, alpha=0.3, omegac=0.1, phi=np.pi/4)
    """
    if omegac is None:
        omegac = 2.0 / length
    L = 1.0 / omegac
    indices = np.linspace(0, length, length, endpoint=False) - length / 2.0
    firstrect = rect(length, L)
    secondrect = alpha * rect(length, L * 2.0)
    costerm = np.cos(np.pi * omegac * indices + phi)
    thewindow = firstrect + secondrect * costerm
    if debug:
        plt.plot(firstrect)
        plt.plot(1.0 + secondrect * costerm)
        plt.show()
    return thewindow / np.max(thewindow)


def hamming(length: int, debug: bool = False) -> NDArray:
    #   return 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
    """
    Returns a Hamming window function of the specified length.

    Once calculated, windows are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    windowfunc : 1D float array
        The Hamming window function of the specified length

    Notes
    -----
    The Hamming window is defined as:
    w(n) = 0.54 - 0.46 * cos(2 * π * n / (N-1))

    where N is the window length and n ranges from 0 to N-1.

    Examples
    --------
    >>> from numpy import array
    >>> hamming(4)
    array([0.08, 1.0 , 1.0 , 0.08])

    >>> hamming(5, debug=True)
    initialized hamming window for length 5
    array([0.08, 1.0 , 1.0 , 1.0 , 0.08])
    """
    try:
        return hammingwindows[str(length)]
    except KeyError:
        hammingwindows[str(length)] = 0.54 - 0.46 * np.cos(
            (np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi
        )
        if debug:
            print("initialized hamming window for length", length)
        return hammingwindows[str(length)]


def windowfunction(length: int, type: str = "hamming", debug: bool = False) -> NDArray:
    """
    Returns a window function of the specified length and type. Once calculated, windows
    are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function.
    type : {'hamming', 'hann', 'blackmanharris', 'None'}, optional
        Window type. Choices are 'hamming' (default), 'hann', 'blackmanharris', and 'None'.
        If 'None' is specified, a window of ones is returned.
    debug : bool, optional
        When True, internal states of the function will be printed to help debugging.
        Default is False.

    Returns
    -------
    windowfunc : NDArray
        The window function as a 1D float array of the specified length.

    Notes
    -----
    This function serves as a wrapper for different window functions and includes
    caching mechanism for improved performance. The supported window types are:

    - 'hamming': Hamming window
    - 'hann': Hann (Hanning) window
    - 'blackmanharris': Blackman-Harris window
    - 'None': Rectangular window (all ones)

    Examples
    --------
    >>> windowfunction(10, 'hamming')
    array([0.08        , 0.15302333, 0.41302333, 0.77102333, 0.99902333,
           0.99902333, 0.77102333, 0.41302333, 0.15302333, 0.08        ])

    >>> windowfunction(5, 'hann')
    array([0.        , 0.5       , 1.        , 0.5       , 0.        ])

    >>> windowfunction(4, 'None')
    array([1., 1., 1., 1.])
    """
    if type == "hamming":
        return hamming(length, debug=debug)
    elif type == "hann":
        return hann(length, debug=debug)
    elif type == "blackmanharris":
        return blackmanharris(length, debug=debug)
    elif type == "None":
        return np.ones(length)
    else:
        print("illegal window function")
        sys.exit()
