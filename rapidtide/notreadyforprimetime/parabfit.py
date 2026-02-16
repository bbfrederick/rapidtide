def parabfit(x_axis: NDArray, y_axis: NDArray, peakloc: int, points: int) -> Tuple[float, float]:
    """
    Fit a parabola to a localized region around a peak and return the peak coordinates.

    This function performs a quadratic curve fitting on a subset of data surrounding
    a specified peak location. It uses a parabolic model of the form a*(x-tau)^2 + c
    to estimate the precise peak position and amplitude.

    Parameters
    ----------
    x_axis : NDArray
        Array of x-axis values (typically time or frequency).
    y_axis : NDArray
        Array of y-axis values (typically signal amplitude).
    peakloc : int
        Index location of the peak in the data arrays.
    points : int
        Number of points to include in the local fit around the peak.

    Returns
    -------
    Tuple[float, float]
        A tuple containing (x_peak, y_peak) - the fitted peak coordinates.

    Notes
    -----
    The function uses a least-squares fitting approach with scipy.optimize.curve_fit.
    Initial parameter estimates are derived analytically based on the peak location
    and a distance calculation. The parabolic model assumes the peak has a symmetric
    quadratic shape.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2 * (x - 5)**2 + 1
    >>> peak_x, peak_y = parabfit(x, y, 50, 10)
    >>> print(f"Peak at x={peak_x:.2f}, y={peak_y:.2f}")
    """
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    distance = abs(x_axis[peakloc[1][0]] - x_axis[peakloc[0][0]]) / 4
    index = peakloc
    x_data = x_axis[index - points // 2 : index + points // 2 + 1]
    y_data = y_axis[index - points // 2 : index + points // 2 + 1]

    # get a first approximation of tau (peak position in time)
    tau = x_axis[index]

    # get a first approximation of peak amplitude
    c = y_axis[index]
    a = np.sign(c) * (-1) * (np.sqrt(abs(c)) / distance) ** 2
    """Derived from ABC formula to result in a solution where A=(rot(c)/t)**2"""

    # build list of approximations
    p0 = (a, tau, c)
    popt, pcov = curve_fit(func, x_data, y_data, p0)

    # retrieve tau and c i.e x and y value of peak
    x, y = popt[1:3]
    return x, y



