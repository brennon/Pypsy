import numpy as np
import scipy.signal
from .conversion import *

def kaiser_method(delta_pass, delta_stop, omega_pass, omega_stop):
    """
    Use the Kaiser method to approximate the required order of an FIR filter. The order is given by

    .. math::

        N \\approx \\frac{-(13 + 20 \\log_{10} \\sqrt{\\delta_p\\delta_s})}{\\frac{14.6}{2\\pi}(\\omega_s - \\omega_p)}

    where :math:`N` is the filter order, :math:`\\delta_p` is the passband ripple magnitude, :math:`\\delta_s` is the
    stopband ripple magnitude, :math:`\\omega_s` is the stopband frequency, and :math:`\\omega_s` is the passband
    frequency.

    Parameters
    ----------
    delta_pass : float
        The passband ripple magnitude
    delta_stop : float
        The stopband ripple magnitude
    omega_pass : float
        The normalized cutoff frequency (:math:`\\pi \\frac{radians}{second}`)
    omega_stop : float
        The normalized stopband frequency (:math:`\\pi \\frac{radians}{second}`)

    Returns
    -------
    float
        The approximated filter order

    Examples
    --------
    >>> kaiser_method(0.108749, 0.001, 0.04, 0.08)
    288
    """

    numerator = -(20. * np.log10(np.sqrt(delta_pass * delta_stop)) + 13.)
    denominator = (14.6 / (2. * np.pi)) * (omega_stop - omega_pass)
    return np.int(np.ceil(numerator / denominator) + 1)


def kaiser_window_length(f_cutoff, f_stop, f_sample, ripple):
    """
    Calculate the length a a Kaiser window for use in an FIR filter.

    Parameters
    ----------
    f_cutoff : float
        The cutoff frequency (Hertz)
    f_stop : float
        The stopband frequency (Hertz)
    f_sample : float
        The sampling frequency (Hertz)
    ripple : float
        The minimum stopband ripple (dB)

    Returns
    -------
    float
        The calculated window length

    Raises
    ------
    ValueError
        If ``ripple`` is less than ``8``

    Examples
    --------
    >>> kaiser_window_length(1., 2., 50., 9.)
    5
    """

    # Normalize frequencies
    f_cutoff_normalize = hertz_to_normalized(f_cutoff, f_sample)
    f_stop_normalize = hertz_to_normalized(f_stop, f_sample)

    length, _ = scipy.signal.kaiserord(ripple, np.abs(f_cutoff_normalize - f_stop_normalize))

    return length


def estimate_filter_order(f_cutoff, f_stop, f_sample, passband_ripple_db, stopband_minimum_attenuation_db):
    """
    Approximate the order of an FIR filter using Kaiser's method.

    Parameters
    ----------
    f_cutoff : float
        The cutoff frequency (Hertz)
    f_stop : float
        The stopband frequency (Hertz)
    f_sample : float
        The sampling frequency (Hertz)
    passband_ripple_db : float
        The maximum passband ripple (dB)
    stopband_attenuation_db :
        The minimum stopband attenuation (dB)

    Returns
    -------
    out : int
        The approximated filter order

    Examples
    --------
    >>> f_cutoff = 1.
    >>> f_stop = 2.
    >>> f_sample = 50.
    >>> passband_ripple_db = 1.
    >>> stopband_minimum_attenuation_db = -60.
    >>> ripple = 9.
    >>> estimate_filter_order(f_cutoff, f_stop, f_sample, passband_ripple_db, stopband_minimum_attenuation_db)
    288
    """

    # Normalize frequencies
    omega_pass = (2. * f_cutoff) / f_sample
    omega_stop = (2. * f_stop) / f_sample

    # Convert decibel values to amplitudes
    delta_pass = 1. - db_to_amplitude(-np.abs(passband_ripple_db))
    delta_stop = db_to_amplitude(-np.abs(stopband_minimum_attenuation_db))

    # Use Kaiser method to determine filter length
    return kaiser_method(delta_pass, delta_stop, omega_pass, omega_stop)

def lowpass_filter(f_cutoff, f_stop, f_sample, passband_ripple_db=1., stopband_minimum_attenuation_db=60.):
    """
    Construct the coefficients for a FIR lowpass filter using the window method with a Kaiser window.

    Parameters
    ----------
    f_cutoff : float
        The cutoff frequency (Hertz)
    f_stop : float
        The stopband frequency (Hertz)
    f_sample : float
        The sampling frequency (Hertz)
    passband_ripple_db : float
        The maximum passband ripple (dB)
    stopband_minimum_attenuation_db : float
        The minimum stopband attenuation (dB)

    Returns
    -------
    :py:class:`numpy.ndarray`
        The feedforward coefficients of the filter

    Examples
    --------
    >>> expected_coefficients = np.array([-0.00871449, -0.01119944, 0.01417487, 0.0177774, -0.0222173, -0.02783627, \
            0.0352281, 0.04552014, -0.06113717, -0.08839823, 0.15050017, 0.45630222, 0.45630222, 0.15050017, \
            -0.08839823, -0.06113717, 0.04552014, 0.0352281, -0.02783627, -0.0222173, 0.0177774, 0.01417487, \
            -0.01119944, -0.00871449])
    >>> actual_coefficients = lowpass_filter(11025., 22050., 44100.)
    >>> np.sum(np.abs(expected_coefficients - actual_coefficients)) < 0.0001
    True
    """

    ripple = 9.
    numtaps = estimate_filter_order(f_cutoff, f_stop, f_sample, passband_ripple_db, stopband_minimum_attenuation_db)
    window_length = kaiser_window_length(f_cutoff, f_stop, f_sample, ripple)
    f_cutoff_normalized = hertz_to_normalized(f_cutoff, f_sample)
    f_stop_normalized = hertz_to_normalized(f_stop, f_sample)

    b = scipy.signal.firwin(numtaps, f_cutoff_normalized, window=('kaiser', window_length))
    return b

