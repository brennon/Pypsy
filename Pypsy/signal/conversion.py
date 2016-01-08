import numpy as np


def hertz_to_normalized(f, f_sample):
    """
    Convert a frequency in Hertz to normalized frequency in
    :math:`\\pi \\frac{\\text{radians}}{\\text{sample}}`

    Parameters
    ----------
    f : float or :py:class:`numpy.ndarray`
        Frequency to convert (Hertz)
    f_sample : float
        Sample rate (Hertz)

    Returns
    -------
    float or :py:class:`numpy.ndarray`
        The normalized frequency

    Examples
    --------
    The Nyquist rate is exactly :math:`\\pi \\frac{\\text{radians}}{\\text{sample}}`

    >>> hertz_to_normalized(100., 200.)
    1.0

    The sample rate is exactly :math:`2\\pi \\frac{\\text{radians}}{\\text{sample}}`

    >>> hertz_to_normalized(44100., 44100.)
    2.0

    >>> hertz_to_normalized(11025., 44100.)
    0.5

    >>> hertz_to_normalized(np.array([11025, 22050, 44100]), 44100)
    array([ 0.5,  1. ,  2. ])
    """
    return (2. * f) / f_sample


def normalized_to_hertz(f, f_sample):
    """
    Convert a normalized frequency in
    :math:`\\pi \\frac{\\text{radians}}{\\text{sample}}` to frequency in
    Hertz

    Parameters
    ----------
    f : float or :py:class:`numpy.ndarray`
        Normalized frequency to convert
    f_sample : float
        Sample rate (Hertz)

    Returns
    -------
    float or :py:class:`numpy.ndarray`
        The frequency in Hertz

    Examples
    --------
    The Nyquist rate is exactly :math:`\\pi \\frac{\\text{radians}}{\\text{sample}}`

    >>> normalized_to_hertz(1., 200.)
    100.0

    The sample rate is exactly :math:`2\\pi \\frac{\\text{radians}}{\\text{sample}}`

    >>> normalized_to_hertz(2., 44100.)
    44100.0

    >>> normalized_to_hertz(0.5, 44100.)
    11025.0

    >>> normalized_to_hertz(np.array([2, 1.5, 1, 0.5]), 2000)
    array([ 2000.,  1500.,  1000.,   500.])
    """
    return (f * f_sample) / 2.


def amplitude_to_db(amplitude):
    """
    Convert amplitude ratios to decibels.

    Parameters
    ----------
    amplitude : float or :py:class:`numpy.ndarray`
        Amplitude ratio(s)

    Returns
    -------
    float or :py:class:`numpy.ndarray`
        Amplitude ratio(s) expressed in decibels

    Examples
    --------
    >>> amplitude_to_db(1)
    0.0

    >>> np.abs(amplitude_to_db(1.995) - 5.999) < 0.1
    True

    >>> amplitude_to_db(0.001)
    -60.0

    >>> amplitude_to_db(np.array([0.5, 4]))
    array([ -6.02059991,  12.04119983])

    >>> amplitude_to_db(0)
    -inf
    """
    return 20. * np.log10(amplitude)


def db_to_amplitude(db):
    """
    Convert measure(s) in decibels to amplitude ratio(s).

    Parameters
    ----------
    db : float or :py:class:`numpy.ndarray`
        Measure(s) in decibels

    Returns
    -------
    float or :py:class:`numpy.ndarray`
        Measure(s) expressed in amplitude ratios

    Examples
    --------
    >>> db_to_amplitude(0)
    1.0

    >>> np.abs(db_to_amplitude(5.999) - 1.995) < 0.1
    True

    >>> db_to_amplitude(-60)
    0.001

    >>> db_to_amplitude(np.array([-6.02059991,  12.04119983]))
    array([ 0.5,  4. ])
    """
    return 10 ** (db / 20.)

