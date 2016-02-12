import numpy as np
import Pypsy.signal.utilities
import sklearn.linear_model

__author__ = "Brennon Bortz"

def bateman(time, onset=0, amplitude=0, tau1=0.5, tau2=3.75):
    """
    Returns an :py:class:`numpy.ndarray` that represents the value of the Bateman function. The Bateman
    function is evaluated over ``time``, and is parameterized by ``tau1`` and ``tau2``.

    The Bateman function is:

    .. math::

        Bateman(t) = e^{\\frac{-t}{\\tau_2}} - e^{\\frac{-t}{\\tau_1}}

    Parameters
    ----------
    time : array_like
        The times (in seconds) at which the Bateman function should be evaluated
    onset : float
        The time (in seconds) of the maximum amplitude of the Bateman function
    amplitude : float
        The maximum amplitude of the Bateman function
    tau1 : float
        The :math:`\\tau_1` parameter for the Bateman function
    tau2 : float
        The :math:`\\tau_2` parameter for the Bateman function

    Returns
    -------
    out : :py:class:`numpy.ndarray`
        The values of the Bateman function evaluated at each point in ``time``

    Raises
    ------
    ValueError
        If ``tau1`` and ``tau2`` are not both greater than zero, or if they are equal
    TypeError
        If ``time`` is not array-like (cannot be converted to a
        :py:class:`numpy.ndarray` using :py:meth:`numpy.array()`)

    Examples
    --------
    >>> bateman_time = np.linspace(0, 20 - (1. / 25), 20 * 25)
    >>> evaluated = bateman(bateman_time, onset=4, amplitude=0.5, tau1=0.5, tau2=3.75)
    >>> np.where(evaluated == np.max(evaluated))[0][0]
    129
    >>> np.abs(0.5 - evaluated[129]) < 0.001
    True
    """
    time = np.asarray(time)

    # tau1 and tau2 must both be greater than 0
    if tau1 < 0 or tau2 < 0:
        raise ValueError('tau1 and tau2 must both be greater than zero.')

    # tau1 and tau2 must not be equal
    if tau1 == tau2:
        raise ValueError('tau1 and tau2 cannot be equal.')

    # Initialize a vector of zeros the length of time vector
    conductance = np.zeros(time.size)

    # Find indices in time vector that are greater than onset
    onset_range = time > onset

    # Return just the conductance zeros vector if no values in time vector were
    # greater than onset
    if not np.any(onset_range):
        return conductance

    # Create xr as the time vector subsetted to just those values that are
    # greater than onset. Subtract the value of onset from each item in this
    # vector.
    xr = time[onset_range] - onset;

    # If amplitude parameter is greater than zero
    if amplitude > 0:

        # Find the value of x at which bateman(x, tau1, tau2) is maximized. This is the value at which the first
        # derivative of bateman(x, tau1, tau2) equals 0
        max_x = tau1 * tau2 * np.log(tau1/tau2) / (tau1 - tau2)

        # Find the value of bateman(x, tau1, tau2) at max_x
        max_amp = np.abs(np.exp(-max_x/tau2) - np.exp(-max_x/tau1))

        # Define c to be the ratio of the amplitude passed as the argument and the maximum amplitude of
        # bateman(x, tau1, tau2)
        c =  amplitude/max_amp

    # If amplitude is equal to zero
    else:

        # Approximate area under bateman(x, tau1, tau2) and define c to scale output by 1/sampling rate, as calculated
        # by taking the reciprocal of the mean of the discrete first derivative of the timestamp vector
        fs = round(1. / np.mean(np.diff(time)));
        c = 1. / ((tau2 - tau1) * fs)

    # If tau1 is greater than zero
    if tau1 > 0:

        # Return c * bateman(x, tau1, tau2)
        conductance[onset_range] = c * (np.exp(-xr/tau2) - np.exp(-xr/tau1));

    else:

        # Return c * e^(-x/tau2)
        conductance[onset_range] = c * np.exp(-xr/tau2)

    return conductance


def bateman_gauss(time, onset=0, amplitude=0, tau1=3.75, tau2=0.5, sigma=0):
    """
    Returns an :py:class:`numpy.ndarray` that represents the value of the Bateman function. The Bateman function is
    evaluated over ``time``, and is parameterized by ``tau1`` and ``tau2``.

    The Bateman function is:

    .. math::

        Bateman(t) = e^{\\frac{-t}{\\tau_2}} - e^{\\frac{-t}{\\tau_1}}

    The output of the Bateman function is smoothed by convolution with a Gaussian window.

    Parameters
    ----------
    time : array_like
        The times (in seconds) at which the Bateman function should be evaluated
    onset : float
        The time (in seconds) of the maximum amplitude of the Bateman function
    amplitude : float
        The maximum amplitude of the Bateman function
    tau1 : float
        The :math:`\\tau_1` parameter for the Bateman function.
    tau2 : float
        The :math:`\\tau_2` parameter for the Bateman function.
    sigma : float
        The :math:`\\sigma` to be used for the Gaussian smoothing function

    Returns
    -------
    out : :py:class:`numpy.ndarray`
        The values of the Gaussian-smoothed Bateman function evaluated at each point in ``time``

    Raises
    ------
    ValueError
        If ``tau1`` and ``tau2`` are not both greater than zero, or if they are equal.
    TypeError
        If ``time`` is not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)
    """

    # FIXME: Add tests

    from scipy.stats import norm
    from scipy.signal import convolve

    time = np.asarray(time)

    # Generate the output of the Bateman Function with the provided parameters,
    # but with amplitude = 0
    component = bateman(time, onset, 0, tau1, tau2)

    # If a sigma parameter was provided that is greater than zero, window the
    # Bateman Function output with a Gaussian window
    if sigma > 0:

        # Estimate sampling rate
        sr = np.round(1. / np.mean(np.diff(time)))

        # We'll prepend and append extensions of the first and last values of
        # the Bateman output to either side of it. The length of each of these
        # extensions is 4*sigma
        winwidth2 = np.ceil(sr * sigma * 4)

        # Create a time vector for the Gaussian window that is 2 * winwidth2 + 1
        t = np.arange(1, (winwidth2 * 2 + 1) + 1)

        # Generate the Gaussian window across t, centered over the middle of t,
        # and with a standard deviation of sigma times the sampling rate.
        g = norm.pdf(t, winwidth2 + 1, sigma * sr)

        # Scale the amplitude of the window to equal the maximum amplitude of
        # the Bateman function output.
        g = g / np.max(g) * amplitude

        # Convolve the extended Bateman function output with the Gaussian
        # window.
        head = np.ones(winwidth2) * component[0]
        tail = np.ones(winwidth2) * component[-1]
        extended_component = np.concatenate([head, component, tail])
        bg = convolve(extended_component, g)

        # Return the 'interesting' portion of the result of the convolution
        # (trim the tail).
        component = bg[int((winwidth2*2+1) -1) : -(int(winwidth2*2))]

    return component


def get_peaks(data):
    """
    Returns the indices of the minima and maxima in ``data``. Minima and maxima are identified by searching for sign
    changes (zero crossings) in the first differential.

    There are two important caveats to this function. First, if the last extremum is a maximum, the last index of the
    signal is returned as the final minimum. Second, because this function looks for sign changes in the differential
    of ``data``, if the last index of the signal is a maximum, it will not be returned, as there is no successive value
    to generate a further sign change in the differential.

    Parameters
    ----------
    data : array_like
        The signal from which to extract peaks

    Returns
    -------
    minima : :py:class:`numpy.ndarray`
        Each entry in the `minima` array is an index of a minimum in the original signal.
    maxima : :py:class:`numpy.ndarray`
        Each entry in the `maxima` array is an index of a minimum in the original signal.

    Raises
    ------
    TypeError
        If ``data`` is not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)

    Examples
    --------
    >>> dc = np.array([1., 1, 1, 1, 1])
    >>> minima, maxima = get_peaks(dc)
    >>> minima.size
    0
    >>> maxima.size
    0
    >>> triangle = np.array([-1., 0, 1, 0, -1, 0, 1])
    >>> minima, maxima = get_peaks(triangle)
    >>> list(minima)
    [0, 4]
    >>> list(maxima)
    [2]
    """

    data = np.asarray(data)

    # First derivative (discrete) of data
    ddata = np.diff(data)

    extrema = np.array([])

    # Get all indices of nonzero values in ddata
    start_idx = np.where(ddata != 0)[0]

    # If there are no nonzero values, return empty minima/maxima vectors
    if start_idx.size == 0:
        return np.array([]), np.array([])

    # Start at index of first nonzero value in ddata
    start_idx = start_idx[0];

    # Get the sign of ddata value at the start index
    pos_neg = np.sign(ddata[start_idx])

    # Iterate over ccd from the next index of nonzero values after start index through the rest of ddata
    for i in range(start_idx + 1, len(ddata)):

        # If the sign of ddata at this index is the opposite of the sign of ddata at pos_neg
        if np.sign(ddata[i]) != pos_neg:

            # If the first extremum is a maximum
            if (extrema.size == 0 and pos_neg == 1):

                # predataidx is a vector of integers from one up to one before the current i
                predataidx = np.arange(start_idx, i)

                # Find the minimum of data before this point, but after start_idx
                pre_min = np.min(data[predataidx])
                pre_idx = np.where(data == pre_min)[0][0]
                shifted_idx = pre_idx + start_idx

                # Store the index of this minimum point as an extremum
                extrema = np.append(extrema, [shifted_idx])

            # Append the current extremum to the cccri vector
            extrema = np.append(extrema, [i])

            # Flip the sign (pos_neg)
            pos_neg = -pos_neg;

    # If last extremum is a maximum (if we have an even number of extrema) add a minimum after it
    if np.mod(extrema.size, 2) == 0:

        # Append the last index of data as the final minimum
        extrema = np.append(extrema, [data.size - 1]);

    # Sort the extrema
    extrema = np.sort(extrema)

    # The first and every other entry in cccri are all minima
    minima = extrema[0::2]

    # The second and every other entry in cccri are all maxima
    maxima = extrema[1::2]

    return (minima.astype(int), maxima.astype(int))


def significant_peaks(data, minima, maxima, threshold):
    """
    Find all peaks in a vector that are greater than a threshold value.

    Parameters
    ----------
    data : array_like
        A signal vector
    minima : array_like
        A vector of the indices of minima in ``data``
    maxima : array_like
        A vector of the indices of maxima in ``data``
    sig : float
        The absolute amplitude above which maximum are to be considered with respect to to their preceding and
        following minima

    Returns
    -------
    significant_minima : :py:class:`numpy.ndarray`
        A two-dimensional array. Each 'row' in the array corresponds to its respective entry in the `significant_maxima`
        array. The first 'column' in the row is the index of the minimum that immediately precedes the maximum referred
        to in `significant_maxima`. The second 'column' in the row is the index of the minimum that immediately follows
        the maximum referred to in `significant_maxima`. Note that *when there are no significant minima, this is an
        empty, single-dimensional array.*
    significant_maxima : :py:class:`numpy.ndarray`
        The indices of all maxima for which the absolute amplitude difference between the preceding and/or following
        minimum is greater than ``threshold``

    Raises
    ------
    TypeError
        If any of ``data``, ``minima``, or ``maxima`` are not array-like (cannot be converted to a
        :py:class:`numpy.ndarray` using :py:meth:`numpy.array()`)

    Examples
    --------
    >>> sig = np.array([0., 0.25, 0, 0.5, 0, 1, 0, 0.5, -0.2, 0, 0.25, 0])
    >>> minima, maxima = get_peaks(sig)
    >>> sig_minima, sig_maxima = significant_peaks(sig, minima, maxima, 0.6)
    >>> sig_minima[0]
    array([4, 6])
    >>> sig_minima[1]
    array([6, 8])
    >>> sig_maxima
    array([5, 7])
    >>> sig_minima, sig_maxima = significant_peaks(sig, minima, maxima, 1.)
    >>> sig_minima
    array([], dtype=int64)

    >>> sig = np.array([0, 1, 0])
    >>> minima, maxima = get_peaks(sig)
    >>> minima
    array([0, 2])
    >>> maxima
    array([1])
    >>> sig_minima, sig_maxima = significant_peaks(sig, minima, maxima, 0.1)
    >>> sig_minima
    array([0, 2])
    >>> sig_maxima
    array([1])
    """

    data = np.asarray(data)
    minima = np.asarray(minima)
    maxima = np.asarray(maxima)

    # Will be a vector of indices in data of all maxima in data that have a rise and/or fall of > sigc.
    kept_maxima = np.array([], dtype='int64')

    # Will be a length(maxL) x 2 matrix. Each row corresponds to a maxima in maxL. The first column gives the index of
    # the minimum immediately before the maximum, and the second column gives the index of the minimum immediately
    # following the maximum.
    kept_minima = np.array([], dtype='int64')

    # Return empty vectors if we didn't get any maxima
    if maxima.size == 0:
        return kept_minima, kept_maxima

    # Build a two-row matrix where each column corresponds to a maximum. The first row is the difference between the
    # maximum and its preceding minimaum. The second is the difference between the maximum and the next minimum.
    rises = data[maxima] - data[minima[:-1]]
    falls = data[maxima] - data[minima[1:]]
    rises_falls = np.array([rises, falls])

    # Take all maxima from the maxima vector where either its rise or fall is greater than sigc.
    kept_maxima = (maxima[np.max(rises_falls, axis=0) > threshold]).astype(int)

    # Iterate over maxL vector (indices of maxima that are > sigc)
    for i in range(kept_maxima.size):

        # Take minima immediately before and after this maximum
        min_idx_before = minima[np.where(minima < kept_maxima[i])[0][-1]]
        min_idx_after = minima[np.where(minima < kept_maxima[i])[0][-1] + 1]

        # Build up minL matrix. One row for every maximum (column) in minL. The columns are the indices of the minima
        # immediately before and after the corresponding maxima=um at the index in maxL.
        new_row = np.array([min_idx_before, min_idx_after]).astype(int)

        if kept_minima.size == 0:
            kept_minima = new_row
        else:
            kept_minima = np.vstack([kept_minima, new_row])

    return kept_minima, kept_maxima


def segment_driver(data, remainder, threshold, window_size):
    """
    Segment a signal into individual impulse_segments. Only those impulse_segments that are greater in amplitude than
    ``threshold`` are extracted. Also, extract corresponding overshoot data from a remainder signal for each extracted
    impulse. The maximum width of segments (in samples) is governed by ``window_size``.

    Parameters
    ----------
    data : array_like
        The signal to be segmented
    remainder : array_like
        A remainder signal from which to extract overshoot data
    threshold : float
        The criterion amplitude for determining those impulse_segments that should be extracted. Impulses are only
        extracted if the absolute amplitude difference between the segment peak and its preceding and/or following
        minimum is greater than ``threshold``.
    maximum_width : int
        The maximum width of a segment in samples

    Returns
    -------
    onsets : numpy.ndarray
        A vector of time indices in the signal that correspond to the start points of the impulse windows
    impulse_segments : list(numpy.ndarray)
        A list of segments of the signal containing a vector for each impulse
    overshoot_segments : list(numpy.ndarray)
        A list of segments of the remainder signal corresponding to overshoot
    impulse_minima : numpy.ndarray
        A matrix of time indices of the data_minima surrounding the data_maxima in ``impulse_maxima``--the first column is the
        timestamp of the data_minima preceding the corresponding impulse in ``impulse_maxima`` and the second column is the
        timestamp of the data_minima following the corresponding impulse in ``impulse_maxima``
    impulse_maxima : numpy.ndarray
        A vector of time indices of the data_maxima of each impulse in ``onsets``

    Raises
    ------
    TypeError
        If either ``data`` or ``remainder`` are not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)

    Examples
    --------
    >>> d = np.array([0, 0, 0.4, 0, 1,   0,   0.3, 0, 0])
    >>> r = np.array([0, 0, 0,   0, 0.5, 0.6, 0,   0, 0])
    >>> onsets, impulse_segments, overshoot_segments, impulse_minima, impulse_maxima = segment_driver(d, r, 0.5, 4)
    >>> onsets
    array([3])
    >>> impulse_segments
    [array([ 0.,  1.,  0.,  0.])]
    >>> overshoot_segments
    [array([ 0. ,  0.5,  0.6,  0. ])]
    >>> impulse_minima
    array([[3, 5]])
    >>> impulse_maxima
    array([4])

    >>> d = np.array([0, 0, 0.4, 0, 1,   0,   0.3, 0, 0])
    >>> r = np.array([0, 0, 0,   0, 0.5, 0.6, 0,   0, 0])
    >>> onsets, impulse_segments, overshoot_segments, impulse_minima, impulse_maxima = segment_driver(d, r, 0.3, 4)
    >>> onsets
    array([1, 3])
    >>> impulse_segments
    [array([ 0. ,  0.4,  0. ,  0. ]), array([ 0.,  1.,  0.,  0.])]
    >>> overshoot_segments
    [array([ 0.,  0.,  0.,  0.]), array([ 0. ,  0.5,  0.6,  0. ])]
    >>> impulse_minima
    array([[1, 3],
           [3, 5]])
    >>> impulse_maxima
    array([2, 4])

    >>> d = np.array([0, 0, 0.4, 0, 1,   0,   0.3, 0, 0])
    >>> r = np.array([0, 0, 0,   0, 0.5, 0.6, 0,   0, 0])
    >>> onsets, impulse_segments, overshoot_segments, impulse_minima, impulse_maxima = segment_driver(d, r, 1.5, 4)
    >>> onsets
    array([], dtype=int64)
    >>> impulse_segments
    []
    >>> overshoot_segments
    []
    >>> impulse_minima
    array([], dtype=int64)
    >>> impulse_maxima
    array([], dtype=int64)
    """

    data = np.asarray(data)
    remainder = np.asarray(remainder)

    # Initialize outputs
    onsets = np.array([], dtype='int64')
    impulse_segments = []
    overshoot_segments = []
    impulse_minima = np.array([])
    impulse_maxima = np.array([])

    # Get indices of minima and maxima in data vector
    data_minima, data_maxima = get_peaks(data)

    # If there were no data_minima or data_maxima, return initial values
    if data_maxima.size == 0:
        return onsets, impulse_segments, overshoot_segments, impulse_minima, impulse_maxima

    # Get matrices for significant peaks
    data_significant_minima, data_significant_maxima = significant_peaks(data, data_minima, data_maxima, threshold)

    # Get data_minima/data_maxima and significant data_minima/data_maxima segments for remainder
    remainder_minima, remainder_maxima = get_peaks(remainder)
    remainder_significant_minima, remainder_significant_maxima = significant_peaks(
        remainder,
        remainder_minima,
        remainder_maxima,
        .005
    )

    # We need to stack another minimum row if there is only one row. Otherwise, this is a single-dimensional array and
    # we'll have indexing problems.
    impulse_minima_row_added = False
    if len(data_significant_minima.shape) == 1 and data_significant_minima.shape[0] != 0:
        data_significant_minima = np.vstack([data_significant_minima, [0, 0]])
        impulse_minima_row_added = True

    if len(remainder_significant_minima.shape) == 1 and remainder_significant_minima.shape[0] != 0:
        remainder_significant_minima = np.vstack([remainder_significant_minima, [0, 0]])

    # Iterate over significant data_maxima in signal
    for i in range(len(data_significant_maxima)):

        # We start a segment at either the minimum before the maximum, or half a segment width before the
        # maximum--whichever is later. The index of this start point is stored in segment_start. segment_end is the
        # index  of the start index + a segment width, or the end of the data--whichever is earlier.
        segment_start = np.int64(
            np.max(
                [data_significant_minima[i, 0], data_significant_maxima[i] - np.round(window_size / 2)]
            )
        )
        segment_end = np.int64(np.min([segment_start + window_size, data.size]))

        # Indices of entire segment
        segment_indices = np.arange(segment_start, segment_end)

        # Data subsetted by segment indices
        segment_data = data[segment_indices]

        # If the window (segment) extends beyond the minimum following the maximum, zero that portion of the signal.
        segment_data[segment_indices >= data_significant_minima[i, 1]] = 0

        # Save segment start index into onset vector
        onsets = np.append(onsets, [segment_start])

        # Save segment signal into impulse cell array
        impulse_segments.append(segment_data)

        # Zero vector the length of the segment
        overshoot_data = np.zeros(segment_indices.size)

        remainder_indices = np.array([])

        # If we haven't reached the last significant maximum in data
        if i < (data_significant_maxima.size - 1):

            # Get the indices of values in the vector of significant maxima extracted from the remainder that are after
            # the current significant maximum in data and before the next significant maximum in data
            remainder_indices = np.nonzero(
                (remainder_significant_maxima > data_significant_maxima[i]) &
                (remainder_significant_maxima < data_significant_maxima[i+1])
            )[0]

        else:

            # Get the indices of values in the vector of significant maxima extracted from the remainder that are after
            # the current significant maximum in data
            remainder_indices = np.nonzero(remainder_significant_maxima > data_significant_maxima[i])[0]

        # If there were no significant data_maxima in the remainder between the current significant maximum in data and the next one
        if remainder_indices.size == 0:

            # If we haven't reached the last significant maximum in data
            if i < (data_significant_maxima.size - 1):

                # Get the indices of values in the vector of ALL maxima extracted from the remainder that are after the
                # current significant maximum in data and before the next significant maximum in data
                remainder_indices = np.nonzero(
                    (remainder_maxima > data_significant_maxima[i]) &
                    (remainder_maxima < data_significant_maxima[i+1])
                )[0]

            else:

                # Get the indices of values in the vector of ALL maxima extracted from the remainder that are after the
                # current significant maximum in data
                remainder_indices = np.nonzero(remainder_maxima > data_significant_maxima[i])[0]

            # FIXME: Pretty sure this is a bug in Ledalab.
            # Assign ALL remainder maxima to remainder_significant_maxima and ALL remainder minima to
            # remainder_significant_minima (converting to the signpeaks format for the data_minima)
            remainder_significant_maxima = remainder_maxima
            starts = remainder_minima[0:-1]
            ends = remainder_minima[1:]
            remainder_significant_minima = np.array([starts, ends]).T

        # If there were significant data_maxima in the remainder between the current data_significant_maxima and the next one
        if not remainder_indices.size == 0:

            # Get the first one
            remainder_indices = remainder_indices[0]
            # print(remainder_indices)

            # Get the later index of either the minimum before remainder significant peak or the beginning of this
            # segment
            overshoot_start = np.max([remainder_significant_minima[remainder_indices, 0], segment_start])

            # Get the earlier index of either the minimum after remainder significant peak or the end of this segment
            overshoot_end = np.min([remainder_significant_minima[remainder_indices, 1], segment_end])

            # Subset remainder from overshoot_start to overshoot_end and put this vector into the appropriate bins in
            # overshoot_data
            overshoot_start_index = overshoot_start - segment_start
            overshoot_end_index = overshoot_end - segment_start
            overshoot_data[overshoot_start_index:overshoot_end_index] = remainder[overshoot_start:overshoot_end]

        # Insert this segment's overshoot into the overshoot_segments cell array
        overshoot_segments.append(overshoot_data)

    # Return data_minima and data_maxima results from significant peak analysis in impulse_minima and impulse_maxima
    if impulse_minima_row_added:
        impulse_minima = data_significant_minima[:-1]
    else:
        impulse_minima = data_significant_minima
    impulse_maxima = data_significant_maxima

    return onsets, impulse_segments, overshoot_segments, impulse_minima, impulse_maxima


def interimpulse_fit(driver, kernel, minima, maxima, original_time, original_signal, original_fs):
    """
    Estimate the tonic EDA driver and signal. The mean (or median) of data between impulses is used to estimate the
    tonic driver. This driver is then convolved with the kernel to produce the tonic signal.

    Parameters
    ----------
    driver : array_like
        The composite EDA driver
    kernel : array_like
        The kernel impulse response
    minima : numpy.ndarray
        The minima in the composite EDA driver
    maxima :
        The maxima in the composite EDA driver
    original_time : array_like
        The timestamps of the original EDA signal
    original_signal : array_like
        The original EDA signal
    original_fs : float
        The sample rate of the original EDA signal

    Returns
    -------
    tonic_driver : numpy.ndarray
        The estimated tonic EDA driver
    tonic_data : numpy.ndarray
        The estimated tonic EDA signal
    """

    # FIXME: Add tests

    from scipy.interpolate import pchip_interpolate
    from scipy.signal import convolve

    tonic_driver = np.array([])
    tonic_data = np.array([])

    # Original timestamps (comes from leda2.analysis0.target.t)
    original_time = original_time.copy()

    # Original EDA signal (comes from leda2.analysis0.target.d)
    original_signal = original_signal.copy()

    # Original sample rate (comes from leda2.analysis0.target.sr)
    original_fs = original_fs

    # Hard coded value came from observing first run during continuous analysis (comes from
    # leda2.set.tonicGridSize_sdeco)
    grid_spacing = 10

    # Number of samples in the kernel
    kernel_length = kernel.size

    # Get inter-impulse data index
    interimpulse_indices = np.array([], dtype='int64')

    # If there are more than two maximum indices in maxima
    if maxima.size > 2:

        # Iterate from the first to the penultimate
        for i in range(maxima.size - 1):

            # Indices of samples between following minimum and next minimum. These indices are the indices of the gaps
            # between impulses.
            current_interimpulse_indices = np.arange(minima[i, 1], minima[i + 1, 0] + 1)

            # Append these indices to interimpulse_indices
            interimpulse_indices = np.concatenate([interimpulse_indices, current_interimpulse_indices])

        # Add the index of the first minimum for the second maximum to the beginning of interimpulse_indices. Add the index of the second minimum for the last maximum, plus all remaining samples except for the last second of data, to the end of interimpulse_indices.
        interimpulse_indices = np.concatenate([[minima[1, 0]], interimpulse_indices, np.arange(minima[-1, 1], driver.size - original_fs + 1)])

    # There weren'original_time any maxima (except for first and last 'global' maxima), so
    # the entire driver is tonic. Set interimpulse_indices to be all those indices where the
    # timestamp is greater than zero.
    else:  #no peaks (exept for pre-peak and may last peak) so data represents tonic only, so ise all data for tonic estimation
        interimpulse_indices = np.nonzero(original_time > 0)[0]

    interimpulse_indices = np.int64(interimpulse_indices)

    # interimpulse_times are the timestamps corresponding to the interimpulse_indices indices.
    interimpulse_times = original_time[interimpulse_indices]

    # interimpulse_data is the driver corresponding to the interimpulse_indices indices.
    interimpulse_data = driver[interimpulse_indices]

    # I don'original_time know what the significance of the name grid_time is, but
    # grid_time is a vector from 0 to the penultimate timestamp with a step size
    # of grid_spacing, and the last timestamp added to the end.
    grid_time = np.arange(0, original_time[-2], grid_spacing)
    grid_time = np.concatenate([grid_time, [original_time[-1]]])

    ### I don'original_time know why they do this...
    if grid_spacing < 30:
        grid_spacing = grid_spacing * 2

    # Initialize vector of ground levels
    tonic_level = np.zeros(grid_time.size)

    # Iterate over length of grid_time vector
    for i in range(0, grid_time.size):

        # Select relevant interimpulse time points for tonic estimate at
        # grid_time

        # If i represents the first grid_time entry
        if i == 0:

            # time_indices is all inter-impulse timestamp indices that are less than or equal
            # to the first grid_time entry plus the grid size, and greater
            # than one.
            time_indices = np.nonzero((interimpulse_times <= (grid_time[i] + grid_spacing)) & (interimpulse_times > 1.0))[0]

            # grid_indices is the same except using all timestamps.
            grid_indices = np.nonzero((original_time <= (grid_time[i] + grid_spacing)) & (original_time > 1.0))[0]

        # If i represents the last grid_time entry
        elif i == grid_time.size - 1:

            # time_indices is all inter-impulse timestamp indices that are after the last
            # grid_time entry minus the grid size, and less than the last
            # overall timestamp minus one second.
            time_indices = np.nonzero((interimpulse_times > (grid_time[i] - grid_spacing)) & (interimpulse_times < (original_time[-1] - 1.0)))[0]

            # grid_indices is the same except using all timestamps.
            grid_indices = np.nonzero((original_time > (grid_time[i] - grid_spacing)) & (original_time < (original_time[-1] - 1.0)))[0]

        else:

            # time_indices is all inter-impulse timestamp indices that are a half a grid size
            # before and after grid_time(i)
            # TODO: The spacing calculated here is different from Ledalab. Confirm that this is a bug in Ledalab.
            time_indices = np.nonzero((interimpulse_times > (grid_time[i] - grid_spacing/2)) & (interimpulse_times <= (grid_time[i] + grid_spacing/2)))[0]

            # grid_indices is the same except using all timestamps.
            grid_indices = np.nonzero((original_time > (grid_time[i] - grid_spacing/2)) & (original_time <= (grid_time[i] + grid_spacing/2)))[0]

        # Estimate tonic_level at grid_time
        # If there are more than two inter-impulse timestamps in this grid
        # window
        if time_indices.size > 2:

            # Take the ground level as the minimum of the mean of inter-impulse
            # data over the window, or the original signal at the nearest
            # possible time index.
            closest_index, closest_time = Pypsy.signal.utilities.closest_time_index(original_time, grid_time[i])
            tonic_level[i] = np.min(np.array([np.mean(interimpulse_data[time_indices]),  original_signal[closest_index]]))

        # If there are two or fewer inter-impulse timestamps in this grid
        # window
        else:

            # Take the ground level as the minimum of the median of
            # inter-impulse data over the window, or the original signal at the
            # nearest possible time index.
            closest_index, closest_time = Pypsy.signal.utilities.closest_time_index(original_time, grid_time[i])
            tonic_level[i] = np.min([np.median(driver[grid_indices]),  original_signal[closest_index]])

    # tonic_driver is the tonic_level signal PCHIP interpolated across the
    # original timestamps
    tonic_driver = pchip_interpolate(grid_time, tonic_level, original_time)

    # Stash currently-computed data
    grid_time_stored = grid_time
    tonic_level_stored = tonic_level
    tonic_driver_stored = tonic_driver

    # Prepend tonic_driver with a kernel-length of the initial value and
    # convolve it with the kernel.
    tonic_driver_extended = np.concatenate([np.ones(kernel_length) * tonic_driver[0], tonic_driver])
    tonic_data = convolve(tonic_driver_extended, kernel)

    # Trim kernel length from beginning and end of convolved signal
    tonic_data = tonic_data[kernel_length:tonic_data.size - kernel_length + 1]

    # Stash data
    tonic_data_stored = tonic_data

    # Correction for tonic sections still higher than raw data
    # Move closest grid_time at time of maximum difference of tonic surpassing data

    # Iterate i from length(grid_time) - 1 down to 0
    for i in range(grid_time.size - 2, 0, -1):

        # Get subrange of original timestamps that are closest to our
        # grid_time timestamps
        time_indices = Pypsy.signal.utilities.subrange_indices(original_time, grid_time[i], grid_time[i+1])

        # Find the max in the tonic data convolved with the kernel plus a
        # minimum allowable distance (defaults to 0) minus the original signal
        minimum_difference = 0
        eps = np.finfo(float).eps
        difference = (tonic_data[time_indices] + minimum_difference) - original_signal[time_indices]
        maximum_difference = np.max(difference)

        # If this max is greater than the distance from 1.0 to the next larget
        # double-precision number (2.2204e-16)
        if maximum_difference > eps:

            index = np.nonzero(difference == maximum_difference)[0]

            # Subtract this max from this tonic_level and the next
            tonic_level[i] = tonic_level[i] - maximum_difference
            tonic_level[i+1] = tonic_level[i+1] - maximum_difference

            # Interpolate tonic_driver and convolve again
            tonic_driver = pchip_interpolate(grid_time, tonic_level, original_time)
            tonic_driver_extended = np.concatenate([np.ones(kernel_length) * tonic_driver[0], tonic_driver])
            tonic_data = convolve(tonic_driver_extended, kernel)

            # Trim kernel length from beginning and end of convolved signal
            tonic_data = tonic_data[kernel_length:tonic_data.size - kernel_length + 1]

    # FIXME: The actual Ledalab function saves all this analysis (and the piecewise polynomial) back to globals

    return tonic_driver, tonic_data


def fit_error(data, fit, npar, errortype='MSE'):
    """
    Calculate the fit error for a model.

    Parameters
    ----------
    data : array_like
        The original data
    fit : array_like
        The fitted data generated by the model
    npar : int
        Number of model parameters
    errortype : str
        The type of error to calculate. Available options are ``'MSE'`` (default), ``'RMSE'``, and ``'adjR2'``.

    Returns
    -------
    out : float
        The model error

    Examples
    --------
    >>> data = np.array([-1.0, -0.5, 0, 0.5, 1])
    >>> fit = np.array([0.0, 0, 0, 0, 0])
    >>> fit_error(data, fit, 0, errortype='MSE')
    0.5
    >>> np.abs(fit_error(data, fit, 0, errortype='RMSE') - 0.70710678118654757) < 0.001
    True
    >>> np.abs(fit_error(data, fit, 0, errortype='adjR2') - 0.43431457505076199) < 0.001
    True
    """

    # Residual is difference between data and model
    residual = data - fit

    n = data.size

    # Degrees of freedom = length of signal minus number of parameters
    df = n - npar

    # Calculate sum of squared errors
    SSE = np.sum(residual**2);

    error = 0

    if errortype == 'MSE':

        # Mean squared error
        error = SSE / n

    elif errortype == 'RMSE':

        # Root mean squared error
        error = np.sqrt(SSE / n)

    elif errortype == 'adjR2':

        # Adjusted r-squared
        # For optimization use 1 - adjR2 since we want to minimize the function
        SST = np.std(data) * n
        r2 = 1. - (SSE / SST)
        error = 1. - (1. - r2) * (n - 1.) / df

        # FIXME: Is Chi-squared ever used in Ledalab? If so, this should be ported.
        # Chi-squared
        # case 'Chi2'
        #     error = SSE/leda2.data.conductance.error;

    return error


def linear_fit(x_data, y_data):
    """
    Compute a straight line fit for a given ``x_data`` and ``y_data``.

    Parameters
    ----------
    x_data : array_like
        An array of values of a function evaluated at each point in ``y_data``
    y_data : array_like
        An array of points at which the function evaluated

    Returns
    -------
    intercept : float
        The calculated intercept
    slope : float
        The calculated slope

    Raises
    ------
    ValueError
        If ``x_data`` and ``y_data`` are not the same length
    TypeError
        If ``x_data`` or ``y_data`` are not array-like (cannot be converted to
        a :py:class:`numpy.ndarray` using :py:meth:`numpy.array()`)

    Examples
    --------
    >>> time = np.array([0, 1, 2])
    >>> signal = np.array([5, 6, 7])
    >>> intercept, slope = linear_fit(time, signal)
    >>> np.testing.assert_almost_equal(intercept, 5)
    >>> np.testing.assert_almost_equal(slope, 1)

    >>> linear_fit([0, 1, 2], [0, 1])
    Traceback (most recent call last):
      ...
    ValueError: x_data and y_data must be the same length
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if len(x) != len(y):
        raise ValueError('x_data and y_data must be the same length')

    model = sklearn.linear_model.LinearRegression()
    fit = model.fit(x, y)
    return fit.intercept_[0], fit.coef_[0][0]

