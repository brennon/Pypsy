import numpy as np

__author__ = 'Brennon Bortz'


def constrain(value, minimum, maximum):
    """
    Constrain a value to be between a minimum and maximum value.

    Parameters
    ----------
    value : float
        The value to constrain
    minimum : float
        The minimum allowable value
    maximum : float
        The maximum allowable value

    Returns
    -------
    out : float
        Returns the original ``value`` if it lies between ``minimum`` and ``maximum``, inclusive. Otherwise, returns
        ``minimum`` if ``value`` is less than ``minimum``, or returns ``maximum`` if ``value`` is greater than
        ``maximum``.

    Notes
    -----
    ``value``, ``minimum``, and ``maximum`` are casted to :py:class:`numpy.float64`.

    Examples
    --------
    >>> constrain(1, 0.5, 1.5)
    1.0
    >>> constrain(0.5, 0.5, 1.5)
    0.5
    >>> constrain(1.5, 0.5, 1.5)
    1.5
    >>> constrain(-4, -3, 1)
    -3.0
    >>> constrain(5, -1, 2)
    2.0
    """
    minimum = np.float64(minimum)
    maximum = np.float64(maximum)
    value = np.float64(value)

    return np.max([np.min([value, maximum]), minimum])

