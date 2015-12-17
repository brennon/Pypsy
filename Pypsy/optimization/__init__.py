import numpy as np
import numpy.linalg

def cgd(start_val, error_fcn, h, crit_error, crit_iter, crit_h):
    """
    Conjugate gradient descent to optimize :math:`\\tau` parameters for EDA signal decomposition.

    Parameters
    ----------
    start_val : array_like
        The initial :math:`\\tau` parameters with :math:`\\tau_1` in ``tau[0]`` and :math:`\\tau_2` in ``tau[1]``
    error_fcn : function
        The error function to minimize. Should return a single ``float``.
    h : array_like
        The initial step sizes for the :math:`\\tau` parameters. Should match the shape of ``start_val``.
    crit_error : float
        Stopping criterion value for ``error_fcn``
    crit_iter : int
        Maximum number of iterations
    crit_h : float
        Minimum step size

    Returns
    -------
    x : :py:class:`numpy.ndarray`
        The optimized :math:`\\tau` parameters
    history : dict
        A record of the gradient descent process

    Raises
    ------
    TypeError
        If ``start_val`` or ``h`` are not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)
    """

    # FIXME: Add tests

    start_val = np.asarray(start_val)
    h = np.asarray(h)

    x = start_val
    newerror, _ = error_fcn(x)

    history = {}
    history['x'] = np.array(x)
    history['direction'] = np.zeros(x.size)
    history['step'] = np.array([-1])
    history['h'] = -np.ones(h.size)
    history['error'] = np.array([newerror])

    iter = 0

    while True:
        iter = iter + 1
        olderror = newerror

        if iter == 1:
            gradient = cgd_get_gradient(x, olderror, error_fcn, h)
            direction = -gradient

            if gradient.size == 0:
                break

        else:
            new_gradient = cgd_get_gradient(x, olderror, error_fcn, h)

            # Have not ported all optimization methods as they are all unused in MATLAB version of Ledalab.
            direction = -new_gradient

        if np.any(direction):
            # LINESEARCH
            x, newerror, step = cgd_linesearch(x, olderror, direction, error_fcn, h)
            error_diff = newerror - olderror

        else:
            error_diff = 0
            step = 0

        history['x'] = np.vstack((history['x'], x))
        history['direction'] = np.vstack((history['direction'], direction))
        history['step'] = np.vstack((history['step'], step))
        history['h'] = np.vstack((history['h'], h))
        history['error'] = np.vstack((history['error'], newerror))

        if iter > crit_iter:
            break

        if error_diff > -crit_error:
            h = h/2
            if np.all(h < crit_h):
                break

    return x, history

def cgd_get_gradient(x, error0, error_fcn, h):
    """
    Calculate the current gradient of an error function.

    Parameters
    ----------
    x : array_like
        The current parameter values
    error0 : float
        The current error
    error_fcn : function
        The error function
    h : array_like
        The current step sizes for each parameter in ``x``

    Returns
    -------
    out : float
        The computed gradient

    Raises
    ------
    TypeError
        If ``x`` or ``h`` are not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)
    """

    # FIXME: Add tests

    x = np.asarray(x)
    h = np.asarray(h)

    # Length of x
    Npars = x.size

    # Initialize gradient vector as a zero vector the length of x
    gradient = np.zeros(Npars)

    # Iterate i from 1 through the length of x
    for i in range(Npars):

        # Copy x
        xc = x.copy()

        # Add h[i] to x[i]
        xc[i] = xc[i] + h[i]

        # Calculate the error with the new x's
        error1, _ = error_fcn(xc)

        # If error is less than the error with which we began
        if error1 < error0:

            # Set the gradient for this x to be the error delta
            gradient[i] = error1 - error0

        # Try subtracting h
        else:

            # Fresh copy of x
            xc = x.copy()

            # Subtract h[i] from x[i]
            xc[i] = xc[i] - h[i]

            # Calculate error for new x's
            error1, _ = error_fcn(xc)

            # If error has decreased
            if error1 < error0:

                # Set gradient to be error delta
                gradient[i] = -(error1 - error0)

            # If error has not decreased
            else:

                # Set gradient for this x to be 0
                gradient[i] = 0

    gradient = gradient.T
    return gradient

def cgd_linesearch(x, error0, direction, error_fcn, h):
    """
    Calculate optimal step size for gradient descent (line search).

    Parameters
    ----------
    x : array_like
        The current parameter values
    error0 : float
        The current error
    direction : array_like
        The calculated step directions/magnitudes for each parameter in ``x``
    error_fcn : function
        The error function
    h : array_like
        The current step sizes for each parameter in ``x``

    Returns
    -------
    out : float
        The computed gradient

    Returns
    -------
    xc : :py:class:`numpy.ndarray`
        The new parameter values given the selected steps
    error1 : float
        The new value of ``error_fcn`` for the new parameters in ``xc``
    step : float
        The number of steps taken to reach these ``xc`` and ``error1``

    Raises
    ------
    TypeError
        If ``x``, ``direction``, or ``h`` are not array-like (cannot be converted to a :py:class:`numpy.ndarray` using
        :py:meth:`numpy.array()`)
    """

    # FIXME: Add tests

    x = np.asarray(x)
    direction = np.asarray(direction)
    h = np.asarray(h)

    direction_n = direction / np.linalg.norm(direction, ord=2)
    error_list = [error0]
    stepsize = h
    maxSteps = 6
    factor = np.zeros(1)

    for iStep in range(1, maxSteps):

        factor = np.concatenate([factor, [2**(iStep-1)]])
        xc = x.copy() + direction_n * stepsize * factor[iStep]
        error, xc = error_fcn(xc) # xc may be changed due to limits
        error_list.append(error)

        if error_list[-1] >= error_list[-2]: # end of decline
            if iStep == 1: # no success
                step = 0
                error1 = error0

            else: # parabolic
                p = np.polyfit(factor, error_list, 2)
                fx = np.arange(factor[0], factor[-1] + .1, .1)
                fy = np.polyval(p, fx)
                idx = np.argmin(fy)
                fxm = fx[idx]
                xcm = x.copy() + direction_n * stepsize * fxm
                error1, xcm = error_fcn(xcm) # xc may be changed due to limits

                if error1 < error_list[iStep - 1]:
                    xc = xcm.copy()
                    step = fxm

                else: # finding Minimum did not work
                    xc = x.copy() + direction_n * stepsize * factor[iStep-1] # before last point
                    error1, xc = error_fcn(xc) # recalculate error in order to check for limits again
                    step = factor[iStep-1]

            return xc, error1, step

    step = factor[iStep]
    error1 = error_list[iStep]

    return xc, error1, step

