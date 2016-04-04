Pypsy
=====

Pypsy is a library for the analysis of of psychophysiological data. Currently,
Pypsy provides a suite of tools for the decomposition and analysis of
electrodermal activity (EDA) signals. Much of this functionality is a
port of the `Ledalab <http://www.ledalab.de>`_ software from the
`MATLAB <http://www.mathworks.com>`_ programming language.

There are two main subpackages in :py:mod:`Pypsy`: :py:mod:`Pypsy.signal` and
:py:mod:`Pypsy.optimization`. The :py:mod:`Pypsy.signal` module contains
resources for representing a psychophysiological signal time series. The
:py:mod:`Pypsy.optimization` module contains facilities for optimization, used
primarily in decomposing EDA signals.

The most interesting class in :py:mod:`Pypsy`:, by far, is
:py:class:`~Pypsy.signal.EDASignal`. This is the class to be used for
representing and decomposing EDA signals. The documentation of this class
demonstrates just how to do so in its examples.

Pypsy is being deceloped in conjunction with my own Ph.D. dissertation work. As
such, it is very much a work in progress, and features and the overall
architecture change often. If you are interested in using Pypsy, please feel
free to do so. If in doing so, you find yourself needing help, please create an
`issue <https://github.com/brennon/Pypsy/issues>`_.

Contents:

.. toctree::
   :maxdepth: 8

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

