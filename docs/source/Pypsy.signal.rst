Pypsy.signal
============

Objects of the :py:class:`~Pypsy.signal.Signal` class (and its subclasses) are
used to represent psychophysiological signal time series. The
:py:class:`~Pypsy.signal.Signal` class can be used to represent any signal with
data and associated time points, while the :py:class:`~Pypsy.signal.EDASignal`
class contains special functionality for decomposing EDA signals into their
tonic and phasic components.

You likely won't need to access the following submodules directly, but they
are documented, nevertheless:

.. toctree::
    :maxdepth: 1

    Pypsy.signal.analysis
    Pypsy.signal.conversion
    Pypsy.signal.filter
    Pypsy.signal.utilities

.. automodule:: Pypsy.signal
    :members:
    :undoc-members:
    :show-inheritance:
