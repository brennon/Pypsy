__author__ = 'brennon'

import numpy as np

class Signal(object):
    """
    YOWZA WOW
    """

    data = np.array([])
    time = np.array([])

    def __init__(self, time, data):
        """
        SOMETHING HERE!

        :param time:
        :type time:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        self.data = np.array(data)
        self.time = np.array(time)

class EDASignal(Signal):
    """
    This represents an EDA signal. And changing things is working!
    """
    pass
