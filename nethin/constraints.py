# -*- coding: utf-8 -*-
"""
This module contains constraints for weights and biases.

Created on Tue Nov 28 08:51:07 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
from six import with_metaclass

import keras.backend as K
from keras.utils import conv_utils

__all__ = ["BaseConstraint", "BoxConstraint"]


class BaseConstraint(with_metaclass(abc.ABCMeta, object)):
    """Base class for constraints.

    Parameters
    ----------
    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".
    """
    def __init__(self, data_format=None):

        self.data_format = conv_utils.normalize_data_format(data_format)

    def __call__(self, weights):
        """Constraint function (projection).

        Specialise this function in your subclasses.
        """
        return weights


class BoxConstraint(BaseConstraint):
    """Corresponds to a box constraint (upper and lower bounds).

    Parameters
    ----------
    lower : float
        The lower limit for the constraint if ``upper`` is given. Otherwise,
        the weights will be put in ``[-lower, lower]``.

    upper : float, optional
        The upper limit for the constraint. If not given, the weights will be
        put in ``[-lower, lower]``. Default is None, which means to use
        ``[-lower, lower]``.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".
    """
    def __init__(self,
                 lower,
                 upper=None,
                 data_format=None):

        super(BoxConstraint, self).__init__(data_format=data_format)

        if upper is None:
            self.lower = -abs(float(lower))
            self.upper = -self.lower
        else:
            self.lower = float(lower)
            self.upper = float(upper)

    def __call__(self, weights):
        """Applies the constraint to the weights (projection on feasible set).

        Parameters
        ----------
        inputs : Tensor
            The weight matrix for which the constraint should be applied.
        """
        return K.clip(weights, self.lower, self.upper)
