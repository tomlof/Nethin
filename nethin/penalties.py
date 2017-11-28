# -*- coding: utf-8 -*-
"""
This module contains penalties for activations, weights and biases.

Created on Mon Nov  6 13:43:46 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
from six import with_metaclass

import keras.backend as K
from keras.utils import conv_utils


class BasePenalty(with_metaclass(abc.ABCMeta, object)):
    """Base class for penalties.

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

        return self.call(weights)

    @abc.abstractmethod
    def call(self, weights):
        raise NotImplementedError('"call" has not been specialised.')

    def get_config(self):
        return {"data_format": self.data_format}


class TotalVariation2D(BasePenalty):
    """Corresponds to the total variation (TV) penalty for 2D images.

    Parameters
    ----------
    gamma : float
        Non-negative float. The regularisation constant for the penalty.

    mean : bool
        Whether or not to compute the mean total variation over the batches or
        the sum of them. Default is True, compute the mean.

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
                 gamma,
                 mean=True,
                 data_format=None):

        super(TotalVariation2D, self).__init__(data_format=data_format)

        self.gamma = max(0.0, float(gamma))
        self.mean = bool(mean)

    def call(self, weights):
        """Computes the total variation of the input.

        Parameters
        ----------
        inputs : Tensor
            The weight matrix for which the total variation should be computed.
            Should have the shape (B, H, W, C), if
            ``data_format=channels_last``, and (B, C, H, W) if
            ``data_format=channels_first``.
        """
        if self.gamma > 0.0:
            if self.data_format == "channels_last":
                dif1 = weights[:, 1:, :, :] - weights[:, :-1, :, :]
                dif2 = weights[:, :, 1:, :] - weights[:, :, :-1, :]

                edge1 = dif1[:, :, -1:, :]
                edge2 = dif2[:, -1:, :, :]

                dif1 = dif1[:, :, :-1, :]
                dif2 = dif2[:, :-1, :, :]
            else:
                dif1 = weights[:, :, 1:, :] - weights[:, :, :-1, :]
                dif2 = weights[:, :, :, 1:] - weights[:, :, :, :-1]

                edge1 = dif1[:, :, :, -1:]
                edge2 = dif2[:, :, -1:, :]

                dif1 = dif1[:, :, :, :-1]
                dif2 = dif2[:, :, :-1, :]

            outputs = K.sum(K.sum(K.sum(K.sqrt(K.square(dif1) + K.square(dif2)),
                                        axis=1),
                                  axis=1),
                            axis=1) \
                    + K.sum(K.sum(K.sum(K.abs(edge1), axis=1), axis=1), axis=1) \
                    + K.sum(K.sum(K.sum(K.abs(edge2), axis=1), axis=1), axis=1)
            if self.mean:
                outputs = K.mean(outputs)
            else:
                outputs = K.sum(outputs)

            outputs = self.gamma * outputs
        else:
            outputs = self.gamma * weights[0, 0, 0, 0]

        return outputs

    def get_config(self):
        base_config = super(TotalVariation2D, self).get_config()
        config =  {"gamma": self.gamma,
                   "mean": self.mean}
        return dict(list(base_config.items()) + list(config.items()))
