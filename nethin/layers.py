# -*- coding: utf-8 -*-
"""
Contains custom or adapted Keras layers.

Created on Thu Oct 12 14:35:08 2017

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import keras.backend as K
from keras.utils import conv_utils
import keras.layers.convolutional as convolutional

__all__ = ["Convolution2DTranspose"]


class Convolution2DTranspose(convolutional.Convolution2DTranspose):
    """Fixes an output shape error of ``Convolution2DTranspose``.

    Described in Tensorflow issue # 8972:

        https://github.com/tensorflow/tensorflow/issues/8972

    A fix has been issued, but may not have propagated. Use this wrapper to
    fix it. If you have the issue fixed in your version of Tensorflow, it will
    work exactly as the corresponding Keras ``Convolution2DTranspose`` layer,
    and nothing will be affected.

    Derived and adapted from:

        https://github.com/tensorflow/tensorflow/pull/13193/commits/6346745f18ded325cdd476d1e521b301b2f38db5

    and:

        https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py:Conv2DTranspose
    """
    def __init__(self, *args, **kwargs):

        super(Convolution2DTranspose, self).__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):

        outputs = super(Convolution2DTranspose, self).call(inputs,
                                                           *args, **kwargs)

        if not outputs.get_shape().is_fully_defined():

            input_shape = K.int_shape(inputs)
            if self.data_format == "channels_last":
                height, width = input_shape[1], input_shape[2]
            else:
                height, width = input_shape[2], input_shape[3]
            kernel_h, kernel_w = self.kernel_size
            stride_h, stride_w = self.strides
            batch_size = inputs.get_shape()[0]

            # Infer the dynamic output shape:
            out_height = conv_utils.deconv_length(height,
                                                  stride_h, kernel_h,
                                                  self.padding)
            out_width = conv_utils.deconv_length(width,
                                                 stride_w, kernel_w,
                                                 self.padding)

            if self.data_format == "channels_last":
                output_shape = (batch_size,
                                out_height, out_width, self.filters)
            else:
                output_shape = (batch_size,
                                self.filters, out_height, out_width)

            outputs.set_shape(output_shape)

        return outputs
