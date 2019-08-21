# -*- coding: utf-8 -*-
"""
Contains custom or adapted Keras padding layers.

Created on Mon Oct  9 13:57:33 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from tensorflow.python import keras as tf_keras

__all__ = ["ReflectPadding2D"]


class ReflectPadding2D(tf_keras.engine.base_layer.Layer):
    """Reflection-padding layer for 2D input (e.g. an image).

    This layer adds rows and columns of reflected versions of the input at the
    top, bottom, left and right side of an image tensor.

    Parameters
    ----------
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int, the same symmetric padding is applied to width and height. If
        tuple of 2 ints, interpreted as two different symmetric padding values
        for height and width: `(symmetric_height_pad, symmetric_width_pad)`. If
        tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad),
        (left_pad, right_pad))`

    data_format : str
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    Examples
    --------
    >>> import numpy as np
    >>> import tensorflow.keras as tf_keras
    >>>
    >>> from nethin.padding import ReflectPadding2D
    >>>
    >>> A = np.arange(12).reshape(3, 4).astype(np.float32)
    >>>
    >>> inputs = tf_keras.layers.Input(shape=(3, 4, 1))
    >>> x = ReflectPadding2D(padding=2, data_format="channels_last")(inputs)
    >>> model = tf_keras.models.Model(inputs=inputs, outputs=x)
    >>> model.predict(A.reshape(1, 3, 4, 1)).reshape(7, 8)
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[10.,  9.,  8.,  9., 10., 11., 10.,  9.],
           [ 6.,  5.,  4.,  5.,  6.,  7.,  6.,  5.],
           [ 2.,  1.,  0.,  1.,  2.,  3.,  2.,  1.],
           [ 6.,  5.,  4.,  5.,  6.,  7.,  6.,  5.],
           [10.,  9.,  8.,  9., 10., 11., 10.,  9.],
           [ 6.,  5.,  4.,  5.,  6.,  7.,  6.,  5.],
           [ 2.,  1.,  0.,  1.,  2.,  3.,  2.,  1.]], dtype=float32)
    >>>
    >>> inputs = tf_keras.layers.Input(shape=(1, 3, 4))
    >>> x = ReflectPadding2D(padding=1, data_format="channels_first")(inputs)
    >>> model = tf_keras.models.Model(inputs=inputs, outputs=x)
    >>> model.predict(A.reshape(1, 1, 3, 4)).reshape(5, 6)
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[ 5.,  4.,  5.,  6.,  7.,  6.],
           [ 1.,  0.,  1.,  2.,  3.,  2.],
           [ 5.,  4.,  5.,  6.,  7.,  6.],
           [ 9.,  8.,  9., 10., 11., 10.],
           [ 5.,  4.,  5.,  6.,  7.,  6.]], dtype=float32)
    """
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):

        super(ReflectPadding2D, self).__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))

            height_padding = tf_keras.utils.conv_utils.normalize_tuple(
                    padding[0], 2, "1st entry of padding")
            width_padding = tf_keras.utils.conv_utils.normalize_tuple(
                    padding[1], 2, "2nd entry of padding")
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))

        self.data_format = tf_keras.utils.conv_utils.normalize_data_format(
                data_format)

        self.input_spec = tf_keras.engine.InputSpec(ndim=4)

    def build(self, input_shape):

        super(ReflectPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":

            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], rows, cols, input_shape[3])

        elif self.data_format == "channels_first":

            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], input_shape[1], rows, cols)

    def get_config(self):

        config = {"padding": self.padding,
                  "data_format": self.data_format}
        base_config = super(ReflectPadding2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Performs the actual padding.

        Parameters
        ----------
        inputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, rows, cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, rows, cols)`

        Returns
        -------
        outputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, padded_rows, padded_cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, padded_rows, padded_cols)`
        """
        from tensorflow.keras import backend as K

        outputs = K.spatial_2d_padding(inputs,
                                       padding=self.padding,
                                       data_format=self.data_format)

        p00, p01 = self.padding[0][0], self.padding[0][1]
        p10, p11 = self.padding[1][0], self.padding[1][1]
        if self.data_format == "channels_last":

            row0 = K.concatenate([inputs[:, p00:0:-1, p10:0:-1, :],
                                  inputs[:, p00:0:-1, :, :],
                                  inputs[:, p00:0:-1, -2:-2-p11:-1, :]],
                                 axis=2)
            row1 = K.concatenate([inputs[:, :, p10:0:-1, :],
                                  inputs,
                                  inputs[:, :, -2:-2-p11:-1, :]],
                                 axis=2)
            row2 = K.concatenate([inputs[:, -2:-2-p01:-1, p10:0:-1, :],
                                  inputs[:, -2:-2-p01:-1, :, :],
                                  inputs[:, -2:-2-p01:-1, -2:-2-p11:-1, :]],
                                 axis=2)

            outputs = K.concatenate([row0, row1, row2], axis=1)

        else:  # self.data_format == "channels_first"

            row0 = K.concatenate([inputs[:, :, p00:0:-1, p10:0:-1],
                                  inputs[:, :, p00:0:-1, :],
                                  inputs[:, :, p00:0:-1, -2:-2-p11:-1]],
                                 axis=3)
            row1 = K.concatenate([inputs[:, :, :, p10:0:-1],
                                  inputs,
                                  inputs[:, :, :, -2:-2-p11:-1]],
                                 axis=3)
            row2 = K.concatenate([inputs[:, :, -2:-2-p01:-1, p10:0:-1],
                                  inputs[:, :, -2:-2-p01:-1, :],
                                  inputs[:, :, -2:-2-p01:-1, -2:-2-p11:-1]],
                                 axis=3)

            outputs = K.concatenate([row0, row1, row2], axis=2)

        return outputs


if __name__ == "__main__":
    import doctest
    doctest.testmod()
