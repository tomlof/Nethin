# -*- coding: utf-8 -*-
"""
Contains custom or adapted layers.

Created on Thu Oct 12 14:35:08 2017

Copyright (c) 2017-2022, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import six
from enum import Enum

import numpy as np

import tensorflow as tf
# from tensorflow.python import keras as tf_keras
import tensorflow.keras.layers
import tensorflow.keras.backend as K

__all__ = ["SharpCosSim2D",
           "MaxPooling2D_", "MaxUnpooling2D_",
           "MaxPoolingMask2D", "MaxUnpooling2D",
           "Convolution1D",
           "Convolution2DTranspose", "Resampling2D"]


class SharpCosSim2D(tensorflow.keras.layers.Layer):
    """A Sharpened Cosine Similarity layer for 2D inputs.

    Adapted from Raphael Pisoni's implementation at:

        https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB

    Parameters
    ----------
    filters: int, one of [1, 3, 5]
        Non-negative integer. The dimensionality of the output space (i.e. the
        number of output filters in the convolution).

    kernel_size: int, one of [1, 3, 5]
        An integer or tuple/list of 2 integers, specifying the size of the 2D
        window. The single integer specifies the same value for all spatial
        dimensions.

    strides: int, optional
        A non-negative integer specifying the stride along the height and
        width. The single integer specify the value for all spatial dimensions.
        The default value is 1.

    depthwise_separable: bool, optional
        The separable operation consist of first performing a depthwise spatial
        filter operation (which acts on each input channel separately) followed
        by a pointwise operation which mixes the resulting output channels. The
        default value is False.

    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). `"valid"` means no
        padding. `"same"` results in padding with zeros evenly to the
        left/right or up/down of the input. When `padding="same"` and
        `strides=1`, the output has the same size as the input.

    data_format : str, optional
        Currently only supports ``"channels_last"``.

    kernel_initializer : keras.initializers.Initializer, optional
        Initializer for the `kernel` weights matrix (see
        `keras.initializers`). Defaults to `'glorot_uniform'`.

    bias_initializer : keras.initializers.Initializer, optional
        Initializer for the bias vector (see `keras.initializers`). Defaults
        to `'zeros'.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 depthwise_separable=False,
                 padding="valid",
                 data_format="channels_last",
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros"):

        super(SharpCosSim2D, self).__init__()

        kernel_size = max(1, int(kernel_size))
        if kernel_size not in [1, 3, 5]:
            raise ValueError("``kernel_size`` must be 1, 3, or 5.")
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            self.stack = lambda x: x
        elif self.kernel_size == 3:
            self.stack = self.stack3x3
        elif self.kernel_size == 5:
            self.stack = self.stack5x5

        self.filters = max(1, int(filters))
        self.strides = max(1, int(strides))
        self.depthwise_separable = bool(depthwise_separable)

        padding = str(padding).lower()
        if padding == "same":
            self.pad = self.kernel_size // 2
            self.pad_1 = 1
            self.clip = 0
        elif padding == "valid":
            self.pad = 0
            self.pad_1 = 0
            self.clip = self.kernel_size // 2
        else:
            raise ValueError("`padding` must be `'same'` or `'valid'`.")

        if data_format != "channels_last":
            raise ValueError("The `SharpCosSim2D` currently only works with "
                             "the data format `'channels_last'`.")

        self.kernel_initializer = tensorflow.keras.initializers.get(
                kernel_initializer)
        self.bias_initializer = tensorflow.keras.initializers.get(
                bias_initializer)

    @tf.function
    def l2_normal(self, x, axis=None, epsilon=K.epsilon()):
        """Compute the L2 norm, but make the result at least epsilon."""
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    @tf.function
    def l2_norm(self, x, axis=None):
        """Compute the L2 norm."""
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_norm = tf.sqrt(square_sum)
        return x_norm

    # @tf.function
    # def sigplus(self, x):
    #     """Compute the sigmoid times the softplus.

    #     This is closer to the ReLU than just a softplus.
    #     """
    #     return tf.nn.sigmoid(x) * tf.nn.softplus(x)

    @tf.function
    def stack3x3(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack([
            # Top row
            tf.pad(image[:, :y - 1 - self.clip:, :x - 1 - self.clip, :], tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]])) [:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 1 - self.clip, self.clip:x - self.clip, :], tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]])) [:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 1 - self.clip, 1 + self.clip:, :], tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]])) [:, ::self.strides, ::self.strides, :],

            # Middle row
            tf.pad(image[:, self.clip:y - self.clip, :x - 1 - self.clip, :], tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]])) [:, ::self.strides, ::self.strides, :],
            image[:, self.clip:y - self.clip:self.strides, self.clip:x - self.clip:self.strides, :],
            tf.pad(image[:, self.clip:y - self.clip, 1 + self.clip:, :], tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]])) [:, ::self.strides, ::self.strides, :],

            # Bottom row
            tf.pad(image[:, 1 + self.clip:, :x - 1 - self.clip, :], tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]])) [:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:, self.clip:x - self.clip, :], tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]])) [:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:, 1 + self.clip:, :], tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]])) [:, ::self.strides, ::self.strides, :]
        ], axis=3)

        return stack
    
    @tf.function
    def stack5x5(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack([
            # Row 0 (top row)
            tf.pad(image[:, :y - 2 - self.clip:, :x - 2 - self.clip, :], tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 2 - self.clip:, 1:x - 1 - self.clip, :], tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 2 - self.clip:, self.clip:x - self.clip, :], tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 2 - self.clip:, 1 + self.clip:-1, :], tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, :y - 2 - self.clip:, 2 + self.clip: , :], tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]]))[:, ::self.strides, ::self.strides, :],

            # Row 1
            tf.pad(image[:, 1:y - 1 - self.clip:, :x - 2 - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1:y - 1 - self.clip:, 1:x - 1 - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1:y - 1 - self.clip:, self.clip:x - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1:y - 1 - self.clip:, 1 + self.clip:-1, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1:y - 1 - self.clip:, 2 + self.clip:, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]]))[:, ::self.strides, ::self.strides, :],

            # Row 2 (center row)
            tf.pad(image[:, self.clip:y - self.clip, :x - 2 - self.clip, :], tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, self.clip:y - self.clip, 1:x - 1 - self.clip, :], tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
                   image[:, self.clip:y - self.clip, self.clip:x - self.clip, :][:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, self.clip:y - self.clip, 1 + self.clip:-1, :], tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, self.clip:y - self.clip, 2 + self.clip:, :], tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]]))[:, ::self.strides, ::self.strides, :],

            # Row 3
            tf.pad(image[:, 1 + self.clip:-1, :x - 2 - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:-1, 1:x - 1 - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:-1, self.clip:x - self.clip, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:-1, 1 + self.clip:-1, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 1 + self.clip:-1, 2 + self.clip:, :], tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]]))[:, ::self.strides, ::self.strides, :],

            # Row 4 (bottom row)
            tf.pad(image[:, 2 + self.clip:, :x - 2 - self.clip, :], tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 2 + self.clip:, 1:x - 1 - self.clip, :], tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 2 + self.clip:, self.clip:x - self.clip, :], tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 2 + self.clip:, 1 + self.clip:-1, :], tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]]))[:, ::self.strides, ::self.strides, :],
            tf.pad(image[:, 2 + self.clip:, 2 + self.clip:, :], tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]]))[:, ::self.strides, ::self.strides, :],
        ], axis=3)

        return stack

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_y = int(
            np.ceil((self.in_shape[1] - 2 * self.clip) / self.strides) + 0.5)
        self.out_x = int(
            np.ceil((self.in_shape[2] - 2 * self.clip) / self.strides) + 0.5)
        self.flat_size = self.out_x * self.out_y
        self.channels = self.in_shape[3]

        if self.depthwise_separable:
            self.w = self.add_weight(
                shape=(1,
                       self.kernel_size * self.kernel_size,
                       self.filters),
                initializer=self.kernel_initializer,
                name='w',
                trainable=True,
            )
        else:
            self.w = self.add_weight(
                shape=(1,
                       self.channels * self.kernel_size * self.kernel_size,
                       self.filters),
                initializer=self.kernel_initializer,
                name='w',
                trainable=True,
            )

        # self.b = self.add_weight(
        #     shape=(self.filters,),
        #     initializer="zeros",
        #     trainable=True,
        #     name='b')
        self.b = self.add_weight(
            shape=(self.filters,),
            initializer=self.bias_initializer,
            trainable=True,
            name='b')

        p_init = tf.keras.initializers.Constant(value=2.0)

        self.p = self.add_weight(
            shape=(self.filters,),
            initializer=p_init,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            name='p')

        noise_init = tf.keras.initializers.Constant(value=K.epsilon())

        self.y = self.add_weight(
            shape=(1,),
            initializer=noise_init,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            name='y')

        self.q = self.add_weight(
            shape=(1,),
            initializer=noise_init,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            name='q')

        # self.e = self.add_weight(
        #     shape=(1,),
        #     initializer=noise_init,
        #     trainable=True,
        #     constraint=tf.keras.constraints.NonNeg(),
        #     name='e')

        self.s = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(value=100.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            name='s')


    def call_body(self, inputs):
        channels = tf.shape(inputs)[-1]
        x = self.stack(inputs)
        x = tf.reshape(x,
                       (-1,
                        self.flat_size,
                        channels * self.kernel_size * self.kernel_size))
        x_norm = self.l2_normal(x, axis=2) + self.y
        w_norm = self.l2_normal(self.w, axis=1) + self.q
        x = tf.matmul(x / x_norm,
                      self.w / w_norm)
        # sign = tf.sign(x)
        sign = tf.tanh(self.s * x)  # "Soft sign"
        x = tf.abs(x) + K.epsilon()
        # x = tf.pow(x, self.sigplus(self.p))
        x = tf.pow(x, self.p)
        x = sign * x + self.b
        x = tf.reshape(x,
                       (-1,
                        self.out_y,
                        self.out_x,
                        self.filters))
        return x
    
    @tf.function
    def call(self, inputs, training=None):

        x = inputs

        if self.depthwise_separable:
            # channels = tf.shape(inputs)[-1]
            x = tf.vectorized_map(
                self.call_body,
                tf.expand_dims(tf.transpose(x, (3, 0, 1, 2)),
                               axis=-1))
            # s = tf.shape(x)
            x = tf.transpose(x, (1, 2, 3, 4, 0))
            x = tf.reshape(x,
                           (-1,
                            self.out_y,
                            self.out_x,
                            self.channels * self.filters))
        else:
            x = self.call_body(x)

        outputs = x

        return outputs


class MaxPoolingMask2D(tensorflow.keras.layers.Layer):
    """A max pooling layer for 2D images that only computes the pooling mask.

    Parameters
    ----------
    pool_size : int or tuple of int, length 2, optional
        Factors by which to downscale the image (vertical, horizontal). A value
        of  (2, 2) will halve the input in both spatial dimension. If only one
        integer is specified, the same window length will be used for both
        dimensions. Default is (2, 2).

    strides : int or tuple of int, length 2, optional
        Strides values. If None, it will default to ``pool_size``. This
        parameter is currently not used, instead a default stride of (2, 2)
        will be used.

    padding : str, optional
        One of ``"valid"`` or ``"same"`` (case-insensitive).

    data_format : str, optional
        One of ``"channels_last"`` (default) or ``"channels_first"``. The
        ordering of the dimensions in the inputs. ``"channels_last"``
        corresponds to inputs with shape ``(batch, height, width, channels)``
        while ``"channels_first"`` corresponds to inputs with shape ``(batch,
        channels, height, width)``. It defaults to the ``image_data_format``
        value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import nethin.layers as layers
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> inputs = Input(shape=(4, 4, 1))
    >>> mpm = layers.MaxPoolingMask2D(pool_size=(2, 2),
    ...                               strides=(2, 2),
    ...                               padding="same")
    >>> outputs = mpm(inputs)
    >>> model = Model(inputs, outputs)
    >>> X = np.random.rand(1, 4, 4, 1)
    >>> X[0, :, :, 0]
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> model.predict_on_batch(X)[0, :, :, 0]
    array([[ 1,  7],
           [12, 11]], dtype=int64)
    >>>
    >>> np.random.seed(42)
    >>> inputs = Input(shape=(1, 4, 4))
    >>> mpm = layers.MaxPoolingMask2D(pool_size=(2, 2),
    ...                               strides=(2, 2),
    ...                               padding="same",
    ...                               data_format="channels_first")
    >>> outputs = mpm(inputs)
    >>> model = Model(inputs, outputs)
    >>> X = np.random.rand(1, 1, 4, 4)
    >>> X[0, 0, :, :]
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> model.predict_on_batch(X)[0, 0, :, :]
    array([[ 1,  7],
           [12, 11]], dtype=int64)
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding="same",
                 data_format="channels_last",
                 **kwargs):

        super(MaxPoolingMask2D, self).__init__(**kwargs)

        self.pool_size = tf_keras.utils.conv_utils.normalize_tuple(
                pool_size, 2, "pool_size")
        self.strides = self.pool_size  # TODO: Allow other strides!
        self.padding = tf_keras.utils.conv_utils.normalize_padding(padding)
        self.data_format = tf_keras.utils.conv_utils.normalize_data_format(
                data_format)

        self.input_spec = tf_keras.engine.InputSpec(ndim=4)

    def build(self, input_shape):

        super(MaxPoolingMask2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):

        config = {"pool_size": self.pool_size,
                  "strides": self.strides,
                  "padding": self.padding,
                  "data_format": self.data_format}

        base_config = super(MaxPoolingMask2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def _maxpool_mask(self, inputs):

        pool_size = (1,) + self.pool_size + (1,)
        strides = (1,) + self.strides + (1,)

        if self.data_format == "channels_last":
            pass
        else:  # self.data_format == "channels_first"
            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        # TODO: Seems to only work on GPU. Keep tabs on updates and fix.
        # Also: Doesn't have gradients defined. Keep tabs on updates at:
        # https://github.com/tensorflow/tensorflow/issues/1793
        _, mask = tf.nn.max_pool_with_argmax(inputs,
                                             pool_size,
                                             strides,
                                             self.padding.upper(),
                                             Targmax=tf.int64)
        mask = tf.stop_gradient(mask, name="mask")

        if self.data_format == "channels_first":
            mask = tf.transpose(mask, (0, 3, 1, 2))

        return mask

    def call(self, inputs):

        mask = self._maxpool_mask(inputs)

        return mask


class MaxPooling2D_(tensorflow.keras.layers.Layer):
    """A max pooling layer for 2D images that also computes the pooling mask.

    Parameters
    ----------
    pool_size : int or tuple of int, length 2, optional
        Factors by which to downscale the image (vertical, horizontal). A value
        of  (2, 2) will halve the input in both spatial dimension. If only one
        integer is specified, the same window length will be used for both
        dimensions. Default is (2, 2).

    strides : int or tuple of int, length 2, optional
        Strides values. If None, it will default to ``pool_size``. This
        parameter is currently not used, instead a default stride of (2, 2)
        will be used.

    padding : str, optional
        One of ``"valid"`` or ``"same"`` (case-insensitive).

    data_format : str, optional
        One of ``"channels_last"`` (default) or ``"channels_first"``. The
        ordering of the dimensions in the inputs. ``"channels_last"``
        corresponds to inputs with shape ``(batch, height, width, channels)``
        while ``"channels_first"`` corresponds to inputs with shape ``(batch,
        channels, height, width)``. It defaults to the ``image_data_format``
        value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    compute_mask : bool, optional
        Whether or not to compute the pooling mask. If ``True``, the mask will
        be stored in ``self.mask``. Otherwise, ``self.mask`` is ``None``.
        Default is True, compute the mask.

    Attributes
    ----------
    mask : Tensor
        The positions of the generated pool that can be used to recreate an
        unpooled tensor, with the pooled values in their correct positions.
        This attribute will be ``None`` if ``compute_mask=False``

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import nethin.layers as layers
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> inputs = Input(shape=(4, 4, 1))
    >>> mp = layers.MaxPooling2D_(pool_size=(2, 2), strides=(2, 2),
    ...                           padding="same", compute_mask=True)
    >>> outputs = mp(inputs)
    >>> model = Model(inputs, outputs)
    >>> X = np.random.rand(1, 4, 4, 1)
    >>> X[0, :, :, 0]
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> model.predict_on_batch(X)[0, :, :, 0]
    array([[ 0.95071429,  0.86617613],
           [ 0.83244264,  0.96990985]], dtype=float32)
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding="same",
                 data_format="channels_last",
                 compute_mask=True,
                 **kwargs):

        super(MaxPooling2D_, self).__init__(**kwargs)

        self.pool_size = tf_keras.utils.conv_utils.normalize_tuple(
                pool_size, 2, "pool_size")
        self.strides = self.pool_size  # TODO: Allow other strides!
        self.padding = tf_keras.utils.conv_utils.normalize_padding(padding)
        self.data_format = tf_keras.utils.conv_utils.normalize_data_format(
                data_format)
        self.compute_mask_ = bool(compute_mask)

        self.mask = None

        self.input_spec = tf_keras.engine.InputSpec(ndim=4)

    def build(self, input_shape):

        super(MaxPooling2D_, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]
        else:  # self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]

        rows = tf_keras.utils.conv_utils.conv_output_length(rows,
                                                            self.pool_size[0],
                                                            self.padding,
                                                            self.strides[0])
        cols = tf_keras.utils.conv_utils.conv_output_length(cols,
                                                            self.pool_size[1],
                                                            self.padding,
                                                            self.strides[1])
        if self.data_format == "channels_last":
            return (input_shape[0], rows, cols, input_shape[3])
        else:  # self.data_format == "channels_first":
            return (input_shape[0], input_shape[1], rows, cols)

    def get_config(self):

        config = {"pool_size": self.pool_size,
                  "strides": self.strides,
                  "padding": self.padding,
                  "data_format": self.data_format,
                  "compute_mask": self.compute_mask_}

        base_config = super(MaxPooling2D_, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def _maxpool(self, inputs):

        if self.data_format == "channels_last":
            pool_size = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
            data_format = "NHWC"

        else:  # self.data_format == "channels_first"
            pool_size = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
            data_format = "NCHW"

            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        # TODO: Seems to only work on GPU. Keep tabs on updates and fix.
        # Also: Doesn't have gradients defined. Keep tabs on updates at:
        # https://github.com/tensorflow/tensorflow/issues/1793
        _, mask = tf.nn.max_pool_with_argmax(inputs,
                                             pool_size,
                                             strides,
                                             self.padding.upper(),
                                             Targmax=tf.int64)
        mask = tf.stop_gradient(mask, name="mask")

        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 3, 1, 2))

        outputs = tf.nn.max_pool(inputs,
                                 pool_size,
                                 strides,
                                 self.padding.upper(),
                                 data_format=data_format)

        return outputs, mask

    def call(self, inputs):

        if self.compute_mask_:
            outputs, mask = self._maxpool(inputs)
            self.mask = mask
        else:
            outputs = tf_keras.layers.MaxPooling2D(
                    pool_size=self.pool_size,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format)(inputs)
            self.mask = None

        return outputs


class MaxUnpooling2D(tensorflow.keras.layers.Layer):
    """A max unpooling layer for 2D images that can also use a pooling mask.

    Parameters
    ----------
    pool_size : int or tuple of int, length 2, optional
        Factors by which to downscale the image (vertical, horizontal). A value
        of (2, 2) will halve the input in both spatial dimension. If only one
        integer is specified, the same window length will be used for both
        dimensions. Default is (2, 2).

    strides : int or tuple of int, length 2, optional
        Strides values. If None, it will default to ``pool_size``. This
        parameter is currently not used, instead a default stride of ``(2, 2)``
        will be used.

    padding : str, optional
        One of ``"valid"`` or ``"same"`` (case-insensitive). This parameter is
        currently not used, instead a default ``"same"`` padding will be used.

    fill_zeros : bool, optional
        When no mask is given, ``fill_zeros`` determines whether to put the
        pooled values in the upper left corner of the upsampled unpooled
        region, or to put the same value in all cells of the upsampled unpooled
        region. Default is ``True``, put the value in the upper left corner and
        fill with zeros.

    data_format : str, optional
        One of ``"channels_last"`` (default) or ``"channels_first"``. The
        ordering of the dimensions in the inputs. ``"channels_last"``
        corresponds to inputs with shape ``(batch, height, width, channels)``
        while ``"channels_first"`` corresponds to inputs with shape ``(batch,
        channels, height, width)``. It defaults to the ``image_data_format``
        value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import nethin.layers as layers
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> inputs = Input(shape=(4, 4, 1))
    >>> mp = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
    ...                          padding="same", compute_mask=True)
    >>> outputs = mp(inputs)
    >>> model = Model(inputs, outputs)
    >>> X = np.random.rand(1, 4, 4, 1)
    >>> X[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> model.predict_on_batch(X)[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.95071429,  0.86617613],
           [ 0.83244264,  0.96990985]], dtype=float32)
    >>> mup = layers.MaxUnpooling2D(pool_size=(2, 2), strides=(2, 2),
    ...                             padding="same")
    >>> pooled = mp(inputs)
    >>> outputs2 = mup(pooled, indices=mp.mask)
    >>> model2 = Model(inputs, outputs2)
    >>> model2.predict_on_batch(X)[0, :, :, 0]
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.        , 0.95071429, 0.        , 0.        ],
           [ 0.        , 0.        , 0.        , 0.86617613],
           [ 0.        , 0.        , 0.        , 0.96990985],
           [ 0.83244264, 0.        , 0.        , 0.        ]], dtype=float32)
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=(2, 2),
                 padding="same",
                 fill_zeros=True,
                 data_format="channels_last",
                 **kwargs):

        super(MaxUnpooling2D, self).__init__(**kwargs)

        self.pool_size = tf_keras.utils.conv_utils.normalize_tuple(
                pool_size, 2, "pool_size")
        self.strides = (2, 2)
        self.padding = "same"
        self.fill_zeros = bool(fill_zeros)
        self.data_format = tf_keras.utils.conv_utils.normalize_data_format(
                data_format)

    def build(self, input_shape):

        super(MaxUnpooling2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        # If so, there are two inputs: [tensor, mask indices]
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.data_format == "channels_last":
            if input_shape[1] is not None:
                height = self.pool_size[0] * input_shape[1]
            else:
                height = None
            if input_shape[2] is not None:
                width = self.pool_size[1] * input_shape[2]
            else:
                width = None

            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

        else:  # self.data_format == "channels_first"
            if input_shape[2] is not None:
                height = self.pool_size[0] * input_shape[2]
            else:
                height = None
            if input_shape[3] is not None:
                width = self.pool_size[1] * input_shape[3]
            else:
                width = None

            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)

    def get_config(self):

        config = {"pool_size": self.pool_size,
                  "strides": self.strides,
                  "padding": self.padding,
                  "fill_zeros": self.fill_zeros,
                  "data_format": self.data_format}

        base_config = super(MaxUnpooling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def _maxunpool(self, inputs):
        """Performs unpooling without a mask.

        If self.fill_zeros is True, put the original values in the upper-left
        corner of the upsampled cells, e.g.

            maxunpool = MaxUnpooling2D(pool_size=2, fill_zeros=True)
            a = tf.Variable([[[[4]]]])
            maxunpool._maxunpool(a)[0, :, :, 0]

        returns

            [[4, 0],
             [0, 0]]

        and

            maxunpool = MaxUnpooling2D(pool_size=2, fill_zeros=False)
            a = tf.Variable([[[[4]]]])
            maxunpool._maxunpool(a)[0, :, :, 0]

        returns

            [[4, 4],
             [4, 4]].

        Adapted from `upsample` in:
            https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
        """
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        rank = tf.rank(inputs)
        pool = (1, self.pool_size[0], self.pool_size[1], 1)

        axis = 2
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0],
                                 shape[1],
                                 shape[2] * pool[axis],
                                 shape[3]])
        if self.fill_zeros:
            padding = tf.zeros(shape, dtype=inputs.dtype)
            parts = [inputs] + [padding for _ in range(pool[axis] - 1)]
        else:
            parts = [inputs] * pool[axis]
        outputs = tf.concat(parts, tf.minimum(axis + 1, rank - 1))
        outputs = tf.reshape(outputs, output_shape)

        inputs = outputs

        axis = 1
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0],
                                 shape[1] * pool[axis],
                                 shape[2],
                                 shape[3]])
        if self.fill_zeros:
            padding = tf.zeros(shape, dtype=inputs.dtype)
            parts = [inputs] + [padding for _ in range(pool[axis] - 1)]
        else:
            parts = [inputs] * pool[axis]
        outputs = tf.concat(parts, tf.minimum(axis + 1, rank - 1))
        outputs = tf.reshape(outputs, output_shape)

        if self.data_format == "channels_first":
            outputs = tf.transpose(outputs, (0, 3, 1, 2))

        return outputs

    def _maxunpool_mask(self, inputs, mask):
        """Performs unpooling with a mask.

        Adapted from `unpool` in:
            https://github.com/tensorflow/tensorflow/issues/2169
            https://github.com/souschefistry/udacity_nns/commit/a6388e7c8f00b6cbd4d035f7f0ac88760f0756a0
            https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
        """
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        if mask.dtype != tf.int64:
            mask = tf.cast(mask, tf.int64)

        input_shape = tf.shape(inputs)
        output_shape = [input_shape[0],
                        input_shape[1] * self.pool_size[0],
                        input_shape[2] * self.pool_size[1],
                        input_shape[3]]

        # flat_input_size = tf.cumprod(input_shape)[-1]
        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = tf.stack([output_shape[0],
                                      output_shape[1]
                                      * output_shape[2]
                                      * output_shape[3]])

        # pool = tf.reshape(inputs, tf.stack([flat_input_size]))
        pool = tf.reshape(inputs, [flat_input_size])
        # batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
        #                                   dtype=self.mask.dtype),
        #                          shape=tf.stack([input_shape[0], 1, 1, 1]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
                                          dtype=mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        # b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        b = tf.reshape(b, [flat_input_size, 1])
        # ind = tf.reshape(self.mask, tf.stack([flat_input_size, 1]))
        ind = tf.reshape(mask, [flat_input_size, 1])
        ind = tf.concat([b, ind], 1)

        outputs = tf.scatter_nd(ind, pool, shape=tf.cast(flat_output_shape,
                                                         tf.int64))
        # outputs = tf.reshape(outputs, tf.stack(output_shape))
        outputs = tf.reshape(outputs, output_shape)

        if self.data_format == "channels_first":
            outputs = tf.transpose(outputs, (0, 3, 1, 2))

        return outputs

    def call(self, inputs):

        if isinstance(inputs, list):
            if len(inputs) != 2:
                raise ValueError("Input must be a tensor or a list of two "
                                 "tensors.")

            return self._maxunpool_mask(*inputs)  # *[x, indices]
        else:
            return self._maxunpool(inputs)


# TODO: Remove!
class MaxUnpooling2D_(tensorflow.keras.layers.Layer):
    """A max unpooling layer for 2D images that can also use a pooling mask.

    Parameters
    ----------
    pool_size : int or tuple of int, length 2, optional
        Factors by which to downscale the image (vertical, horizontal). A value
        of (2, 2) will halve the input in both spatial dimension. If only one
        integer is specified, the same window length will be used for both
        dimensions. Default is (2, 2).

    strides : int or tuple of int, length 2, optional
        Strides values. If None, it will default to ``pool_size``. This
        parameter is currently not used, instead a default stride of (2, 2)
        will be used.

    padding : str, optional
        One of ``"valid"`` or ``"same"`` (case-insensitive). This parameter is
        currently not used, instead a default ``"same"`` padding will be used.

    mask : Tensor, optional
        The mask from a ``MaxPooling2D`` layer to use in the upsampling.
        Default is None, which means that no mask is provided, and will thus
        not be used.

    fill_zeros : bool, optional
        When no mask is given, ``fill_zeros`` determines whether to put the
        pooled values in the upper left corner of the upsampled unpooled
        region, or to put the same value in all cells of the upsampled unpooled
        region. Default is True, put the value in the upper left corner and
        fill with zeros.

    data_format : str, optional
        One of ``"channels_last"`` (default) or ``"channels_first"``. The
        ordering of the dimensions in the inputs. ``"channels_last"``
        corresponds to inputs with shape ``(batch, height, width, channels)``
        while ``"channels_first"`` corresponds to inputs with shape ``(batch,
        channels, height, width)``. It defaults to the ``image_data_format``
        value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import nethin.layers as layers
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> inputs = Input(shape=(4, 4, 1))
    >>> mp = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
    ...                          padding="same", compute_mask=True)
    >>> outputs = mp(inputs)
    >>> model = Model(inputs, outputs)
    >>> X = np.random.rand(1, 4, 4, 1)
    >>> X[0, :, :, 0]
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> model.predict_on_batch(X)[0, :, :, 0]
    array([[ 0.95071429,  0.86617613],
           [ 0.83244264,  0.96990985]], dtype=float32)
    >>> mup = layers.MaxUnpooling2D(pool_size=(2, 2), strides=(2, 2),
    ...                             padding="same", mask=mp.mask)
    >>> outputs2 = mup(mp(inputs))
    >>> model2 = Model(inputs, outputs2)
    >>> model2.predict_on_batch(X)[0, :, :, 0]
    array([[ 0.        ,  0.95071429,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.86617613],
           [ 0.        ,  0.        ,  0.        ,  0.96990985],
           [ 0.83244264,  0.        ,  0.        ,  0.        ]], dtype=float32)
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=(2, 2),
                 padding="same",
                 mask=None,
                 fill_zeros=True,
                 data_format="channels_last",
                 **kwargs):

        super(MaxUnpooling2D_, self).__init__(**kwargs)

        self.pool_size = tf_keras.utils.conv_utils.normalize_tuple(
                pool_size, 2, "pool_size")
        self.strides = (2, 2)
        self.padding = "same"
        self.mask = mask
        self.fill_zeros = bool(fill_zeros)
        self.data_format = tf_keras.utils.conv_utils.normalize_data_format(
                data_format)

    def build(self, input_shape):

        super(MaxUnpooling2D_, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":
            if input_shape[1] is not None:
                height = self.pool_size[0] * input_shape[1]
            else:
                height = None
            if input_shape[2] is not None:
                width = self.pool_size[1] * input_shape[2]
            else:
                width = None

            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

        else:  # channels_first
            if input_shape[2] is not None:
                height = self.pool_size[0] * input_shape[2]
            else:
                height = None
            if input_shape[3] is not None:
                width = self.pool_size[1] * input_shape[3]
            else:
                width = None

            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)

    def get_config(self):

        if self.mask is None:
            _mask = self.mask
        else:
            if isinstance(self.mask, six.string_types):
                _mask = self.mask
            else:
                _mask = self.mask.name

        config = {"pool_size": self.pool_size,
                  "strides": self.strides,
                  "padding": self.padding,
                  "mask": _mask,
                  "fill_zeros": self.fill_zeros,
                  "data_format": self.data_format}

        base_config = super(MaxUnpooling2D_, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def _maxunpool(self, inputs):
        """Performs unpooling without a mask.

        If self.fill_zeros is True, put the original values in the upper-left
        corner of the upsampled cells, e.g.

            maxunpool = MaxUnpooling2D(pool_size=2, fill_zeros=True)
            a = tf.Variable([[[[4]]]])
            maxunpool._maxunpool(a)[0, :, :, 0]

        returns

            [[4, 0],
             [0, 0]]

        and

            maxunpool = MaxUnpooling2D(pool_size=2, fill_zeros=False)
            a = tf.Variable([[[[4]]]])
            maxunpool._maxunpool(a)[0, :, :, 0]

        returns

            [[4, 4],
             [4, 4]].

        Adapted from `upsample` in:
            https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
        """
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        rank = tf.rank(inputs)
        pool = (1, self.pool_size[0], self.pool_size[1], 1)

        axis = 2
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0],
                                 shape[1] * pool[axis],
                                 shape[2],
                                 shape[3]])
        if self.fill_zeros:
            padding = tf.zeros(shape, dtype=inputs.dtype)
            parts = [inputs] + [padding for _ in range(pool[axis] - 1)]
        else:
            parts = [inputs] * pool[axis]
        outputs = tf.concat(parts, tf.minimum(axis + 1, rank - 1))
        outputs = tf.reshape(outputs, output_shape)

        inputs = outputs

        axis = 1
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0],
                                 shape[1],
                                 shape[2] * pool[axis],
                                 shape[3]])
        if self.fill_zeros:
            padding = tf.zeros(shape, dtype=inputs.dtype)
            parts = [inputs] + [padding for _ in range(pool[axis] - 1)]
        else:
            parts = [inputs] * pool[axis]
        outputs = tf.concat(parts, tf.minimum(axis + 1, rank - 1))
        outputs = tf.reshape(outputs, output_shape)

        if self.data_format == "channels_first":
            outputs = tf.transpose(outputs, (0, 3, 1, 2))

        return outputs

    def _maxunpool_mask(self, inputs):
        """Performs unpooling with a mask.

        Adapted from `unpool` in:
            https://github.com/tensorflow/tensorflow/issues/2169
            https://github.com/souschefistry/udacity_nns/commit/a6388e7c8f00b6cbd4d035f7f0ac88760f0756a0
            https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
        """
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (0, 2, 3, 1))

        if self.mask.dtype != tf.int64:
            self.mask = tf.cast(self.mask, tf.int64)

        input_shape = tf.shape(inputs)
        output_shape = [input_shape[0],
                        input_shape[1] * self.pool_size[0],
                        input_shape[2] * self.pool_size[1],
                        input_shape[3]]

        # flat_input_size = tf.cumprod(input_shape)[-1]
        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = tf.stack([output_shape[0],
                                      output_shape[1]
                                      * output_shape[2]
                                      * output_shape[3]])

        # pool = tf.reshape(inputs, tf.stack([flat_input_size]))
        pool = tf.reshape(inputs, [flat_input_size])
        # batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
        #                                   dtype=self.mask.dtype),
        #                          shape=tf.stack([input_shape[0], 1, 1, 1]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
                                          dtype=self.mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(self.mask) * batch_range
        # b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        b = tf.reshape(b, [flat_input_size, 1])
        # ind = tf.reshape(self.mask, tf.stack([flat_input_size, 1]))
        ind = tf.reshape(self.mask, [flat_input_size, 1])
        ind = tf.concat([b, ind], 1)

        outputs = tf.scatter_nd(ind, pool, shape=tf.cast(flat_output_shape,
                                                         tf.int64))
        # outputs = tf.reshape(outputs, tf.stack(output_shape))
        outputs = tf.reshape(outputs, output_shape)

        if self.data_format == "channels_first":
            outputs = tf.transpose(outputs, (0, 3, 1, 2))

        return outputs

    def call(self, x):
        if (self.mask is None) or isinstance(self.mask, str):
            return self._maxunpool(x)
        else:
            return self._maxunpool_mask(x)

#        cond = tf.cond((self.mask is None) or isinstance(self.mask, str),
#                       true_fn=lambda: self._maxunpool(x),
#                       false_fn=lambda: self._maxunpool_mask(x))
#
#        return cond


class MaxPooling1D(tensorflow.keras.layers.MaxPooling1D):
    """Keras' ``MaxPooling1D`` requires the input be in "channels_last" data
    format. This layer adds functionality for both "channels_last" and
    "channels_first".
    """
    def __init__(self, *args, data_format="channels_last", **kwargs):

        super(MaxPooling1D, self).__init__(*args, **kwargs)

        self._data_format = str(data_format)

    def call(self, inputs, *args, **kwargs):

        if self._data_format == "channels_first":
            inputs = tf_keras.backend.permute_dimensions(inputs, (0, 2, 1))

        outputs = super(MaxPooling1D, self).call(inputs, *args, **kwargs)

        if self._data_format == "channels_first":
            outputs = tf_keras.backend.permute_dimensions(outputs, (0, 2, 1))

        return outputs


class Convolution1D(tensorflow.keras.layers.Convolution1D):
    """Keras' ``Convolution1D`` requires the input be in "channels_last" data
    format. This layer adds functionality for both "channels_last" and
    "channels_first".
    """
    def __init__(self, *args, data_format="channels_last", **kwargs):

        super(Convolution1D, self).__init__(*args, **kwargs)

        self._data_format = str(data_format)

    def call(self, inputs, *args, **kwargs):

        if self._data_format == "channels_first":
            inputs = tf_keras.backend.permute_dimensions(inputs, (0, 2, 1))

        outputs = super(Convolution1D, self).call(inputs, *args, **kwargs)

        if self._data_format == "channels_first":
            outputs = tf_keras.backend.permute_dimensions(outputs, (0, 2, 1))

        return outputs


class Convolution2DTranspose(tensorflow.keras.layers.Convolution2DTranspose):
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
                                                           *args,
                                                           **kwargs)

        if not outputs.get_shape().is_fully_defined():

            input_shape = tf_keras.backend.int_shape(inputs)
            if self.data_format == "channels_last":
                height, width = input_shape[1], input_shape[2]
            else:
                height, width = input_shape[2], input_shape[3]
            kernel_h, kernel_w = self.kernel_size
            stride_h, stride_w = self.strides
            batch_size = inputs.get_shape()[0]

            if self.output_padding is None:
                out_pad_h = out_pad_w = None
            else:
                out_pad_h, out_pad_w = self.output_padding

            # Infer the dynamic output shape:
            # if LooseVersion(tF_keras.__version__) >= LooseVersion("2.2.4"):
            out_height = tf_keras.utils.conv_utils.deconv_output_length(
                    height,
                    kernel_h,
                    padding=self.padding,
                    output_padding=out_pad_h,
                    stride=stride_h,
                    dilation=self.dilation_rate[0])
            out_width = tf_keras.utils.conv_utils.deconv_output_length(
                    width,
                    kernel_w,
                    padding=self.padding,
                    output_padding=out_pad_w,
                    stride=stride_w,
                    dilation=self.dilation_rate[1])
            #if LooseVersion(tf_keras.__version__) >= LooseVersion("2.2.1"):
            #    # TODO: Better value for output_padding?
            #    out_height = tf_keras.utils.conv_utils.deconv_length(
            #            height,
            #            stride_h,
            #            kernel_h,
            #            self.padding,
            #            None)  # output_padding
            #    out_width = tf_keras.utils.conv_utils.deconv_length(
            #            width,
            #            stride_w,
            #            kernel_w,
            #            self.padding,
            #            None)  # output_padding
            #else:
            #    out_height = tf_keras.utils.conv_utils.deconv_length(
            #            height,
            #            stride_h,
            #            kernel_h,
            #            self.padding)
            #    out_width = tf_keras.utils.conv_utils.deconv_length(
            #            width,
            #            stride_w,
            #            kernel_w,
            #            self.padding)

            if self.data_format == "channels_last":
                output_shape = (batch_size,
                                out_height, out_width, self.filters)
            else:
                output_shape = (batch_size,
                                self.filters, out_height, out_width)

            outputs.set_shape(output_shape)

        return outputs


class Resampling2D(tensorflow.keras.layers.UpSampling2D):
    """Resampling layer for 2D inputs.

    Resizes the input images to ``size``.

    Parameters
    ----------
    size : int, or tuple of 2 integers.
        The size of the resampled image, in the format ``(rows, columns)``.

    method : Resampling2D.ResizeMethod or str, optional
        The resampling method to use. Default is None, which means
        ``Resampling2D.ResizeMethod.BILINEAR`` is used. The string name of the
        enum may also be provided.

    data_format: str
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape
        ``(batch, channels, height, width)``. It defaults to the
        ``image_data_format`` value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> import tensorflow.keras
    >>>
    >>> import nethin.layers
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 2, 2, 1))
    >>>
    >>> resize = nethin.layers.Resampling2D(
    ...         (2, 2),
    ...         data_format="channels_last")
    >>>
    >>> inputs = tensorflow.keras.layers.Input(shape=(2, 2, 1))
    >>> outputs = resize(inputs)
    >>> model = tensorflow.keras.models.Model(inputs, outputs)
    >>>
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[1.  , 1.25, 1.75, 2.  ],
           [1.25, 1.5 , 2.  , 2.25],
           [1.75, 2.  , 2.5 , 2.75],
           [2.  , 2.25, 2.75, 3.  ]], dtype=float32)
    >>>
    >>> resize = nethin.layers.Resampling2D(
    ...         (0.5, 0.5),
    ...         data_format="channels_last")
    >>>
    >>> inputs = tensorflow.keras.layers.Input(shape=(4, 4, 1))
    >>> outputs = resize(inputs)
    >>> model = tensorflow.keras.models.Model(inputs, outputs)
    >>>
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[1.25, 2.  ],
           [2.  , 2.75]], dtype=float32)
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 1, 2, 2))
    >>>
    >>> resize = nethin.layers.Resampling2D(
    ...         (2, 2),
    ...         data_format="channels_first")
    >>>
    >>> inputs = tensorflow.keras.layers.Input(shape=(1, 2, 2))
    >>> outputs = resize(inputs)
    >>> model = tensorflow.keras.models.Model(inputs, outputs)
    >>>
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[1.  , 1.25, 1.75, 2.  ],
           [1.25, 1.5 , 2.  , 2.25],
           [1.75, 2.  , 2.5 , 2.75],
           [2.  , 2.25, 2.75, 3.  ]], dtype=float32)
    >>>
    >>> resize = nethin.layers.Resampling2D(
    ...         (0.5, 0.5),
    ...         data_format="channels_first")
    >>>
    >>> inputs = tensorflow.keras.layers.Input(shape=(1, 4, 4))
    >>> outputs = resize(inputs)
    >>> model = tensorflow.keras.models.Model(inputs, outputs)
    >>>
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[1.25, 2.  ],
           [2.  , 2.75]], dtype=float32)
    """

    class ResizeMethod(Enum):
        """Enum for the resize method to use."""

        NEAREST_NEIGHBOR = "NEAREST_NEIGHBOR"  # Nearest neighbor interpolation
        BILINEAR = "BILINEAR"  # Bilinear interpolation
        BICUBIC = "BICUBIC"  # Bicubic interpolation
        AREA = "AREA"  # Area interpolation

    def __init__(self,
                 size,
                 method=None,
                 data_format=None,
                 **kwargs):

        if method is None:
            method = Resampling2D.ResizeMethod.BILINEAR

        if method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
            super(Resampling2D, self).__init__(size=size,
                                               data_format=data_format,
                                               interpolation="nearest",
                                               **kwargs)
        else:
            super(Resampling2D, self).__init__(size=size,
                                               data_format=data_format,
                                               **kwargs)

        if isinstance(method, Resampling2D.ResizeMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = Resampling2D.ResizeMethod[method]
            except KeyError:
                print(f"1 {method}")
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")
        else:
            print(f"2 {method}")
            raise ValueError("``method`` must be of type "
                             "``Resampling2D.ResizeMethod`` or one of "
                             "their string representations.")

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_first":

            if input_shape[2] is not None:
                height = int(self.size[0] * input_shape[2] + 0.5)
            else:
                height = None

            if input_shape[3] is not None:
                width = int(self.size[1] * input_shape[3] + 0.5)
            else:
                width = None

            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)

        elif self.data_format == "channels_last":

            if input_shape[1] is not None:
                height = int(self.size[0] * input_shape[1] + 0.5)
            else:
                height = None

            if input_shape[2] is not None:
                width = int(self.size[1] * input_shape[2] + 0.5)
            else:
                width = None

            print((input_shape[0],
                    height,
                    width,
                    input_shape[3]))

            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):

        import tensorflow.keras.backend as K

        if self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
            outputs = super(Resampling2D, self).call(inputs)

        else:
            if self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif self.method == Resampling2D.ResizeMethod.BILINEAR:
                method = tf.image.ResizeMethod.BILINEAR
            elif self.method == Resampling2D.ResizeMethod.BICUBIC:
                method = tf.image.ResizeMethod.BICUBIC
            elif self.method == Resampling2D.ResizeMethod.AREA:
                method = tf.image.ResizeMethod.AREA
            else:  # Should not be able to happen!
                print(method)
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")

            orig_shape = K.shape(inputs)

            if self.data_format == "channels_first":

                img_h = K.cast(orig_shape[2], K.floatx())
                img_w = K.cast(orig_shape[3], K.floatx())
                fac_h = K.constant(self.size[0], dtype=K.floatx())
                fac_w = K.constant(self.size[1], dtype=K.floatx())
                new_h = K.cast(img_h * fac_h + 0.5, "int32")
                new_w = K.cast(img_w * fac_w + 0.5, "int32")

                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                outputs = tf.image.resize(inputs,
                                          [new_h, new_w],
                                          method=method)
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

                # new_shape = K.cast(orig_shape[2:], K.floatx())
                # new_shape *= K.constant(self.size, dtype=K.floatx())
                # new_shape = tf.to_int32(new_shape + 0.5)  # !TF
                # inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                # outputs = tf.image.resize_images(inputs, new_shape,  # !TF
                #                                  method=method,
                #                                  align_corners=True)
                # outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

            elif self.data_format == "channels_last":

                img_h = K.cast(orig_shape[1], K.floatx())
                img_w = K.cast(orig_shape[2], K.floatx())
                fac_h = K.constant(self.size[0], dtype=K.floatx())
                fac_w = K.constant(self.size[1], dtype=K.floatx())
                new_h = K.cast(img_h * fac_h + 0.5, "int32")
                new_w = K.cast(img_w * fac_w + 0.5, "int32")

                outputs = tf.image.resize(inputs,
                                          [new_h, new_w],
                                          method=method)

            else:
                raise ValueError("Invalid data_format:", self.data_format)

        # input_shape = K.int_shape(inputs)
        # if self.data_format == "channels_last":
        #     output_shape = (input_shape[0],
        #                     int(input_shape[1] * self.size[0] + 0.5),
        #                     int(input_shape[2] * self.size[1] + 0.5),
        #                     input_shape[3])
        # else:
        #     output_shape = (input_shape[0],
        #                     input_shape[1],
        #                     int(input_shape[2] * self.size[0] + 0.5),
        #                     int(input_shape[3] * self.size[1] + 0.5))

        # outputs.set_shape(output_shape)

        return outputs

    def get_config(self):

        config = {"method": self.method.name}

        base_config = super(Resampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
