# -*- coding: utf-8 -*-
"""
Contains custom or adapted Keras layers.

Created on Thu Oct 12 14:35:08 2017

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from enum import Enum

import tensorflow as tf

import keras.backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.engine.topology import Layer
import keras.layers.convolutional as convolutional
from keras.layers.pooling import MaxPooling2D as keras_MaxPooling2D

__all__ = ["MaxPooling2D", "MaxUnpooling2D",
           "Convolution2DTranspose", "Resampling2D"]


class MaxPooling2D(Layer):
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
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding="same",
                 data_format="channels_last",
                 compute_mask=True,
                 **kwargs):

        super(MaxPooling2D, self).__init__(**kwargs)

        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = self.pool_size  # TODO: Allow other strides!
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.compute_mask_ = bool(compute_mask)

        self.mask = None

        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):

        super(MaxPooling2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]
        else:  # self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]

        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
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

        base_config = super(MaxPooling2D, self).get_config()

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
            outputs = keras_MaxPooling2D(pool_size=self.pool_size,
                                         strides=self.strides,
                                         padding=self.padding,
                                         data_format=self.data_format)(inputs)
            self.mask = None

        return outputs


class MaxUnpooling2D(Layer):
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

        super(MaxUnpooling2D, self).__init__(**kwargs)

        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = (2, 2)
        self.padding = "same"
        self.mask = mask
        self.fill_zeros = bool(fill_zeros)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):

        super(MaxUnpooling2D, self).build(input_shape)

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

        config = {"pool_size": self.pool_size,
                  "strides": self.strides,
                  "padding": self.padding,
                  "fill_zeros": self.fill_zeros,
                  "data_format": self.data_format,
                  "mask": None if self.mask is None else self.mask.name}

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


class Resampling2D(convolutional.UpSampling2D):
    """Resampling layer for 2D inputs.

    Resizes the input images to ``size``.

    Parameters
    ----------
    size : int, or tuple of 2 integers.
        The size of the resampled image, in the format ``(rows, columns)``.

    method : Resampling2D.ResizeMethod or str, optional
        The resampling method to use. Default is
        ``Resampling2D.ResizeMethod.BILINEAR``. The string name of the enum
        may also be provided.

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
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> import nethin.layers as layers
    >>> np.random.seed(42)
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 2, 2, 1))
    >>> resize = layers.Resampling2D((2, 2), data_format="channels_last")
    >>> inputs = Input(shape=(2, 2, 1))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1. ,  1.5,  2. ,  2. ],
           [ 1.5,  2. ,  2.5,  2.5],
           [ 2. ,  2.5,  3. ,  3. ],
           [ 2. ,  2.5,  3. ,  3. ]], dtype=float32)
    >>> resize = layers.Resampling2D((0.5, 0.5), data_format="channels_last")
    >>> inputs = Input(shape=(4, 4, 1))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 2.,  3.]], dtype=float32)
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 1, 2, 2))
    >>> resize = layers.Resampling2D((2, 2), data_format="channels_first")
    >>> inputs = Input(shape=(1, 2, 2))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1. ,  1.5,  2. ,  2. ],
           [ 1.5,  2. ,  2.5,  2.5],
           [ 2. ,  2.5,  3. ,  3. ],
           [ 2. ,  2.5,  3. ,  3. ]], dtype=float32)
    >>> resize = layers.Resampling2D((0.5, 0.5), data_format="channels_first")
    >>> inputs = Input(shape=(1, 4, 4))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 2.,  3.]], dtype=float32)
    """
    class ResizeMethod(Enum):
        BILINEAR = "BILINEAR"  # Bilinear interpolation
        NEAREST_NEIGHBOR = "NEAREST_NEIGHBOR"  # Nearest neighbor interpolation
        BICUBIC = "BICUBIC"  # Bicubic interpolation
        AREA = "AREA"  # Area interpolation

    def __init__(self,
                 size,
                 method=ResizeMethod.BILINEAR,
                 data_format=None,
                 **kwargs):

        super(Resampling2D, self).__init__(size=size,
                                           data_format=data_format,
                                           **kwargs)

        if isinstance(method, Resampling2D.ResizeMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = Resampling2D.ResizeMethod[method]
            except KeyError:
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")
        else:
            raise ValueError("``method`` must be of type "
                             "``Resampling2D.ResizeMethod`` or one of "
                             "their string representations.")

    def call(self, inputs):

        if self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
            return super(Resampling2D, self).call(inputs)

        else:
            if self.method == Resampling2D.ResizeMethod.BILINEAR:
                method = tf.image.ResizeMethod.BILINEAR
            elif self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif self.method == Resampling2D.ResizeMethod.BICUBIC:
                method = tf.image.ResizeMethod.BICUBIC
            elif self.method == Resampling2D.ResizeMethod.AREA:
                method = tf.image.ResizeMethod.AREA
            else:  # Should not be able to happen!
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")

            if self.data_format == "channels_first":
                original_shape = K.int_shape(inputs)
                new_shape = K.cast(K.shape(inputs)[2:], K.floatx())
                new_shape *= K.constant(self.size, dtype=K.floatx())
                new_shape = tf.to_int32(new_shape + 0.5)  # !TF
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                inputs = tf.image.resize_images(inputs, new_shape,  # !TF
                                                method=method)
                inputs = K.permute_dimensions(inputs, [0, 3, 1, 2])
                inputs.set_shape((None,
                                  None,
                                  int((original_shape[2] * self.size[0]) + 0.5) if original_shape[2] is not None else None,
                                  int((original_shape[3] * self.size[1]) + 0.5) if original_shape[3] is not None else None))
            elif self.data_format == "channels_last":
                original_shape = K.int_shape(inputs)
                new_shape = K.cast(K.shape(inputs)[1:3], K.floatx())
                new_shape *= K.constant(self.size, dtype=K.floatx())
                new_shape = tf.to_int32(new_shape + 0.5)  # !TF
                inputs = tf.image.resize_images(inputs, new_shape,  # !TF
                                                method=method)
                inputs.set_shape((None,
                                  int((original_shape[1] * self.size[0]) + 0.5) if original_shape[1] is not None else None,
                                  int((original_shape[2] * self.size[1]) + 0.5) if original_shape[2] is not None else None,
                                  None))
            else:
                raise ValueError("Invalid data_format:", self.data_format)

        return inputs

    def get_config(self):

        config = {"method": self.method.name}

        base_config = super(Resampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
