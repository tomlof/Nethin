# -*- coding: utf-8 -*-
"""
Contains custom or adapted normalization layers.

Created on Mon Oct  9 13:55:24 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import keras.backend as K
# K = utils.LazyImport("keras.backend")  # import keras.backend as K
import keras.engine as keras_engine
# keras_engine = utils.LazyImport("keras.engine")  # import keras.engine
import keras.engine.topology as keras_engine_topology
# keras_engine_topology = utils.LazyImport("keras.engine.topology")

__all__ = ["InstanceNormalization2D"]


class InstanceNormalization2D(keras_engine_topology.Layer):
    """Instance normalisation layer.

    Adapted from:
        https://github.com/PiscesDream/CycleGAN-keras/blob/master/CycleGAN/layers/normalization.py

    Parameters
    ----------
    axis: int
        The axis that should be normalized (typically the features axis). For
        instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.

    center : bool
        If True, add offset is added to the normalized tensor. If False, the
        offset is ignored.

    scale : bool
        If True, multiply by a scale factor. If False, the scale factor is not
        used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling will be done by the next layer.

    Examples
    --------
    >>> import parsimony.functions.neural as neural
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> from keras import optimizers
    >>> import numpy as np
    >>>
    >>> A = np.arange(12).reshape(3, 4).astype(np.float32)
    >>>
    >>> inputs = Input(shape=(3, 4, 1))
    >>> x = neural.InstanceNormalization2D(axis=3)(inputs)
    >>> model = Model(inputs=inputs, outputs=x)
    >>> B = model.predict(A.reshape(1, 3, 4, 1)).reshape(3, 4)
    >>> np.abs(np.sum(B)) < 5e-7
    True
    >>> np.abs(np.var(B) - 1.0) < 5e-7
    True
    >>>
    >>> inputs = Input(shape=(1, 3, 4))
    >>> x = neural.InstanceNormalization2D(axis=1)(inputs)
    >>> model = Model(inputs=inputs, outputs=x)
    >>> B = model.predict(A.reshape(1, 1, 3, 4)).reshape(3, 4)
    >>> np.abs(np.sum(B)) < 5e-7
    True
    >>> np.abs(np.var(B) - 1.0) < 5e-7
    True
    """
    def __init__(self, axis=-1, center=True, scale=True, **kwargs):

        super(InstanceNormalization2D, self).__init__(**kwargs)

        self.axis = int(axis)
        self.center = bool(center)
        self.scale = bool(scale)

        self.input_spec = keras_engine.InputSpec(ndim=4)

    def build(self, input_shape):

        if self.center:
            self.beta = self.add_weight(name="beta",
                                        shape=(input_shape[self.axis],),
                                        initializer="zero",
                                        trainable=True)
        else:
            self.beta = None

        if self.scale:
            self.gamma = self.add_weight(name="gamma",
                                         shape=(input_shape[self.axis],),
                                         initializer="one",
                                         trainable=True)
        else:
            self.gamma = None

        super(InstanceNormalization2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):

        config = {"axis": self.axis,
                  "center": self.center,
                  "scale": self.scale}
        base_config = super(InstanceNormalization2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(tensor, -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        if self.axis == 3:
            hw = K.cast(x.shape[1] * x.shape[2], K.floatx())
            mu_vec = K.sum(x, axis=[1, 2], keepdims=True) / hw
            sig2 = K.sum(K.square(x - mu_vec), axis=[1, 2], keepdims=True) / hw

            gamma = K.expand_dims(K.expand_dims(K.expand_dims(self.gamma, 0),
                                                0),
                                  0)
            beta = K.expand_dims(K.expand_dims(K.expand_dims(self.beta, 0),
                                               0),
                                 0)

        else:  # self.axis == 1  # "channels_first"
            hw = K.cast(x.shape[2] * x.shape[3], K.floatx())
            mu_vec = K.sum(x, axis=[2, 3], keepdims=True) / hw
            sig2 = K.sum(K.square(x - mu_vec), axis=[2, 3], keepdims=True) / hw

            gamma = K.expand_dims(K.expand_dims(K.expand_dims(self.gamma, 0),
                                                -1),
                                  -1)
            beta = K.expand_dims(K.expand_dims(K.expand_dims(self.beta, 0),
                                               -1),
                                 -1)

        y = (x - mu_vec) / (K.sqrt(sig2) + K.epsilon())

        return gamma * y + beta
