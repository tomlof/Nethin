# -*- coding: utf-8 -*-
"""
This module contains penalties for activations, weights and biases.

Created on Mon Nov  6 13:43:46 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""




class TotalVariation2D(BasePenalty):
    """Corresponds to the total variation (TV) penalty for 2D images.

    Parameters
    ----------
    name : str
        The name of the loss. Must be unique. Should only contain alpha-numeric
        characters.

    gamma : float
        Non-negative float. The regularisation constant for the penalty.

    mean : bool
        Whether or not to compute the mean total variation over the batches or
        the sum of them. Default is True, compute the mean.

    data_format : str, optional
        The format of the input data. Either "NHWC" for shapes (num_samples,
        rows, columns, channels) or "NCHW" for shapes (num_samples, channels,
        rows, columns). Default is "NHWC", i.e. with the channels last.
    """
    def __init__(self, name, gamma, mean=True, data_format="NHWC"):

        super(TotalVariation2D, self).__init__(name, trainable=False,
                                               assignable=False)

        self.gamma = max(0.0, float(gamma))
        self.mean = bool(mean)
        self.data_format = TensorflowHelper.check_data_format(data_format)

        self._outputs = None

    def generate(self, inputs):
        """Generates a node in the computational graph for this loss.

        Parameters
        ----------
        inputs : tensor or list of tensor, length 1
            The tensor that connects to this loss.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError("Layer accepts only a single input!")
        inputs = inputs[0]

        with tf.name_scope(self.name):

            if self.data_format == "NCHW":
                inputs = tf.transpose(inputs, (0, 2, 3, 1))

#            outputs = tf.reduce_sum(tf.image.total_variation(inputs))

            dif1 = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
            dif2 = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]

            edge1 = dif1[:, :, -1:, :]
            edge2 = dif2[:, -1:, :, :]

            dif1 = dif1[:, :, :-1, :]
            dif2 = dif2[:, :-1, :, :]

            if self.mean:
                outputs = tf.reduce_sum(
                              tf.sqrt(tf.square(dif1) + tf.square(dif2)),
                              axis=[1, 2, 3]
                          ) \
                        + tf.reduce_sum(tf.abs(edge1), axis=[1, 2, 3]) \
                        + tf.reduce_sum(tf.abs(edge2), axis=[1, 2, 3])
                outputs = tf.reduce_mean(outputs)
            else:
                outputs = tf.reduce_sum(
                              tf.sqrt(tf.square(dif1) + tf.square(dif2))
                          ) \
                        + tf.reduce_sum(tf.abs(edge1)) \
                        + tf.reduce_sum(tf.abs(edge2))

            # TODO: Check if gamma is non-zero before computing anything?
            outputs = self.gamma * outputs

            self._outputs = outputs

        return self._outputs

    def get_shape(self, input_shape=None):
        """Returns the shape of the node output.

        Parameters
        ----------
        input_shape : list of tuple of int, optional
            The shape of the input Tensor to this node.

        Returns
        -------
        output_shape : tuple of int
            The shape of the node's output.
        """
        if input_shape is not None:
            if isinstance(input_shape, list):  # Then a list of a tuple
                if isinstance(input_shape[0], tuple) \
                        and (len(input_shape) == 1):

                    input_shape = input_shape[0]
                else:
                    raise ValueError("The input_shape is not a list with a "
                                     "single tuple.")

        output_shape = tuple()  # Scalar tensor output

        return output_shape