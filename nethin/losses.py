# -*- coding: utf-8 -*-
"""
This module contains custom loss functions.

Created on Mon Nov  6 13:50:57 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import keras.backend as K

__all__ = ["BaseLoss",
           "GradientDifferenceLoss"]


class BaseLoss(object):

    def __init__(self, name):

        self.__name__ = str(name)

    def __call__(self, y_true, y_pred):

        return self.call(y_true, y_pred)


class GradientDifferenceLoss(BaseLoss):

    def __init__(self, input_shape):

        super(GradientDifferenceLoss, self).__init__("gradient_difference_loss")

        self.input_shape = tuple(input_shape)

        assert(len(self.input_shape) == 3 or len(self.input_shape) == 4)

    def call(self, y_true, y_pred):

        if len(self.input_shape) == 3:
            dif11 = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
            dif12 = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
            dif13 = None
            dif21 = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
            dif22 = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
            dif23 = None
        elif len(self.input_shape) == 4:
            dif11 = y_true[:, 1:, :, :, :] - y_true[:, :-1, :, :, :]
            dif12 = y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]
            dif13 = y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :]
            dif21 = y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]
            dif22 = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
            dif23 = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]

        dif1 = K.sqrt(K.sum(K.square(dif11))) - K.sqrt(K.sum(K.square(dif21)))
        gdl = K.square(dif1)
        dif2 = K.sqrt(K.sum(K.square(dif12))) - K.sqrt(K.sum(K.square(dif22)))
        gdl = gdl + K.square(dif2)
        if dif13 is not None:
            dif3 = K.sqrt(K.sum(K.square(dif13))) - K.sqrt(K.sum(K.square(dif23)))
            gdl = gdl + K.square(dif3)

        return gdl
