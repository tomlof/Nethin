"""
This module contains trainer classes for the models, that can be used to train
the models in a simple manner.

Created on Fri Oct 27 12:02:42 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
from six import with_metaclass
import gc
import time

import numpy as np

import nethin.utils as utils

__all__ = ["BasicTrainer", "CVTrainer"]


class BasicTrainer(object):
    """Basic trainer class used also as base class for other trainers.

    Assumes that generated images has the channels last data format
    ("channels_last").
    """
    def __init__(self,
                 model_factory,
                 input_channels,
                 output_channels,
                 generator_train,
                 generator_validation,
                 generator_test,
                 max_epochs=100,
                 max_iter_train=None,
                 max_iter_validation=None,
                 max_iter_test=None,
                 verbose=False):

        self.model_factory = model_factory
        self.input_channels = max(1, int(input_channels))
        self.output_channels = max(1, int(output_channels))
        self.generator_train = generator_train
        self.generator_validation = generator_validation
        self.generator_test = generator_test

        if max_epochs is not None:
            self.max_epochs = max(1, int(max_epochs))
        else:
            self.max_epochs = max_epochs

        if max_iter_train is not None:
            self.max_iter_train = max(1, int(max_iter_train))
        else:
            self.max_iter_train = max_iter_train

        if max_iter_validation is not None:
            self.max_iter_validation = max(1, int(max_iter_validation))
        else:
            self.max_iter_validation = max_iter_validation

        if max_iter_test is not None:
            self.max_iter_test = max(1, int(max_iter_test))
        else:
            self.max_iter_test = max_iter_test

        self.verbose = bool(verbose)

    def train(self):
        """Perform a full training program on the given model.
        """
        model = self.model_factory()

        restart_train = True
        restart_validation = True

        loss_train = []
        loss_validation = []
        for it in range(self.max_epochs):

            batch_it_train = 0
            loss_batch_train = []
            if restart_train:
                generator_train = self.generator_train()
                restart_train = False
            try:
                while True:
                    images = np.array(next(generator_train),
                                      dtype=np.float32)

                    inputs = images[..., :self.input_channels]
                    outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                    loss = model.train_on_batch(inputs, outputs)

                    loss_batch_train.append(loss)

                    if self.verbose:
                        print("train it: %d, batch: %d, loss: %s"
                              % (it, batch_it_train, str(loss)))

                    batch_it_train += 1

                    if self.max_iter_train is not None:
                        if batch_it_train >= self.max_iter_train:
                            break

            except (StopIteration) as e:
                restart_train = True
            loss_train.append(loss_batch_train)

            batch_it_validation = 0
            loss_batch_validation = []
            if restart_validation:
                generator_validation = self.generator_validation()
                restart_validation = False
            try:
                while True:
                    images = np.array(next(generator_validation),
                                      dtype=np.float32)

                    inputs = images[..., :self.input_channels]
                    outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                    loss = model.test_on_batch(inputs, outputs)

                    loss_batch_validation.append(loss)

                    if self.verbose:
                        print("validation it: %d, batch: %d, loss: %s"
                              % (it, batch_it_validation, str(loss)))

                    batch_it_validation += 1

                    if self.max_iter_validation is not None:
                        if batch_it_validation >= self.max_iter_validation:
                            break

            except (StopIteration) as e:
                restart_validation = True
            loss_validation.append(loss_batch_validation)

        batch_it_test = 0
        loss_test = []
        generator_test = self.generator_test()
        try:
            while True:
                images = np.array(next(generator_test), dtype=np.float32)

                inputs = images[..., :self.input_channels]
                outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                loss = model.test_on_batch(inputs, outputs)

                loss_test.append(loss)

                if self.verbose:
                    print("test batch: %d, loss: %s"
                          % (batch_it_test, str(loss)))

                batch_it_test += 1

                if self.max_iter_test is not None:
                    if batch_it_test >= self.max_iter_test:
                        break

        except (StopIteration) as e:
            pass

        return loss_train, loss_validation, loss_test


class CVTrainer(object):
    """A trainer class that uses cross-validation to estimate the validation
    error.

    Assumes that generated images has the channels last data format
    ("channels_last").
    """
    def __init__(self,
                 model_factory,
                 input_channels,
                 output_channels,
                 generator_train,
                 generator_validation,
                 max_epochs=100,
                 max_iter_train=None,
                 max_iter_validation=None,
                 verbose=False):

        self.model_factory = model_factory
        self.input_channels = max(1, int(input_channels))
        self.output_channels = max(1, int(output_channels))
        self.generator_train = generator_train
        self.generator_validation = generator_validation

        assert(len(self.generator_train) == len(self.generator_validation))

        self._cv_rounds = len(self.generator_train)

        if max_epochs is not None:
            self.max_epochs = max(1, int(max_epochs))
        else:
            self.max_epochs = max_epochs

        if max_iter_train is not None:
            self.max_iter_train = max(1, int(max_iter_train))
        else:
            self.max_iter_train = max_iter_train

        if max_iter_validation is not None:
            self.max_iter_validation = max(1, int(max_iter_validation))
        else:
            self.max_iter_validation = max_iter_validation

        self.verbose = bool(verbose)

    def train(self):
        """Perform a full training program on the given model.
        """
        loss_cv_train = []
        loss_cv_validation = []
        for cv in range(self._cv_rounds):
            model = self.model_factory()

            restart_train = True
            restart_validation = True

            loss_train = []
            loss_validation = []
            for it in range(self.max_epochs):

                batch_it_train = 0
                loss_batch_train = []
                if restart_train:
                    generator_train = self.generator_train[cv]()
                    restart_train = False
                try:
                    while True:
                        images = np.array(next(generator_train),
                                          dtype=np.float32)

                        inputs = images[..., :self.input_channels]
                        outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                        loss = model.train_on_batch(inputs, outputs)

                        loss_batch_train.append(loss)

                        if self.verbose:
                            print("train cv: %d, it: %d, batch: %d, loss: %s"
                                  % (cv, it, batch_it_train, str(loss)))

                        batch_it_train += 1

                        if self.max_iter_train is not None:
                            if batch_it_train >= self.max_iter_train:
                                break

                except (StopIteration) as e:
                    restart_train = True
                loss_train.append(loss_batch_train)

                loss_cv_train.append(loss_train)

                batch_it_validation = 0
                loss_batch_validation = []
                if restart_validation:
                    generator_validation = self.generator_validation[cv]()
                    restart_validation = False
                try:
                    while True:
                        images = np.array(next(generator_validation),
                                          dtype=np.float32)

                        inputs = images[..., :self.input_channels]
                        outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                        loss = model.test_on_batch(inputs, outputs)

                        loss_batch_validation.append(loss)

                        if self.verbose:
                            print("validation cv: %d, it: %d, batch: %d, loss: %s"
                                  % (cv, it, batch_it_validation, str(loss)))

                        batch_it_validation += 1

                        if self.max_iter_validation is not None:
                            if batch_it_validation >= self.max_iter_validation:
                                break

                except (StopIteration) as e:
                    restart_validation = True
                loss_validation.append(loss_batch_validation)

            loss_cv_validation.append(loss_validation)

            del model
            gc.collect()
            if cv < self._cv_rounds - 1:
                for i in range(4):
                    time.sleep(1)
                    gc.collect()

        return loss_cv_train, loss_cv_validation
