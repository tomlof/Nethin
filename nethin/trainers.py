"""
This module contains trainer classes for the models, that can be used to train
the models in a simple manner.

Created on Fri Oct 27 12:02:42 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import os
import abc
from six import with_metaclass
import gc
import time
import pickle

import numpy as np

from tensorflow.python.framework.errors_impl import ResourceExhaustedError

import keras.backend as K

import nethin.utils as utils

__all__ = ["BaseTraner", "BasicTrainer", "CVTrainer"]


class BaseTraner(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError('Abstract method "train" has not been '
                                  'specialised.')

    def _get_checkpoint_name(self):
        filename = "checkpoint_%s" \
                % (time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()),)

        return filename


class BasicTrainer(BaseTraner):
    """Basic trainer class used also as base class for other trainers.

    Assumes that generated images has the channels last data format
    ("channels_last").
    """
    def __init__(self,
                 model_factory,
                 # input_channels,
                 # output_channels,
                 generator_train=None,
                 generator_validation=None,
                 generator_test=None,
                 max_epochs=100,
                 max_iter_train=None,
                 max_iter_validation=None,
                 max_iter_test=None,
                 verbose=False,
                 save_best=False):

        self.model_factory = model_factory
        # self.input_channels = max(1, int(input_channels))
        # self.output_channels = max(1, int(output_channels))
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
        self.save_best = bool(save_best)

        self.model_ = None

    def _save_every(self, it, save_path,
                    pickle_ending=".pkl", model_ending=".h5"):
        filename = "epoch_%d_%s" % (it,
                                    self._get_checkpoint_name())
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, filename)

        save_list = [self.loss_train_]
        if self.generator_validation is not None:
            save_list.append(self.loss_validation_)

        with open(filename + pickle_ending, "wb") as fp:
            pickle.dump(save_list, fp)

        self.model_.save(filename + model_ending)

    def train(self, keep_training=False,
              save_every=None, save_path=None):
        """Perform a full training program on the given model.

        Parameters
        ----------
        keep_training : bool, optional, default: False
            Whether or not to continue training using the last trained model.
            If there is no previous model, a new one will be created and
            trained. Default is False, create a new model to train.

        save_every : int, optional, default: None
            When not None, will save the model and current loss every
            ``save_every`` epoch.

        save_path : str, optional, default: None
            When ``save_every`` is not None, the files will be stored in the
            directory specified by ``save_path``. If ``save_path`` is None, the
            files will be saved in the current working directory. The files
            will have a checkpoint name, prefixed by "_epoch_%d.pkl" (loss)
            and "_epoch_%d.h5" (model), where "%d" is the epoch number.
        """
        if (self.model_ is None) or (not keep_training):
            self.model_ = self.model_factory()

        save_every = int(save_every) if save_every is not None else None
        save_path = str(save_path) if save_path is not None else None

        restart_train = True
        restart_validation = True

        self.loss_train_ = []
        self.loss_validation_ = []
        best_loss = np.inf
        for it in range(self.max_epochs):

            # Save untrained model
            if (save_every is not None) and (it == 0):
                self._save_every(it, save_path)

            batch_it_train = 0
            loss_batch_train = []
            if restart_train:
                generator_train = self.generator_train()
                restart_train = False
            try:
                while True:
                    inputs, outputs = next(generator_train)
                    inputs = np.array(inputs, dtype=np.float32)
                    outputs = np.array(outputs, dtype=np.float32)

                    # inputs = images[..., :self.input_channels]
                    # outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                    loss = self.model_.train_on_batch(inputs, outputs)

                    loss_batch_train.append(loss)

                    if self.verbose:
                        if self.max_iter_train is None:
                            print("train it: %d/%d, batch: %d, loss: %s"
                                  % (it + 1, self.max_epochs,
                                     batch_it_train, str(loss)))
                        else:
                            print("train it: %d/%d, batch: %d/%d, loss: %s"
                                  % (it + 1, self.max_epochs,
                                     batch_it_train, self.max_iter_train,
                                     str(loss)))

                    batch_it_train += 1

                    if self.max_iter_train is not None:
                        if batch_it_train >= self.max_iter_train:
                            break

            except (StopIteration) as e:
                restart_train = True
            self.loss_train_.append(loss_batch_train)

            if self.generator_validation is not None:

                batch_it_validation = 0
                loss_batch_validation = []
                if restart_validation:
                    generator_validation = self.generator_validation()
                    restart_validation = False
                try:
                    while True:
                        inputs, outputs = next(generator_validation)
                        inputs = np.array(inputs, dtype=np.float32)
                        outputs = np.array(outputs, dtype=np.float32)

                        # inputs = images[..., :self.input_channels]
                        # outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                        loss = self.model_.test_on_batch(inputs, outputs)

                        loss_batch_validation.append(loss)

                        if self.verbose:
                            if self.max_iter_validation is None:
                                print("validation it: %d/%d, batch: %d, loss: %s"
                                      % (it + 1, self.max_epochs,
                                         batch_it_validation, str(loss)))
                            else:
                                print("validation it: %d/%d, batch: %d/%d, loss: %s"
                                      % (it + 1, self.max_epochs,
                                         batch_it_validation, self.max_iter_validation,
                                         str(loss)))

                        batch_it_validation += 1

                        if self.max_iter_validation is not None:
                            if batch_it_validation >= self.max_iter_validation:
                                break

                except (StopIteration) as e:
                    restart_validation = True

                self.loss_validation_.append(loss_batch_validation)

                if self.save_best:
                    try:
                        float(loss_batch_validation[0][0])
                        convertable = True
                    except:
                        convertable = False

                    if convertable and hasattr(self.model_, "save"):
                        _loss = []
                        for l in loss_batch_validation:
                            _loss.append(l[0])
                        _loss = np.mean(_loss)
                        if _loss < best_loss:
                            best_loss = _loss

                            filename = self._get_checkpoint_name() \
                                + "_%f.h5" % (float(_loss),)  # TODO: Constant
                            self.model_.save(filename)

            # Save the model and losses every save_every epochs
            if save_every is not None:
                if (it > 0) and ((it + 1) % save_every == 0):
                    self._save_every(it + 1, save_path)

        if self.generator_test is not None:
            batch_it_test = 0
            self.loss_test_ = []
            generator_test = self.generator_test()
            try:
                while True:
                    inputs, outputs = next(generator_test)
                    inputs = np.array(inputs, dtype=np.float32)
                    outputs = np.array(outputs, dtype=np.float32)

                    # inputs = images[..., :self.input_channels]
                    # outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                    loss = self.model_.test_on_batch(inputs, outputs)

                    self.loss_test_.append(loss)

                    if self.verbose:
                        if self.max_iter_test is None:
                            print("test batch: %d, loss: %s"
                                  % (batch_it_test, str(loss)))
                        else:
                            print("test batch: %d/%d, loss: %s"
                                  % (batch_it_test, self.max_iter_test,
                                     str(loss)))

                    batch_it_test += 1

                    if self.max_iter_test is not None:
                        if batch_it_test >= self.max_iter_test:
                            break

            except (StopIteration) as e:
                pass

        ret_list = [self.loss_train_]
        if self.generator_validation is not None:
            ret_list.append(self.loss_validation_)
        if self.generator_test is not None:
            ret_list.append(self.loss_test_)

        return tuple(ret_list)


class CVTrainer(BaseTraner):
    """A trainer class that uses cross-validation to estimate the validation
    error.

    Assumes that generated images has the channels last data format
    ("channels_last").
    """
    def __init__(self,
                 model_factory,
                 # input_channels,
                 # output_channels,
                 generator_train,
                 generator_validation,
                 max_epochs=100,
                 max_iter_train=None,
                 max_iter_validation=None,
                 verbose=False,
                 save_intermediate=False,
                 threshold=None):

        self._name = "CVTrainer"

        self.model_factory = model_factory
        # self.input_channels = max(1, int(input_channels))
        # self.output_channels = max(1, int(output_channels))
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
        self.save_intermediate = bool(save_intermediate)

        self.threshold = threshold

        self.model_ = None

    def train(self, keep_training=False):
        """Perform a full training program on the given model.

        Parameters
        ----------
        keep_training : bool, optional, default: False
            Whether or not to continue training using the last trained model.
            If there is no previous model, a new one will be created and
            trained. Default is False, create a new model to train.
        """
        self.loss_train_ = []
        self.loss_validation_ = []
        for cv in range(self._cv_rounds):

            self.model_ = None
            for i in range(10):
                try:
                    self.model_ = self.model_factory()
                    break
                except (ResourceExhaustedError) as e:
                    try:
                        del self.model_
                        K.clear_session()
                    except Exception:
                        pass
                    if cv > 0:
                        gc.collect()
                        time.sleep(1)
                    else:
                        raise e

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
                        inputs, outputs = next(generator_train)
                        inputs = np.array(inputs, dtype=np.float32)
                        outputs = np.array(outputs, dtype=np.float32)

                        # inputs = images[..., :self.input_channels]
                        # outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                        loss = self.model_.train_on_batch(inputs, outputs)

                        loss_batch_train.append(loss)

                        if self.verbose:
                            if self.max_iter_train is None:
                                print("train cv: %d/%d, "
                                      "it: %d/%d, "
                                      "batch: %d, "
                                      "loss: %s"
                                      % (cv + 1, self._cv_rounds,
                                         it + 1, self.max_epochs,
                                         batch_it_train,
                                         str(loss)))
                            else:
                                print("train cv: %d/%d, "
                                      "it: %d/%d, "
                                      "batch: %d/%d, "
                                      "loss: %s"
                                      % (cv + 1, self._cv_rounds,
                                         it + 1, self.max_epochs,
                                         batch_it_train, self.max_iter_train,
                                         str(loss)))

                        batch_it_train += 1

                        if self.max_iter_train is not None:
                            if batch_it_train >= self.max_iter_train:
                                break

                        if self.threshold is not None:
                            if "cv" in self.threshold:
                                __cv = self.threshold["cv"]
                            else:
                                __cv = lambda x: True
                            if "it" in self.threshold:
                                __it = self.threshold["it"]
                            else:
                                __it = lambda x: True
                            if "batch" in self.threshold:
                                __batch = self.threshold["batch"]
                            else:
                                __batch = lambda x: True
                            if "loss" in self.threshold:
                                __loss = self.threshold["loss"]
                            else:
                                __loss = lambda x: True
                            if "action" in self.threshold:
                                __action = self.threshold["action"]
                            else:
                                __action = lambda: None

                            if __cv(cv) and __it(it) and \
                                    __batch(batch_it_train - 1) and \
                                    __loss(loss):
                                __action()

                except (StopIteration) as e:
                    restart_train = True
                loss_train.append(loss_batch_train)

                self.loss_train_.append(loss_train)

                batch_it_validation = 0
                loss_batch_validation = []
                if restart_validation:
                    generator_validation = self.generator_validation[cv]()
                    restart_validation = False
                try:
                    while True:
                        inputs, outputs = next(generator_validation)
                        inputs = np.array(inputs, dtype=np.float32)
                        outputs = np.array(outputs, dtype=np.float32)

                        # inputs = images[..., :self.input_channels]
                        # outputs = images[..., self.input_channels:self.input_channels + self.output_channels]

                        loss = self.model_.test_on_batch(inputs, outputs)

                        loss_batch_validation.append(loss)

                        if self.verbose:
                            if self.max_iter_validation is None:
                                print("validation cv: %d/%d, "
                                      "it: %d/%d, "
                                      "batch: %d, "
                                      "loss: %s"
                                      % (cv + 1, self._cv_rounds,
                                         it + 1, self.max_epochs,
                                         batch_it_validation,
                                         str(loss)))
                            else:
                                print("validation cv: %d/%d, "
                                      "it: %d/%d, "
                                      "batch: %d/%d, "
                                      "loss: %s"
                                      % (cv + 1, self._cv_rounds,
                                         it + 1, self.max_epochs,
                                         batch_it_validation,
                                         self.max_iter_validation,
                                         str(loss)))

                        batch_it_validation += 1

                        if self.max_iter_validation is not None:
                            if batch_it_validation >= self.max_iter_validation:
                                break

                except (StopIteration) as e:
                    restart_validation = True
                loss_validation.append(loss_batch_validation)

            self.loss_validation_.append(loss_validation)

            if self.save_intermediate:
                with open("%s_intermediate_%d.pkl"
                          % (self._name, cv), "wb") as fp:
                    pickle.dump((self.loss_train_,
                                 self.loss_validation_),
                                fp)

            # If there are more CV rounds left, free memory used by the model.
            if cv < self._cv_rounds - 1:
                self.model_ = None
                del self.model_
                K.clear_session()
                gc.collect()
                for i in range(5):
                    time.sleep(1)
                    gc.collect()

        return self.loss_train_, self.loss_validation_
