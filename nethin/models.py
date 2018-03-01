# -*- coding: utf-8 -*-
"""
This module contains ready-to-use Keras-like and Keras-compatible models.

Created on Mon Oct  9 13:48:25 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import json
import six
import warnings

import h5py
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.utils import conv_utils
from keras.initializers import TruncatedNormal
from keras.engine.topology import get_source_inputs
from keras.layers.merge import Add
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.convolutional import Convolution2DTranspose
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, Activation, Dropout, Dense, Flatten, Lambda
from keras.layers import Concatenate, Add
from keras.optimizers import Optimizer, Adam, RMSprop
from keras.engine import Layer
import keras.optimizers as optimizers
import keras.activations
import keras.initializers
import keras.losses as losses

from nethin.utils import with_device, Helper, to_snake_case
import nethin.consts as consts
import nethin.utils as utils
import nethin.padding as padding
import nethin.layers as layers
import nethin.losses as losses

__all__ = ["BaseModel",
           "ConvolutionalNetwork", "FullyConvolutionalNetwork", "UNet", "GAN"]


#def load_model(filepath, custom_objects=None, compile=True):
#    """Loads a model saved via ``save_model``.
#
#    Parameters
#    ----------
#    filepath : str
#        The path to the saved config file.
#
#    # Arguments
#        filepath: String, path to the saved model.
#        custom_objects: Optional dictionary mapping names
#            (strings) to custom classes or functions to be
#            considered during deserialization.
#        compile: Boolean, whether to compile the model
#            after loading.
#    # Returns
#        A Keras model instance. If an optimizer was found
#        as part of the saved model, the model is already
#        compiled. Otherwise, the model is uncompiled and
#        a warning will be displayed. When `compile` is set
#        to False, the compilation is omitted without any
#        warning.
#    # Raises
#        ImportError: if h5py is not available.
#        ValueError: In case of an invalid savefile.
#    """


class BaseModel(six.with_metaclass(abc.ABCMeta, object)):
    # TODO: Add fit_generator, evaluate_generator and predict_generator
    # TODO: What about get_layer?
    def __init__(self, nethin_name, data_format=None, device=None, name=None):

        self._nethin_name = str(nethin_name)

        self.data_format = conv_utils.normalize_data_format(data_format)

        if device is not None:
            device = str(device)

            supp_devices = Helper.get_devices()
            if device not in supp_devices:
                raise ValueError("Device %s not supported. The supported "
                                 "devices are: %s" % (device, supp_devices))
        self.device = device

        if name is not None:
            name = str(name)
        self.name = name

        self.model = None

    def _with_device(self, function, *args, **kwargs):
        """Call the given function (with provided arguments) on the provided
        device name, or on the default device (if ``self.device`` is ``None``).

        Use ``nethin.utils.with_device`` if you need more control of which
        device runs which code.

        Parameters
        ----------
        function
            The function or class to run/construct.

        args : list
            The list of arguments to ``function``.

        kwargs : list
            The list of keyword arguments to ``function``.
        """
        return with_device(self.device, function, *args, **kwargs)

    @abc.abstractmethod
    def _generate_model(self):
        raise NotImplementedError('"_generate_model" has not been '
                                  'specialised.')

    def __call__(self, inputs):

        return self.model(inputs)

    def get_model(self):

        return self.model

    def get_config(self):
        """Returns the config of the model.

        A model config is a Python dictionary (serializable) containing the
        configuration of a model. The same model can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a model does not include connectivity information, nor
        the model class name. These are handled by `Container` (one layer of
        abstraction above).

        Returns
        -------
        config : dict
            A Python dictionary containing the model configuration.
        """
        config = {"data_format": self.data_format,
                  "device": self.device,
                  "name": self.name}

        return config

    def save(self, filepath, overwrite=True, include_optimizer=True):
        """Save the model to a single HDF5 file.

        The savefile includes:
            - The model architecture, allowing to re-instantiate the model.
            - The model weights.
            - The state of the optimizer, allowing to resume training
              exactly where you left off.

        This allows you to save the entirety of the state of a model
        in a single file.

        Saved models can be reinstantiated via `keras.models.load_model`.
        # TODO: Update when load_model added to Nethin!

        The model returned by `load_model` is a compiled model ready to be used
        (unless the saved model was never compiled in the first place).

        Parameters
        ----------
        filepath : str
            Path to the file to save the weights to.

        overwrite : bool, optional
            Whether to silently overwrite any existing file at the target
            location, or provide the user with a manual prompt.

        include_optimizer : bool, optional
            If True, save optimizer's state together.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> from keras.layers import Input, Dense
        >>> from keras.models import Model
        >>> from keras.models import load_model
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = np.random.rand(1, 3, 1)
        >>> failed = False
        >>> try:
        ...     fd, fn = tempfile.mkstemp(prefix="nethin_example_",
        ...                               suffix=".h5")
        ...     os.close(fd)
        ... except:
        ...     try:
        ...         os.remove(fn)
        ...     except:
        ...         failed = True
        >>>
        >>> if not failed:
        ...     inputs = Input(shape=(3, 1))
        ...     outputs = Dense(1)(inputs)
        ...     model = Model(inputs, outputs)
        ...     model.compile("Adam", loss="MSE")
        ...     yhat = model.predict_on_batch(X)
        ...     np.round(yhat, 6)
        array([[[-0.446014],
                [-1.132141],
                [-0.871682]]], dtype=float32)
        >>> if not failed:
        ...     model.save(fn)  # Creates a temporary HDF5 file
        ...     del model  # Deletes the existing model
        ...     # Returns a compiled model identical to the previous one
        ...     model = load_model(fn)
        ...     yhat = model.predict_on_batch(X)
        ...     np.round(yhat, 6)
        array([[[-0.446014],
                [-1.132141],
                [-0.871682]]], dtype=float32)
        >>> if not failed:
        ...     try:
        ...         os.remove(fn)
        ...     except:
        ...         pass
        """
        self.model.save(filepath,
                        overwrite=overwrite,
                        include_optimizer=include_optimizer)

    def save_weights(self, filepath, overwrite=True):
        """Dumps all layer weights to a HDF5 file.

        The weight file has:
            - ``layer_names`` (attribute), a list of strings (ordered names of
              model layers).
            - For every layer, a ``group`` named ``layer.name``
                - For every such layer group, a group attribute `weight_names`,
                  a list of strings (ordered names of weights tensor of the
                  layer).
                - For every weight in the layer, a dataset storing the weight
                  value, named after the weight tensor.

        Parameters
        ----------
        filepath : str
            Path to the file to save the weights to.

        overwrite : bool, optional
            Whether to silently overwrite any existing file at the target
            location, or provide the user with a manual prompt.

        Raises
        ------
        ImportError
            If h5py is not available.
        """
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath, by_name=False):
        """Loads all layer weights from a HDF5 save file.

        If ``by_name=False`` (default) weights are loaded based on the
        network's topology, meaning the architecture should be the same as when
        the weights were saved. Note that layers that don't have weights are
        not taken into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers only if they share
        the same name. This is useful for fine-tuning or transfer-learning
        models where some of the layers have changed.

        Parameters
        ----------
        filepath : str
            Path to the weights file to load.

        by_name : bool, optional
            Whether to load weights by name or by topological order.

        Raises
        ------
        ImportError
            If h5py is not available.
        """
        self.model.load_weights(filepath, by_name=by_name)

    def get_weights(self):
        """Retrieves the weights of the model.

        Returns
        -------
        A flat list of Numpy arrays.
        """
        return self.model.get_weights()

    def compile(self,
                optimizer,
                loss,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None):
        """Configures the model for training.

        Parameters
        ----------
        optimizer : str or keras.optimizers.Optimizer
            String (name of optimizer) or optimizer object. See
            `optimizers <https://keras.io/optimizers>`_.

        loss : str
            String (name of objective function) or objective function. See
            `losses <https://keras.io/losses>`_. If the model has multiple
            outputs, you can use a different loss on each output by passing a
            dictionary or a list of losses. The loss value that will be
            minimized by the model will then be the sum of all individual
            losses.

         metrics : list of str, optional
             List of metrics to be evaluated by the model during training and
             testing. Typically you will use ``metrics=["accuracy"]``. To
             specify different metrics for different outputs of a multi-output
             model, you could also pass a dictionary, such as
             ``metrics={"output_a": "accuracy"}``.

        loss_weights : list or dict, optional
            Optional list or dictionary specifying scalar coefficients (Python
            floats) to weight the loss contributions of different model
            outputs. The loss value that will be minimized by the model will
            then be the weighted sum of all individual losses, weighted by the
            ``loss_weights`` coefficients. If a list, it is expected to have a
            1:1 mapping to the model's outputs. If a tensor, it is expected to
            map output names (strings) to scalar coefficients.

        sample_weight_mode : None, str, list or dict, optional
            If you need to do timestep-wise sample weighting (2D weights), set
            this to ``"temporal"``. ``None`` defaults to sample-wise weights
            (1D). If the model has multiple outputs, you can use a different
            ``sample_weight_mode`` on each output by passing a dictionary or a
            list of modes.

        weighted_metrics : list, optional
            List of metrics to be evaluated and weighted by ``sample_weight``
            or ``class_weight`` during training and testing.

        target_tensors : Tensor, optional
            By default, Keras will create placeholders for the model's target,
            which will be fed with the target data during training. If instead
            you would like to use your own target tensors (in turn, Keras will
            not expect external Numpy data for these targets at training time),
            you can specify them via the ``target_tensors`` argument. It can be
            a single tensor (for a single-output model), a list of tensors, or
            a dict mapping output names to target tensors.

        **kwargs
            When using the Theano/CNTK backends, these arguments are passed
            into ``K.function``. When using the TensorFlow backend, these
            arguments are passed into ``tf.Session.run``.

        Raises
        ------
        ValueError
            In case of invalid arguments for ``optimizer``, ``loss``,
            ``metrics`` or ``sample_weight_mode``.
        """
        if (weighted_metrics is None) and (target_tensors is None):
            # Recent additions to compile may not be available.
            self._with_device(self.model.compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode)
        else:
            self._with_device(self.model.compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode,
                              weighted_metrics=weighted_metrics,
                              target_tensors=target_tensors)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None):
        """Trains the model for a fixed number of epochs (iterations on a
        dataset).

        Parameters
        ----------
        x : numpy.ndarray or list of numpy.ndarray, optional
            Numpy array of training data, or list of Numpy arrays if the model
            has multiple inputs. If all inputs in the model are named, you can
            also pass a dictionary mapping input names to Numpy arrays. Can be
            ``None`` (default) if feeding from framework-native tensors.

        y : numpy.ndarray or list of numpy.ndarray, optional
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.
            Can be ``None`` (default) if feeding from framework-native tensors.

        batch_size : int or None, optional
            Integer or ``None``. Number of samples per gradient update. If
            unspecified, it will default to 32.

        epochs : int, optional
            The number of times to iterate over the training data arrays.

        verbose : int, optional
            One of 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one
            log line per epoch.

        callbacks : list of keras.callbacks.Callback
            List of callbacks to be called during training. See
            `callbacks <https://keras.io/callbacks>`_.

        validation_split : float, optional
            Float between 0 and 1. The fraction of the training data to be used
            as validation data. The model will set apart this fraction of the
            training data, will not train on it, and will evaluate the loss and
            any model metrics on this data at the end of each epoch.

        validation_data : tuple, optional
            Data on which to evaluate the loss and any model metrics at the end
            of each epoch. The model will not be trained on this data. This
            could be a tuple ``(x_val, y_val)`` or a tuple ``(x_val, y_val,
            val_sample_weights)``.

        shuffle : bool, optional
            Boolean, whether to shuffle the training data before each epoch.
            Has no effect when ``steps_per_epoch`` is not None.

        class_weight : dict, optional
            Optional dictionary mapping class indices (integers) to a weight
            (float) to apply to the model's loss for the samples from this
            class during training. This can be useful to tell the model to
            "pay more attention" to samples from an under-represented class.

        sample_weight : numpy.ndarray, optional
            Optional array of the same length as ``x``, containing weights to
            apply to the model's loss for each sample. In the case of temporal
            data, you can pass a 2D array with shape (samples,
            sequence_length), to apply a different weight to every timestep of
            every sample. In this case you should make sure to specify
            ``sample_weight_mode="temporal"`` in ``compile()``.

        initial_epoch : int, optional
            Epoch at which to start training (useful for resuming a previous
            training run).

        steps_per_epoch : int, optional
            Total number of steps (batches of samples) before declaring one
            epoch finished and starting the next epoch. When training with
            Input Tensors such as TensorFlow data tensors, the default ``None``
            is equal to the number of unique samples in your dataset divided by
            the batch size, or 1 if that cannot be determined.

        validation_steps : int, optional
            Only relevant if ``steps_per_epoch`` is specified. Total number of
            steps (batches of samples) to validate before stopping.

        Returns
        -------
        A ``History`` instance. Its ``history`` attribute contains all
        information collected during training.

        Raises
        ------
        ValueError
            In case of mismatch between the provided input data and what the
            model expects.
        """
        return self._with_device(self.model.fit,
                                 x=x,
                                 y=y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_split=validation_split,
                                 validation_data=validation_data,
                                 shuffle=shuffle,
                                 class_weight=class_weight,
                                 sample_weight=sample_weight,
                                 initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        Arguments
        ---------
        x : numpy.ndarray or dict of numpy.ndarray, optional
            Numpy array of test data, or list of Numpy arrays if the model has
            multiple inputs. If all inputs in the model are named, you can also
            pass a dictionary mapping input names to Numpy arrays. Can be
            ``None`` (default) if feeding from framework-native tensors.

        y : numpy.ndarray, optional
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.
            Can be ``None`` (default) if feeding from framework-native tensors.

        batch_size : int, optional
            If unspecified, it will default to 32.

        verbose : int, optional
            Verbosity mode, 0 or 1.

        sample_weight : numpy.ndarray, optional
            Array of weights to weight the contribution of different samples to
            the loss and metrics.

        steps : int, optional
            Total number of steps (batches of samples) before declaring the
            evaluation round finished. Ignored with the default value of
            ``None``.

        Returns
        -------
        Scalar test loss (if the model has a single output and no metrics) or
        list of scalars (if the model has multiple outputs and/or metrics). The
        attribute ``model.metrics_names`` will give you the display labels for
        the scalar outputs.
        """
        return self._with_device(self.model.evaluate,
                                 x=x,
                                 y=y,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 sample_weight=sample_weight,
                                 steps=steps)

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        Arguments
        ---------
        x : numpy.ndarray or list of numpy.ndarray
            The input data, as a Numpy array (or list of Numpy arrays if the
            model has multiple outputs).

        batch_size : int, optional
            If unspecified, it will default to 32.

        verbose : int, optional
            Verbosity mode, 0 or 1.

        steps : int, optional
            Total number of steps (batches of samples) before declaring the
            prediction round finished. Ignored with the default value of
            ``None``.

        Returns
        -------
        Numpy array(s) of predictions.

        Raises
        ------
        ValueError
            In case of mismatch between the provided input data and the model's
            expectations, or in case a stateful model receives a number of
            samples that is not a multiple of the batch size.
        """
        return self._with_device(self.model.predict,
                                 x,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps)

    def train_on_batch(self,
                       x,
                       y,
                       sample_weight=None,
                       class_weight=None):
        """Runs a single gradient update on a single batch of data.

        Arguments
        ---------
        x : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of training data, or list of Numpy arrays if the model
            has multiple inputs. If all inputs in the model are named, you can
            also pass a dictionary mapping input names to Numpy arrays.

        y : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.

        sample_weight : numpy.ndarray, optional
            Optional array of the same length as ``x``, containing weights to
            apply to the model's loss for each sample. In the case of temporal
            data, you can pass a 2D array with shape (samples,
            sequence_length), to apply a different weight to every timestep of
            every sample. In this case you should make sure to specify
            ``sample_weight_mode="temporal"`` in ``compile()``.

        class_weight : dict, optional
            Optional dictionary mapping class indices (integers) to a weight
            (float) to apply to the model's loss for the samples from this
            class during training. This can be useful to tell the model to
            "pay more attention" to samples from an under-represented class.

        Returns
        -------
        Scalar training loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs and/or metrics).
        The attribute ``model.metrics_names`` will give you the display labels
        for the scalar outputs.
        """
        return self._with_device(self.model.train_on_batch,
                                 x,
                                 y,
                                 sample_weight=sample_weight,
                                 class_weight=class_weight)

    def test_on_batch(self, x, y,
                      sample_weight=None):
        """Test the model on a single batch of samples.

        Arguments
        ---------
        x : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of test data, or list of Numpy arrays if the model has
            multiple inputs. If all inputs in the model are named, you can also
            pass a dictionary mapping input names to Numpy arrays.

        y : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.

        sample_weight : numpy.ndarray, optional
            Optional array of the same length as ``x``, containing weights to
            apply to the model's loss for each sample. In the case of temporal
            data, you can pass a 2D array with shape (samples,
            sequence_length), to apply a different weight to every timestep of
            every sample. In this case you should make sure to specify
            ``sample_weight_mode="temporal"`` in ``compile()``.

        Returns
        -------
        Scalar test loss (if the model has a single output and no metrics) or
        list of scalars (if the model has multiple outputs and/or metrics). The
        attribute model.metrics_names will give you the display labels for the
        scalar outputs.
        """
        return self._with_device(self.model.test_on_batch,
                                 x,
                                 y,
                                 sample_weight=sample_weight)

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Arguments
        ---------
        x : numpy.ndarray
            Input samples, as a Numpy array.

        Returns
        -------
        Numpy array(s) of predictions.
        """
        return self._with_device(self.model.predict_on_batch,
                                 x)


class ConvolutionalNetwork(BaseModel):
    """Generates a standard ConvNet architechture with Conv + BN + ReLU
    + Maxpool layers.

    Parameters
    ----------
    input_shape : tuple of ints, length 3
        The shape of the input data, excluding the batch dimension.

    num_filters : tuple of int, optional
        The number of convolution filters in each convolution layer. Default is
        ``(64, 64, 64, 64, 64)``, i.e. with five layers.

    filter_sizes : tuple of int, optional
        The size of the convolution filters in each convolution layer. Must
        have the same length as ``num_filters``. Default is ``(3, 3, 3, 3,
        3)``, i.e. with five layers.

    subsample : tuple of bool, optional
        Whether or not to perform subsampling at each of the layers. Must have
        the same length as ``num_filters``. Default is ``(True, True, True,
        True, True)``, which means to subsample after each convolution layer's
        activation function.

    activations : str or tuple of str or Activation, optional
        The activations to use in each subsampling path. If a single str or
        Layer, the same activation will be used for all layers. If a tuple,
        then it must have the same length as ``num_conv_layers``. Default
        is "relu".

    dense_sizes : tuple of int, optional
        The number of dense layers following the convolution. Default is (1,),
        which means to output a single value.

    dense_activations : str or Activation, optional
        The activation to use for the dense layers and the network output.
        Default is "sigmoid".

    use_batch_normalization : bool, optional
        Whether or not to use batch normalization after each convolution in the
        encoding part. Default is True, use batch normalization.

    use_dropout : bool, optional
        Whether or not to use dropout on the dense layers. Default is True, use
        dropout.

    dropout_rate : float, optional
        Float in [0, 1]. The dropout rate, if dropout is used. Default is 0.5.

    use_maxpool : bool, optional
        Whether or not to use maxpooling instead of strided convolutions when
        subsampling. Default is True, use maxpooling.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".

    device : str, optional
        A particular device to run the model on. Default is ``None``, which
        means to run on the default device (usually "/gpu:0"). Use
        ``nethin.utils.Helper.get_device()`` to see available devices.

    name : str, optional
        The name of the network. Default is "ConvolutionalNetwork".

    Examples
    --------
    >>> import nethin
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import tensorflow as tf
    >>> tf.set_random_seed(42)
    >>>
    >>> input_shape = (128, 128, 1)
    >>> num_filters = (16, 16)
    >>> filter_sizes = (3, 3)
    >>> subsample = (True, True)
    >>> activations = ("relu", "relu")
    >>> dense_sizes = (1,)
    >>> dense_activations = "sigmoid"
    >>> use_batch_normalization = True
    >>> use_dropout = True
    >>> dropout_rate = 0.5
    >>> use_maxpool = True
    >>> data_format = "channels_last"
    >>> device = "/device:CPU:0"
    >>>
    >>> X = np.random.randn(*input_shape)
    >>> X = X[np.newaxis, ...]
    >>>
    >>> model = nethin.models.ConvolutionalNetwork(input_shape=input_shape,
    ...                                            num_filters=num_filters,
    ...                                            filter_sizes=filter_sizes,
    ...                                            subsample=subsample,
    ...                                            activations=activations,
    ...                                            dense_sizes=dense_sizes,
    ...                                            dense_activations=dense_activations,
    ...                                            use_batch_normalization=use_batch_normalization,
    ...                                            use_dropout=use_dropout,
    ...                                            dropout_rate=dropout_rate,
    ...                                            use_maxpool=use_maxpool,
    ...                                            data_format=data_format,
    ...                                            device=device)
    >>> model.input_shape
    (128, 128, 1)
    >>> X.shape
    (1, 128, 128, 1)
    >>> model.compile(optimizer="Adam", loss="mse")
    >>> Y = model.predict_on_batch(X)
    >>> Y.shape
    (1, 1)
    >>> np.abs(np.sum(Y - X) - 9986) < 1.0
    True
    """
    def __init__(self,
                 input_shape,
                 num_filters=(64, 64, 64, 64, 64),
                 filter_sizes=(3, 3, 3, 3, 3),
                 subsample=(True, True, True, True, True),
                 activations="relu",
                 dense_sizes=(1,),
                 dense_activations=("sigmoid",),
                 use_batch_normalization=True,
                 use_dropout=True,
                 dropout_rate=0.5,
                 use_maxpool=True,
                 data_format=None,
                 device=None,
                 name="ConvolutionalNetwork"):

        super(ConvolutionalNetwork, self).__init__(
                "nethin.models.ConvolutionalNetwork",
                data_format=data_format,
                device=device,
                name=name)

        self.input_shape = tuple([int(s) for s in input_shape])

        if len(num_filters) < 1:
            raise ValueError("``num_conv_layers`` must have length at least "
                             "1.")
        else:
            self.num_filters = tuple(num_filters)

        self.filter_sizes = tuple(filter_sizes)
        assert(len(self.num_filters) == len(self.filter_sizes))

        self.subsample = tuple([bool(s) for s in subsample])
        assert(len(self.num_filters) == len(self.subsample))

        if isinstance(activations, str) or activations is None:

            activations = [self._with_device(Activation, activations)
                           for i in range(len(self.num_filters))]
            self.activations = tuple(activations)

        if isinstance(activations, Activation):

            raise ValueError("May not be able to reuse the activation for "
                             "each layer. Pass a tuple with the activations "
                             "for each layer instead.")

        elif len(activations) != len(self.num_filters):
            raise ValueError("``activations`` should have the same length as "
                             "``num_conv_layers``.")

        elif isinstance(activations, (list, tuple)):

            _activations = [None] * len(activations)
            for i in range(len(activations)):
                if isinstance(activations[i], str) or activations[i] is None:
                    _activations[i] = self._with_device(Activation, activations[i])
                else:
                    _activations[i] = activations[i]

            self.activations = tuple(_activations)

        else:
            raise ValueError("``activations`` must be a str, or a tuple of "
                             "str or ``Activation``.")

        self.dense_sizes = tuple([int(s) for s in dense_sizes])

        if isinstance(dense_activations, str):
            _activations = [self._with_device(Activation, dense_activations)
                            for i in range(len(self.dense_sizes))]
            self.dense_activations = tuple(_activations)

        elif len(dense_activations) != len(self.dense_sizes):
            raise ValueError("``dense_activations`` should have the same "
                             "length as `dense_sizes``.")

        elif isinstance(dense_activations, (list, tuple)):

            _activations = [None] * len(dense_activations)
            for i in range(len(dense_activations)):
                _activations[i] = self._with_device(Activation,
                                                    dense_activations[i])
            self.dense_activations = tuple(_activations)

        else:
            raise ValueError("``dense_activations`` must be a str, or a tuple "
                             "of str or ``Activation``.")

        self.use_batch_normalization = bool(use_batch_normalization)

        if isinstance(use_dropout, bool):
            do = [bool(use_dropout) for i in range(len(self.dense_sizes))]
            self.use_dropout = tuple(do)

        elif len(use_dropout) != len(self.dense_sizes):
            raise ValueError("``use_dropout`` should have the same length as "
                             "`dense_sizes``.")

        elif isinstance(use_dropout, (list, tuple)):

            do = [bool(use_dropout[i]) for i in range(len(use_dropout))]
            self.use_dropout = tuple(do)

        else:
            raise ValueError("``use_dropout`` must be a str, or a tuple of "
                             "str or ``Activation``.")

        self.dropout_rate = max(0.0, min(float(dropout_rate), 1.0))

        self.use_maxpool = bool(use_maxpool)

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        self.model = self._with_device(self._generate_model)

    def _generate_model(self):

        inputs = Input(shape=self.input_shape)
        x = inputs

        # Build the network
        for i in range(len(self.num_filters)):

            num_filters_i = self.num_filters[i]
            filter_size_i = self.filter_sizes[i]
            activation_i = self.activations[i]
            subsample_i = self.subsample[i]

            x = Convolution2D(num_filters_i,
                              (filter_size_i, filter_size_i),
                              strides=(1, 1),
                              padding="same",
                              data_format=self.data_format)(x)
            if self.use_batch_normalization:
                x = BatchNormalization(axis=self._axis)(x)
            x = activation_i(x)

            if subsample_i:
                if self.use_maxpool:
                    x = MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding="same",
                                     data_format=self.data_format)(x)
                else:
                    x = Convolution2D(num_filters_i,
                                      (3, 3),
                                      strides=(2, 2),
                                      padding="same",
                                      data_format=self.data_format)(x)

        if len(self.dense_sizes) > 0:
            x = Flatten()(x)

            for i in range(len(self.dense_sizes)):
                size_i = self.dense_sizes[i]
                activation_i = self.dense_activations[i]
                use_dropout_i = self.use_dropout[i]

                x = Dense(size_i)(x)
                x = activation_i(x)

                if use_dropout_i:
                    x = Dropout(self.dropout_rate)(x)

        outputs = x

        # Create model
        model = Model(inputs, outputs, name=self.name)

        return model


class FullyConvolutionalNetwork(BaseModel):
    """Generates a standard FCN architechture with Conv + BN + ReLU layers.

    Parameters
    ----------
    input_shape : tuple of ints, length 3
        The shape of the input data, excluding the batch dimension.

    output_channels : int
        The number of output channels of the network.

    num_filters : tuple of int, optional
        The number of convolution filters in each convolution layer. Default is
        ``(64, 64, 64, 64, 64)``, i.e. with five layers.

    filter_sizes : tuple of int, optional
        The size of the convolution filters in each convolution layer. Must
        have the same length as ``num_filters``. Default is ``(3, 3, 3, 3,
        3)``, i.e. with five layers.

    activations : str or tuple of str or Activation, optional
        The activations to use in each subsampling path. If a single str or
        Layer, the same activation will be used for all layers. If a tuple,
        then it must have the same length as ``num_conv_layers``. Default
        is "relu".

    use_batch_normalization : bool or tuple of bool, optional
        Whether or not to use batch normalization after each convolution in the
        encoding part. Default is True, use batch normalization for all layers.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".

    device : str, optional
        A particular device to run the model on. Default is ``None``, which
        means to run on the default device (usually "/gpu:0"). Use
        ``nethin.utils.Helper.get_device()`` to see available devices.

    name : str, optional
        The name of the network. Default is "FullyConvolutionalNetwork".

    Examples
    --------
    >>> import nethin
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import tensorflow as tf
    >>> tf.set_random_seed(42)
    >>>
    >>> input_shape = (128, 128, 1)
    >>> output_channels = 1
    >>> num_filters = (16, 16)
    >>> filter_sizes = (3, 3)
    >>> activations = ("relu", "relu")
    >>> use_batch_normalization = True
    >>> data_format = "channels_last"
    >>> device = "/device:CPU:0"
    >>>
    >>> X = np.random.randn(*input_shape)
    >>> X = X[np.newaxis, ...]
    >>>
    >>> model = nethin.models.FullyConvolutionalNetwork(input_shape=input_shape,
    ...                                                 output_channels=output_channels,
    ...                                                 num_filters=num_filters,
    ...                                                 filter_sizes=filter_sizes,
    ...                                                 activations=activations,
    ...                                                 use_batch_normalization=use_batch_normalization,
    ...                                                 data_format=data_format,
    ...                                                 device=device)
    >>> model.input_shape
    (128, 128, 1)
    >>> model.output_channels
    1
    >>> X.shape
    (1, 128, 128, 1)
    >>> model.compile(optimizer="Adam", loss="mse")
    >>> Y = model.predict_on_batch(X)
    >>> Y.shape
    (1, 128, 128, 1)
    >>> np.abs(np.sum(Y - X) - 2487) < 1.0
    True
    """
    def __init__(self,
                 input_shape,
                 output_channels,
                 num_filters=(64, 64, 64, 64, 64),
                 filter_sizes=(3, 3, 3, 3, 3),
                 activations="relu",
                 use_batch_normalization=True,
                 data_format=None,
                 device=None,
                 name="FullyConvolutionalNetwork"):

        super(FullyConvolutionalNetwork, self).__init__(
                "nethin.models.FullyConvolutionalNetwork",
                data_format=data_format,
                device=device,
                name=name)

        self.input_shape = tuple([int(s) for s in input_shape])
        self.output_channels = int(output_channels)

        if len(num_filters) < 1:
            raise ValueError("``num_filters`` must have length at least 1.")
        else:
            self.num_filters = tuple(num_filters)

        self.filter_sizes = tuple(filter_sizes)

        assert(len(self.num_filters) == len(self.filter_sizes))

        if isinstance(activations, str) or activations is None:

            activations = [self._with_device(Activation, activations)
                           for i in range(len(self.num_filters))]
            self.activations = tuple(activations)

        if isinstance(activations, Activation):

            raise ValueError("May not be able to reuse the activation for "
                             "each layer. Pass a tuple with the activations "
                             "for each layer instead.")

        elif len(activations) != len(self.num_filters):
            raise ValueError("``activations`` should have the same length as "
                             "``num_conv_layers``.")

        elif isinstance(activations, (list, tuple)):

            _activations = [None] * len(activations)
            for i in range(len(activations)):
                if isinstance(activations[i], str) or activations[i] is None:
                    _activations[i] = self._with_device(Activation, activations[i])
                else:
                    _activations[i] = activations[i]

            self.activations = tuple(_activations)

        else:
            raise ValueError("``activations`` must be a str, or a tuple of "
                             "str or ``Activation``.")

        if isinstance(use_batch_normalization, bool):
            use_batch_normalization = [use_batch_normalization] \
                    * len(self.num_filters)
            self.use_batch_normalization = tuple(use_batch_normalization)

        elif len(use_batch_normalization) != len(self.num_filters):
            raise ValueError("``use_batch_normalization`` should have the "
                             "same length as ``num_filters``.")

        elif isinstance(use_batch_normalization, (list, tuple)):

            use_bn = [None] * len(use_batch_normalization)
            for i in range(len(use_batch_normalization)):
                use_bn[i] = bool(use_batch_normalization[i])
            self.use_batch_normalization = tuple(use_bn)

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        self.model = self._with_device(self._generate_model)

    def _generate_model(self):

        inputs = Input(shape=self.input_shape)
        x = inputs

        # Build the network
        for i in range(len(self.num_filters)):

            num_filters = self.num_filters[i]
            filter_size = self.filter_sizes[i]
            activation_function = self.activations[i]
            use_batch_normalization = self.use_batch_normalization[i]

            x = Convolution2D(num_filters,
                              (filter_size, filter_size),
                              strides=(1, 1),
                              padding="same",
                              data_format=self.data_format)(x)
            if use_batch_normalization:
                x = BatchNormalization(axis=self._axis)(x)
            x = activation_function(x)

        # Final convolution to generate the requested output number of channels
        x = Convolution2D(self.output_channels,
                          (1, 1),  # Filter size
                          strides=(1, 1),
                          padding="same",
                          data_format=self.data_format)(x)
        outputs = x

        # Create model
        model = Model(inputs, outputs, name=self.name)

        return model


class UNet(BaseModel):
    """Generates the U-Net model by Ronneberger et al. (2015) [1]_.

    Parameters
    ----------
    input_shape : tuple of ints, length 3
        The shape of the input data, excluding the batch dimension.

    output_channels : int
        The number of output channels of the network.

    num_conv_layers : tuple of int, optional
        Must be of length at least one. The number of chained convolutions
        for each subsampling path. Default is ``(2, 2, 2, 2, 1)``, as was used
        by Ronneberger et al. (2015).

    num_filters : tuple of int, optional
        The number of convolution filters in each subsampling path. Must have
        the same length as ``num_conv_layers``. Default is ``(64, 128, 256,
        512, 1024)``, as was used by Ronneberger et al. (2015).

    filter_sizes : int or tuple of int, optional
        The filter size to use in each subsampling path. If a single int, the
        same filter size will be used in all layers. If a tuple, then it must
        have the same length as ``num_conv_layers``. Default is 3, as was used
        by Ronneberger et al. (2015).

    activations : str or tuple of str or Activation, optional
        The activations to use in each subsampling path. If a single str or
        Layer, the same activation will be used for all layers. If a tuple,
        then it must have the same length as ``num_conv_layers``. Default
        is "relu".

    use_downconvolution : bool, optional
        If True, uses downsampling followed by a 2x2 convolution for spatial
        downsampling. Default is False, which means to not use downconvolution.

    use_upconvolution : bool, optional
        The use of upsampling followed by a convolution can reduce artifacts
        that appear when e.g. deconvolutions with ``stride > 1`` are used.
        Default is True, which means to use upsampling followed by a 2x2
        convolution (denoted "upconvolution" in [1]_).

    use_strided_convolution : bool, optional
        If True, uses strided convolution for downsampling. Default is False,
        which means to not use strided convolutions.

    use_strided_deconvolution : bool, optional
        If True, uses strided deconvolution for upsampling. Default is False,
        which means to not use strided deconvolutions.

    use_maxpooling : bool, optional
        If True, uses 2x2 maxpooling for downsampling. Default is True, which
        means to use maxpooling for downsampling.

    use_maxunpooling : bool, optional
        If True, uses 2x2 maxunpooling for upsampling. Default is False, which
        means to not use unpooling.

    use_maxunpooling_mask : bool, optional
        If ``use_maxunpooling=True`` and  ``use_maxunpooling_mask=True``, then
        the maxunpooling will use the sites of the maximum values from the
        maxpooling operation during the unpooling. Default is False, do not use
        the mask during unpooling.

    use_deconvolutions : bool, optional
        Use deconvolutions (transposed convolutinos) in the deconding part
        instead of convolutions. This should have only small practical effect,
        since deconvolutions are also convolutions, but may be preferred in
        some cases. Default is False, do not use deconvolutions in the decoding
        part.

    use_batch_normalization : bool, optional
        Whether or not to use batch normalization after each convolution in the
        encoding part. Default is False, do not use batch normalization.

    kernel_initializer : str, keras.Initializer or Callable, optional
        The initializer to use for all layers. Default is ``"glorot_uniform"``.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".

    device : str, optional
        A particular device to run the model on. Default is ``None``, which
        means to run on the default device (usually the first GPU device). Use
        ``nethin.utils.Helper.get_device()`` to see available devices.

    name : str, optional
        The name of the network. Default is "UNet".

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer and T. Brox (2015). "U-Net: Convolutional
       Networks for Biomedical Image Segmentation". arXiv:1505.04597v1 [cs.CV],
       available at: https://arxiv.org/abs/1505.04597.

    Examples
    --------
    >>> import nethin
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import tensorflow as tf
    >>> tf.set_random_seed(42)
    >>>
    >>> input_shape = (128, 128, 1)
    >>> output_shape = (128, 128, 1)
    >>> num_conv_layers = (2, 1)
    >>> num_filters = (32, 64)
    >>> filter_sizes = 3
    >>> activations = "relu"
    >>> use_upconvolution = True
    >>> use_deconvolutions = True
    >>> data_format = None
    >>>
    >>> X = np.random.randn(*input_shape)
    >>> X = X[np.newaxis, ...]
    >>>
    >>> model = nethin.models.UNet(input_shape=input_shape,
    ...                            output_channels=output_shape[-1],
    ...                            num_conv_layers=num_conv_layers,
    ...                            num_filters=num_filters,
    ...                            filter_sizes=filter_sizes,
    ...                            activations=activations,
    ...                            use_upconvolution=use_upconvolution,
    ...                            use_deconvolutions=use_deconvolutions,
    ...                            data_format=data_format)
    >>> model.input_shape
    (128, 128, 1)
    >>> model.output_channels
    1
    >>> model.compile(optimizer="Adam", loss="mse")
    >>> X.shape
    (1, 128, 128, 1)
    >>> Y = model.predict_on_batch(X)
    >>> Y.shape
    (1, 128, 128, 1)
    >>> np.abs(np.sum(Y - X) - 1500.0) < 5.0
    True
    """
    def __init__(self,
                 input_shape,
                 output_channels=1,
                 num_conv_layers=[2, 2, 2, 2, 1],
                 num_filters=[64, 128, 256, 512, 1024],
                 filter_sizes=3,
                 activations="relu",
                 use_downconvolution=False,
                 use_upconvolution=True,
                 use_strided_convolution=False,
                 use_strided_deconvolution=False,
                 use_maxpooling=True,
                 use_maxunpooling=False,
                 use_maxunpooling_mask=False,
                 use_deconvolutions=False,
                 use_batch_normalization=False,
                 kernel_initializer="glorot_uniform",
                 data_format=None,
                 device=None,
                 name="UNet",
                 _model=None):

        super(UNet, self).__init__("nethin.models.UNet",
                                   data_format=data_format,
                                   device=device,
                                   name=name)

#        if input_tensor is None and input_shape is None:
#            raise ValueError("Both input_shape and input_tensor can not be "
#                             "None. ")
#        else:
#            if input_tensor is not None:
#                if not K.is_keras_tensor(input_tensor):
#                    raise ValueError('"input_tensor" is not a Keras tensor.')
#            self.input_tensor = input_tensor
#            if input_shape is not None:
#                if len(input_shape) != 3:
#                    raise ValueError('"input_shape" must be a tuple of '
#                                     'length 3.')
#            self.input_shape = tuple([int(s) for s in input_shape])

        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError('"input_shape" must be a tuple of length 3.')
        self.input_shape = tuple([int(s) for s in input_shape])

        self.output_channels = int(output_channels)

        if len(num_conv_layers) < 1:
            raise ValueError("``num_conv_layers`` must have length at least "
                             "1.")
        else:
            self.num_conv_layers = tuple(num_conv_layers)

        self.num_filters = tuple([int(nf) for nf in num_filters])

        if isinstance(filter_sizes, int):
            self.filter_sizes = (filter_sizes,) * len(self.num_conv_layers)
        elif len(filter_sizes) != len(self.num_conv_layers):
                raise ValueError("``filter_sizes`` should have the same "
                                 "length as ``num_conv_layers``.")
        else:
            self.filter_sizes = tuple([int(fs) for fs in filter_sizes])

        self._activations_orig = activations
        if isinstance(activations, six.string_types):
            activations_enc = [self._with_device(Activation,
                                                 activations)
                               for i in range(len(num_conv_layers))]
            self.activations_enc = tuple(activations_enc)

            activations_dec = [self._with_device(Activation,
                                                 activations)
                               for i in range(len(num_conv_layers))]
            self.activations_dec = tuple(activations_dec)

        elif isinstance(activations, (list, tuple)):
            if len(activations) != len(num_conv_layers):
                raise ValueError("``activations`` should have the same length "
                                 "as ``num_conv_layers``.")

            activations_enc = [None] * len(activations)
            for i in range(len(activations)):
                if isinstance(activations[i], six.string_types):
                    activations_enc[i] = self._with_device(Activation,
                                                           activations[i])
                elif isinstance(activations[i], Layer):
                    activations_enc[i] = activations[i]
                else:
                    raise ValueError("``activations`` must be a str, or a "
                                     "tuple of str or ``Activation``.")
            self.activations_enc = tuple(activations_enc)

            activations_dec = [None] * len(activations)
            for i in range(len(activations)):
                if isinstance(activations[i], six.string_types):
                    activations_dec[i] = self._with_device(Activation,
                                                           activations[i])
                elif isinstance(activations[i], Layer):
                    activations_dec[i] = activations[i]
                else:
                    raise ValueError("``activations`` must be a str, or a "
                                     "tuple of str or ``Activation``.")
            self.activations_dec = tuple(activations_dec)

        else:
            raise ValueError("``activations`` must be a str, or a tuple of "
                             "str or ``Activation``.")

        if use_downconvolution == False \
                and use_strided_convolution == False \
                and use_maxpooling == False:
            raise ValueError("One of ``use_downconvolution``, "
                             "``use_strided_convolution``, "
                             "``use_maxpooling`` must be ``True``.")
        if use_upconvolution == False \
                and use_strided_deconvolution == False \
                and use_maxunpooling == False:
            raise ValueError("One of ``use_upconvolution``, "
                             "``use_strided_deconvolution``, "
                             "``use_maxunpooling`` must be ``True``.")

        self.use_downconvolution = bool(use_downconvolution)
        self.use_upconvolution = bool(use_upconvolution)
        self.use_strided_convolution = bool(use_strided_convolution)
        self.use_strided_deconvolution = bool(use_strided_deconvolution)
        self.use_maxpooling = bool(use_maxpooling)
        self.use_maxunpooling = bool(use_maxunpooling)
        self.use_maxunpooling_mask = bool(use_maxunpooling_mask)
        self.use_deconvolutions = bool(use_deconvolutions)
        self.use_batch_normalization = bool(use_batch_normalization)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        if _model is None:
            self.model = self._with_device(self._generate_model)
        else:
            self.model = _model

    def save(self, filepath, overwrite=True, include_optimizer=True):
        """Save the model to a single HDF5 file.

        The savefile includes:
            - The model architecture, allowing to re-instantiate the model.
            - The model weights.
            - The state of the optimizer, allowing to resume training
              exactly where you left off.

        This allows you to save the entirety of the state of a model
        in a single file.

        Saved models can be reinstantiated via `keras.models.load_model`.

        The model returned by `load_model` is a compiled model ready to be used
        (unless the saved model was never compiled in the first place).

        Parameters
        ----------
        filepath : str
            Path to the file to save the weights to.

        overwrite : bool, optional
            Whether to silently overwrite any existing file at the target
            location, or provide the user with a manual prompt.

        include_optimizer : bool, optional
            If True, save optimizer's state together.
        """
        # TODO: Update pydoc text when load_model is added to Nethin!
        # TODO: This is an issue with Keras. Follow the development. See issue
        # here:
        #     https://github.com/fchollet/keras/issues/6021
        #     https://github.com/fchollet/keras/issues/5442
        # raise NotImplementedError("The save method currencly does not work in "
        #                           "Keras when there are skip connections. Use "
        #                           "``save_weights`` instead.")

        self.model.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

        # Append own config.
        with h5py.File(filepath, "r+") as f:
            f.attrs[consts.NETHIN_SAVE_KEY] = json.dumps({
                        "class_name": self.__class__.__name__,
                        "config": self.get_config()},
                    default=utils.get_json_type).encode("utf8")
            f.flush()

    @classmethod
    def load(cls,
             filepath,
             custom_objects=None,
             compile=True,
             device=None):
        """Loads a model saved via ``save``.

        Parameters
        ----------
        filepath : str
            Path to the saved model.

        custom_objects : dict, optional
            Optional dictionary mapping names (strings) to custom classes or
            functions to be considered during deserialization.

        compile : bool, optional
            Whether to compile the model after loading. Default is True, the
            model will be compiled.

        device : str, optional
            A particular device to run the model on. Default is ``None``, which
            means to run on the device defined in ``filepath``. Use
            ``nethin.utils.Helper.get_device()`` to see available devices.

        Returns
        -------
            A model instance. If an optimizer was found as part of the saved
            model, the model is already compiled. Otherwise, the model is
            uncompiled and a warning will be displayed. When ``compile`` is set
            to False, the compilation is omitted without any warning.

        Raises
        ------
        ImportError
            If h5py is not available.

        ValueError
            In case of an invalid savefile.
        """
        with h5py.File(filepath, mode="r") as f:
            # Instantiate model
            nethin_config = f.attrs.get(consts.NETHIN_SAVE_KEY)
            if nethin_config is None:
                raise ValueError("No Nethin model found in config file.")
            nethin_config = json.loads(nethin_config.decode("utf-8"))

            if "class_name" not in nethin_config:
                raise ValueError("Invalid format of config file.")
            class_name = nethin_config["class_name"]
            if class_name != cls.__name__:
                raise ValueError('File not saved by this class '
                                 '("%s"" != ""%s").'
                                 % (class_name, cls.__name__,))

            if "config" not in nethin_config:
                raise ValueError("Invalid format of config file.")
            config = nethin_config["config"]

            # Redefine the device in the model config if it was given.
            if device is None:
                if "device" in config:  # Always true
                    device = config["device"]
            else:
                config["device"] = device

        # TODO: Required when device is None?
        model = with_device(device,
                            load_model,
                            filepath,
                            custom_objects=custom_objects,
                            compile=compile)
        # model = load_model(filepath,
        #                    custom_objects=custom_objects,
        #                    compile=compile)

        def _find_layer(layer_name):
            for layer in model.layers:
                if layer.name == layer_name:
                    return layer
            return None

#        # Deserialise the mask if there are unpooling layers
#        for layer in model.layers:
#            if layer.name.startswith("maxunpool"):
#                if isinstance(layer, layers.MaxUnpooling2D):
#                    if isinstance(layer.mask, str):
#                        parts = layer.mask.split("/")
#                        if len(parts) >= 2:
#                            sought_layer_name = parts[0]
#                            sought_layer = _find_layer(sought_layer_name)
#                            if sought_layer is not None:
#                                layer.mask = sought_layer.mask

        unet = UNet.from_config(config, _model=model)

        return unet

    def get_config(self):
        """Returns the config of the model.

        A model config is a Python dictionary (serializable) containing the
        configuration of a model. The same model can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a model does not include connectivity information, nor
        the model class name. These are handled by `Container` (one layer of
        abstraction above).

        Returns
        -------
        config : dict
            A Python dictionary containing the model configuration.
        """
        config = super(UNet, self).get_config()

        if "data_format" not in config:
            config["data_format"] = self.data_format

        if "device" not in config:
            config["device"] = self.device

        if "name" not in config:
            config["name"] = self.name

        config["input_shape"] = self.input_shape
        config["output_channels"] = self.output_channels
        config["num_conv_layers"] = self.num_conv_layers
        config["num_filters"] = self.num_filters
        config["filter_sizes"] = self.filter_sizes
        config["activations"] = self._activations_orig
        config["use_upconvolution"] = self.use_upconvolution
        config["use_maxunpooling"] = self.use_maxunpooling
        config["use_maxunpooling_mask"] = self.use_maxunpooling_mask
        config["use_deconvolutions"] = self.use_deconvolutions
        config["use_batch_normalization"] = self.use_batch_normalization

        return config

    @classmethod
    def from_config(cls, config, _model=None):
        """Creates a model from its config.

        This method is the reverse of `get_config`, capable of instantiating
        the same model from the config dictionary. It does not handle model
        connectivity (handled by Container), nor weights (handled by
        `set_weights`).

        Parameters
        ----------
        config : dict
            Typically the output of get_config.

        Returns
        -------
            An instance of the model.

        Raises
        ------
        KeyError
            If "input_shape" is not in the ``config`` dictionary.
        """
        input_shape = config.pop("input_shape")

        return cls(input_shape, **config, _model=_model)

    def set_weights(self, weights):
        """Sets the weights of the model.

        Parameters
        ----------
        weights : list of numpy arrays
            A list of Numpy arrays with shapes and types matching the output of
            ``model.get_weights()``. If the shapes do not match, Keras will
            throw an error. If the length is shorter than that of
            ``model.get_weights()``, only the first ``len(weights)`` layer's
            weights will be set; if the length is longer, a ``ValueError`` will
            be raised.
        """
        if not isinstance(weights, list):
            raise ValueError("``weights`` must be a list of numpy arrays.")

        weight_values = []
        num_layers = len(self.model.layers)
        for layer_i in range(num_layers):
            layer = self.model.layers[layer_i]
            num_param = len(layer.weights)
            layer_weights = weights[:num_param]

            for tensor, weight in zip(layer.weights, layer_weights):
                if weight is not None:
                    weight_values.append((tensor, weight))

            weights = weights[num_param:]  # Update list

            if len(weights) == 0:
                break  # Silent stop. Allows only the first layers to be set

        if len(weights) > 0:
            warnings.warn("The number of provided parameters does not match "
                          "the output of ``model.get_weights()``.")

        K.batch_set_value(weight_values)

    def _generate_model(self):

        # if self.input_tensor is None:
        #     inputs = Input(shape=self.input_shape)
        # else:
        #     if K.is_keras_tensor(self.input_tensor):
        #         inputs = self.input_tensor
        #     else:
        #         inputs = Input(tensor=self.input_tensor,
        #                        shape=self.input_shape)
        inputs = Input(shape=self.input_shape)
        x = inputs

        # Build the encoding part (contractive path)
        skip_connections = []
        maxpooling_layers = []
        for i in range(len(self.num_conv_layers) - 1):

            num_conv_layers_i = self.num_conv_layers[i]
            num_filters_i = self.num_filters[i]
            filter_sizes_i = self.filter_sizes[i]
            activation_function_i = self.activations_enc[i]
            kernel_initializer_i = self.kernel_initializer

            for j in range(num_conv_layers_i):
                x = Convolution2D(num_filters_i,
                                  (filter_sizes_i, filter_sizes_i),
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format,
                                  kernel_initializer=kernel_initializer_i)(x)

                if self.use_batch_normalization:
                    x = BatchNormalization(axis=self._axis)(x)

                x = activation_function_i(x)

            skip_connections.append(x)

            if self.use_maxpooling:
                if self.use_maxunpooling:
                    pool_name = "maxpool_%d" % (i,)
                    pool_name = pool_name + "_" + str(K.get_uid(pool_name))
                    maxpooling = layers.MaxPooling2D(pool_size=(2, 2),
                                                     strides=(2, 2),
                                                     padding="same",
                                                     data_format=self.data_format,
                                                     compute_mask=self.use_maxunpooling_mask,
                                                     name=pool_name)
                    x = maxpooling(x)
    
                    maxpooling_layers.append(maxpooling)
                else:
                    x = MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding="same",
                                     data_format=self.data_format)(x)

            elif self.use_strided_convolution:
                x = Convolution2D(num_filters_i,
                                  (filter_sizes_i, filter_sizes_i),
                                  strides=(2, 2),  # Downsampling
                                  padding="same",
                                  data_format=self.data_format,
                                  kernel_initializer=kernel_initializer_i)(x)

            elif self.use_downconvolution:
                x = layers.Resampling2D(
                            [0.5, 0.5],
                            method=layers.Resampling2D.ResizeMethod.BILINEAR,
                            data_format=self.data_format)(x)
            else:  # Should not be able to happen!
                raise RuntimeError("No subsampling method selected.")

        # Last encoding part (contractive path)
        num_conv_layers_i = self.num_conv_layers[-1]
        num_filters_i = self.num_filters[-1]
        filter_sizes_i = self.filter_sizes[-1]
        activation_function_i = self.activations_enc[-1]
        kernel_initializer_i = self.kernel_initializer

        for j in range(num_conv_layers_i):
            x = Convolution2D(num_filters_i,
                              (filter_sizes_i, filter_sizes_i),
                              strides=(1, 1),
                              padding="same",
                              data_format=self.data_format,
                              kernel_initializer=kernel_initializer_i)(x)

            if self.use_batch_normalization:
                x = BatchNormalization(axis=self._axis)(x)

            x = activation_function_i(x)

        # First decoding part (expansive path)
        num_conv_layers_i = self.num_conv_layers[-1]
        num_filters_i = self.num_filters[-1]
        filter_sizes_i = self.filter_sizes[-1]
        activation_function_i = self.activations_dec[0]
        kernel_initializer_i = self.kernel_initializer

        for j in range(num_conv_layers_i):
            if self.use_deconvolutions:
                x = layers.Convolution2DTranspose(
                                            num_filters_i,
                                            (filter_sizes_i, filter_sizes_i),
                                            strides=(1, 1),
                                            padding="same",
                                            data_format=self.data_format,
                                            kernel_initializer=kernel_initializer_i)(x)
            else:
                x = Convolution2D(num_filters_i,
                                  (filter_sizes_i, filter_sizes_i),
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format,
                                  kernel_initializer=kernel_initializer_i)(x)

            x = activation_function_i(x)

        # Build decoding part (expansive path)
        for i in range(2, len(self.num_conv_layers) + 1):
            num_conv_layers_i = self.num_conv_layers[-i]
            num_filters_i = self.num_filters[-i]
            filter_sizes_i = self.filter_sizes[-i]
            activation_function_i = self.activations_dec[i - 1]
            kernel_initializer_i = self.kernel_initializer

            if self.use_upconvolution:
                x = layers.Resampling2D((2, 2),
                                        data_format=self.data_format)(x)
                # x = UpSampling2D(size=(2, 2), data_format=self.data_format)(x)

                x = Convolution2D(num_filters_i,
                                  (2, 2),  # Filter size of up-convolution
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format,
                                  kernel_initializer=kernel_initializer_i)(x)
            elif self.use_strided_deconvolution:
                # Strided deconvolution for upsampling
                x = layers.Convolution2DTranspose(
                                            num_filters_i,
                                            (filter_sizes_i, filter_sizes_i),
                                            strides=(2, 2),  # Upsampling
                                            padding="same",
                                            data_format=self.data_format,
                                            kernel_initializer=kernel_initializer_i)(x)
            elif self.use_maxunpooling:
                # Max unpooling
                if self.use_maxunpooling_mask:
                    mask = maxpooling_layers[-(i - 1)].mask

                    # Adjust the number of channels, if necessary
                    if self.num_filters[-i] != self.num_filters[-(i - 1)]:
                        x = Convolution2D(self.num_filters[-i],
                                          (1, 1),
                                          strides=(1, 1),
                                          padding="same",
                                          data_format=self.data_format,
                                          kernel_initializer=kernel_initializer_i)(x)
                else:
                    mask = None

                unpool_name = "maxunpool_%d" % (len(self.num_conv_layers) - i,)
                unpool_name = unpool_name + "_" + str(K.get_uid(unpool_name))
                x = layers.MaxUnpooling2D(pool_size=(2, 2),
                                          strides=(2, 2),  # Not used
                                          padding="same",  # Not used
                                          data_format="channels_last",
                                          mask=mask,
                                          fill_zeros=True,
                                          name=unpool_name)(x)
            else:  # Should not be able to happen!
                raise RuntimeError("No upsampling method selected.")

            skip_connection = skip_connections[-(i - 1)]

            x = Concatenate(axis=self._axis)([x, skip_connection])

            for j in range(num_conv_layers_i):
                if self.use_deconvolutions:
                    x = layers.Convolution2DTranspose(
                                            num_filters_i,
                                            (filter_sizes_i, filter_sizes_i),
                                            strides=(1, 1),
                                            padding="same",
                                            data_format=self.data_format,
                                            kernel_initializer=kernel_initializer_i)(x)
                else:
                    x = Convolution2D(num_filters_i,
                                      (filter_sizes_i, filter_sizes_i),
                                      strides=(1, 1),
                                      padding="same",
                                      data_format=self.data_format,
                                      kernel_initializer=kernel_initializer_i)(x)

                x = activation_function_i(x)

        # Final convolution to generate the requested output number of channels
        x = Convolution2D(self.output_channels,
                          (1, 1),  # Filter size
                          strides=(1, 1),
                          padding="same",
                          data_format=self.data_format,
                          kernel_initializer=kernel_initializer_i)(x)

        outputs = x

#        # Ensure that the model takes into account
#        # any potential predecessors of `input_tensor`.
#        if self.input_tensor is not None:
#            inputs = get_source_inputs(self.input_tensor)

        # Create model
        model = Model(inputs, outputs, name=self.name)

        return model


class GAN(BaseModel):
    """A basic Generative Adversarial Network [1]_.

    Parameters
    ----------
    noise_shape : tuple of ints
        The shape of the input to the generator network, excluding the batch
        dimension.

    data_shape : tuple of ints
        The shape of the input to the discriminator network (i.e., the training
        data), excluding the batch dimension.

    generator : BaseModel or keras.models.Model
        An instance of the generator network.

    discriminator : BaseModel or keras.models.Model
        An instance of the discriminator network. Warning! Avoid relying on
        the discriminator until you have compiled the model.

    num_iter_discriminator : int, optional
        Positive integer. The number of iterations to train the discriminator
        for, for every training iteration of the adversarial model. Default is
        1.

    instance_noise : float, optional
        A factor by which instance noise is scaled. Instance (Gaussian) noise
        is added to the inputs to the discriminator in order to improve
        training of GANs (see [2]_). The noise is annealed as ``1/n``, for
        ``n`` the number of iterations (updates of the generator). Default is
        1.0, which means to add instance noise using a factor 1.0.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".

    device : str, optional
        A particular device to run the model on. Default is ``None``, which
        means to run on the default device (usually "/gpu:0"). Use
        ``nethin.utils.Helper.get_device()`` to see available devices.

    name : str, optional
        The name of the network. Default is "GAN".

    References
    ----------
    .. [1] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu,
       D. Warde-Farley, S. Ozair, A. C. Courville and Y. Bengio (2014).
       "Generative Adversarial Networks". arXiv:1406.2661 [stat.ML], available
       at: https://arxiv.org/abs/1406.2661. NIPS 2014.

    .. [2] M. Arjovsky and L. Bottou. "Towards Principled Methods for Training
       Generative Adversarial Networks". arXiv:1701.04862 [stat.ML], available
       at: https://arxiv.org/abs/1701.04862. ICLR 2017.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import tensorflow as tf
    >>> tf.set_random_seed(42)
    >>> import nethin
    >>> import nethin.models as models
    >>> import nethin.utils as utils
    >>> from keras.models import Sequential
    >>> from keras.layers.convolutional import Convolution2D, UpSampling2D, Convolution2DTranspose
    >>> from keras.layers.advanced_activations import LeakyReLU
    >>> from keras.layers import Flatten, Dense, Dropout, Activation, Reshape  # Input, Flatten, Lambda
    >>> from keras.layers.normalization import BatchNormalization
    >>> from keras.optimizers import RMSprop, Adam
    >>> from keras.datasets import mnist
    >>>
    >>> (x_train, y_train), (x_test, y_test) = mnist.load_data()
    >>> x_train = np.concatenate((x_train, x_test))[..., np.newaxis]
    >>> x_train = (x_train - 127.5) / 127.5
    >>> dropout_rate = 0.4
    >>> alpha = 0.2
    >>> noise_shape = (100,)
    >>> data_shape = (28, 28, 1)
    >>> # batch_size = 256
    >>> batch_size = 100
    >>> device = "/device:CPU:0"
    >>> def discr():
    ...     D = Sequential()
    ...     D.add(Convolution2D(64, (5, 5), strides=2, input_shape=data_shape,
    ...                         padding="same"))
    ...     D.add(BatchNormalization(momentum=0.9))
    ...     D.add(LeakyReLU(alpha))
    ...     D.add(Dropout(dropout_rate))
    ...     D.add(Convolution2D(128, (5, 5), strides=2, input_shape=data_shape,
    ...                         padding="same"))
    ...     D.add(BatchNormalization(momentum=0.9))
    ...     D.add(LeakyReLU(alpha))
    ...     D.add(Dropout(dropout_rate))
    ...     D.add(Convolution2D(256, (5, 5), strides=2, input_shape=data_shape,
    ...                         padding="same"))
    ...     D.add(BatchNormalization(momentum=0.9))
    ...     D.add(LeakyReLU(alpha))
    ...     D.add(Dropout(dropout_rate))
    ...     D.add(Convolution2D(256, (5, 5), strides=2, input_shape=data_shape,
    ...                         padding="same"))
    ...     D.add(BatchNormalization(momentum=0.9))
    ...     D.add(LeakyReLU(alpha))
    ...     D.add(Dropout(dropout_rate))
    ...     D.add(Flatten())
    ...     D.add(Dense(1))
    ...     D.add(Activation("sigmoid"))
    ...
    ...     return D
    >>>
    >>> def gener():
    ...     G = Sequential()
    ...     G.add(Dense(7 * 7 * 256, input_shape=noise_shape))
    ...     G.add(BatchNormalization(momentum=0.9))
    ...     G.add(LeakyReLU(alpha))
    ...     G.add(Dropout(dropout_rate))
    ...     G.add(Reshape((7, 7, 256)))
    ...     G.add(UpSampling2D())
    ...     G.add(Convolution2DTranspose(128, (5, 5), padding="same"))
    ...     G.add(BatchNormalization(momentum=0.9))
    ...     G.add(LeakyReLU(alpha))
    ...     G.add(UpSampling2D())
    ...     G.add(Convolution2DTranspose(64, (5, 5), padding="same"))
    ...     G.add(BatchNormalization(momentum=0.9))
    ...     G.add(LeakyReLU(alpha))
    ...     G.add(Convolution2DTranspose(1, (5, 5), padding="same"))
    ...     G.add(Activation("tanh"))
    ...
    ...     return G
    >>>
    >>> gan = models.GAN(noise_shape, data_shape, gener, discr,
    ...                  num_iter_discriminator=2,
    ...                  device=device)
    >>> gan.compile(optimizer=RMSprop(lr=0.0001),
    ...             loss="binary_crossentropy",
    ...             metrics=["accuracy"])
    >>> loss_a = []
    >>> loss_d = []
    >>> # for it in range(5000):
    >>> im_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    >>> loss = gan.train_on_batch(im_real)
    >>> loss_d.append(loss[0][1])
    >>> loss_a.append(loss[1][1])
    >>> # print("it: %d, loss d: %f, loss a: %f" % (it, loss[0][1], loss[1][1]))
    >>> # Loss D (CE, acc)
    >>> np.round(loss[0], 2)  # doctest: +NORMALIZE_WHITESPACE
    array([3.42, 0.34])
    >>> # Loss D real (CE, acc)
    >>> np.round(loss[1], 2)  # doctest: +NORMALIZE_WHITESPACE
    array([0.6 , 0.68], dtype=float32)
    >>> # Loss D fake (CE, acc)
    >>> np.round(loss[2], 2)  # doctest: +NORMALIZE_WHITESPACE
    array([6.24, 0.  ], dtype=float32)
    >>> # gan = models.GAN(noise_shape, data_shape, gener, discr,
    >>> #                  num_iter_discriminator=1,
    >>> #                  instance_noise=1.0,
    >>> #                  device=device)
    >>> # gan.compile(optimizer=[RMSprop(lr=0.0001),
    >>> #                        Adam(lr=0.0001, beta_1=0.9)],
    >>> #             loss="binary_crossentropy",
    >>> #             metrics=["accuracy"])
    >>> # acc = []
    >>> # for it in range(10000):  # equals 5000 iterations
    >>> #     im_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    >>> #     loss = gan.train_on_batch(im_real)
    >>> #     acc.append(loss[0][1])
    >>> #     print("it: %d, iterations: %d, acc: %s, all: %s"
    >>> #           % (it, gan._iterations, str(loss[0][1]), str(loss)))
    >>> # # import matplotlib.pyplot as plt
    >>> # # plt.figure()
    >>> # # plt.subplot(1, 2, 1)
    >>> # # plt.plot(loss_d)
    >>> # # plt.subplot(1, 2, 2)
    >>> # # plt.plot(loss_a)
    >>> # # plt.show()
    >>> # # noise = np.random.uniform(-1.0, 1.0, size=(1,) + noise_shape)
    >>> # # fake_im = G.predict_on_batch(noise)
    >>> # # plt.figure()
    >>> # # plt.subplot(1, 2, 1)
    >>> # # plt.imshow(im_real[0, :, :, 0])
    >>> # # plt.subplot(1, 2, 2)
    >>> # # plt.imshow(fake_im[0, :, :, 0])
    >>> # # plt.show()
    """
    def __init__(self,
                 input_shape,
                 output_shape,
                 generator,
                 discriminator,
                 num_iter_discriminator=1,
                 instance_noise=1.0,
                 data_format=None,
                 device=None,
                 name="GAN"):

        super(GAN, self).__init__("nethin.models.GAN",
                                  data_format=data_format,
                                  device=device,
                                  name=name)

        if input_shape is not None:
            if not isinstance(input_shape, (tuple, list)):
                raise ValueError('"input_shape" must be a tuple.')
        self.input_shape = tuple([int(d) for d in input_shape])

        if output_shape is not None:
            if not isinstance(output_shape, (tuple, list)):
                raise ValueError('"data_shape" must be a tuple.')
        self.output_shape = tuple([int(d) for d in output_shape])

        self.generator = generator
        self.discriminator = discriminator
        self.num_iter_discriminator = max(1, int(num_iter_discriminator))
        if instance_noise is None:
            self.instance_noise = instance_noise
        else:
            self.instance_noise = max(0.0, float(instance_noise))

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        self._model = self._with_device(self._generate_model)
        self._batch_updates = 0
        self._iterations = 0

    def _generate_model(self):

        self._D = self.discriminator()
        self._G = self.generator()

        def _generate_D():

            inputs = Input(shape=self.output_shape)
            outputs = self._D(inputs)
            model_D = Model(inputs, outputs)

            return model_D

        def _generate_G():

            inputs = Input(shape=self.input_shape)
            outputs = self._G(inputs)
            model_G = Model(inputs, outputs)

            return model_G

        def _generate_GAN(G, D):

            inputs = Input(shape=self.input_shape)
            x = G(inputs)
            D.trainable = False
            outputs = D(x)

            model_GAN = Model(inputs, outputs)

            return model_GAN

        model_G = _generate_G()
        model_D = _generate_D()

        self._model_GAN_factory = _generate_GAN

        return [model_G, model_D, None]

    def compile(self,
                optimizer,
                loss,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None):
        """Configures the model for training.

        Parameters
        ----------
        optimizer : str, keras.optimizers.Optimizer or list of str or
                    keras.optimizers.Optimizer, length 2
            String (name of optimizer) or optimizer object. See
            `optimizers <https://keras.io/optimizers>`_. If a list, the first
            optimiser is used for the discriminator and the second optimiser is
            used for the adversarial model.

        loss : str or list of str, length 2
            String (name of objective function) or objective function. See
            `losses <https://keras.io/losses>`_. If a list, the first loss is
            used for the discriminator and the second loss is used for the
            adversarial model.

         metrics : list of str or list of list of str, length 2, optional
             List of metrics to be evaluated by the model during training and
             testing. Typically you will use ``metrics=["accuracy"]``. If a
             list of lists of str, the first list of metrics are used for the
             discriminator and the second list of metrics is used for the
             adversarial model.

        loss_weights : list or dict, optional
            Currently ignored by this model.

        sample_weight_mode : None, str, list or dict, optional
            Currently ignored by this model.

        weighted_metrics : list, optional
            Currently ignored by this model.

        target_tensors : Tensor, optional
            Currently ignored by this model.

        **kwargs
            When using the Theano/CNTK backends, these arguments are passed
            into ``K.function``. When using the TensorFlow backend, these
            arguments are passed into ``tf.Session.run``.

        Raises
        ------
        ValueError
            In case of invalid arguments for ``optimizer``, ``loss``,
            ``metrics`` or ``sample_weight_mode``.
        """
        if (weighted_metrics is None) and (target_tensors is None):
            # Recent additions to compile may not be available.
            self._with_device(self._compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode)
        else:
            self._with_device(self._compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode,
                              weighted_metrics=weighted_metrics,
                              target_tensors=target_tensors)

    def _compile(self,
                 optimizer,
                 loss,
                 metrics=None,
                 loss_weights=None,
                 sample_weight_mode=None,
                 weighted_metrics=None,
                 target_tensors=None):

        optimizer = utils.normalize_object(optimizer, 2, "optimizers")
        loss = utils.normalize_str(loss, 2, "losses")
        if metrics is not None:
            if isinstance(metrics, list):
                if isinstance(metrics[0], str):
                    metrics = (metrics,) * 2
                elif not isinstance(metrics, list):
                    raise ValueError('The "metrics" argument must be a list '
                                     'of str or a list of a list of str.')
            else:
                raise ValueError('The "metrics" argument must be a list of '
                                 'str or a list of a list of str.')
        else:
            metrics = [metrics, metrics]

        model_G, model_D, model_GAN = self._model

        model_D.compile(optimizer=optimizer[0],
                        loss=loss[0],
                        metrics=metrics[0])

        if model_GAN is None:
            model_GAN = self._model_GAN_factory(model_G, model_D)

        model_GAN.compile(optimizer=optimizer[1],
                          loss=loss[1],
                          metrics=metrics[1])

        self._model = [model_G, model_D, model_GAN]

    def train_on_batch(self,
                       x,
                       y=None,
                       sample_weight=None,
                       class_weight=None):
        """Runs a single gradient update on a single batch of data.

        Arguments
        ---------
        x : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of training data, or list of Numpy arrays if the model
            has multiple inputs. If all inputs in the model are named, you can
            also pass a dictionary mapping input names to Numpy arrays.

        y : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray,
            optional
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.

        sample_weight : numpy.ndarray, optional
            Optional array of the same length as ``x``, containing weights to
            apply to the model's loss for each sample. In the case of temporal
            data, you can pass a 2D array with shape (samples,
            sequence_length), to apply a different weight to every timestep of
            every sample. In this case you should make sure to specify
            ``sample_weight_mode="temporal"`` in ``compile()``.

        class_weight : dict, optional
            Optional dictionary mapping class indices (integers) to a weight
            (float) to apply to the model's loss for the samples from this
            class during training. This can be useful to tell the model to
            "pay more attention" to samples from an under-represented class.

        Returns
        -------
        Scalar training loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs and/or metrics).
        The attribute ``model.metrics_names`` will give you the display labels
        for the scalar outputs. Returns the discriminator loss followed by the
        adversarial model's loss.
        """
        return self._with_device(self._train_on_batch,
                                 x,
                                 y,
                                 sample_weight=sample_weight,
                                 class_weight=class_weight)

    def _train_on_batch(self,
                        x,
                        y=None,
                        sample_weight=None,
                        class_weight=None):

        model_G, model_D, model_GAN = self._model

        assert(model_GAN is not None)

        batch_size = x.shape[0]
        if y is None:
            y = np.zeros([batch_size, 1])

        # Train discriminator
        if self.instance_noise is not None:
            noise = np.random.randn(*x.shape) \
                * (float(self.instance_noise) / float(self._batch_updates + 1))
            noise = noise.astype(x.dtype)
            loss_D_real = model_D.train_on_batch(x + noise, y)
        else:
            loss_D_real = model_D.train_on_batch(x, y)

        z = np.random.normal(0.0, 1.0, size=(batch_size,) + self.input_shape)
        x_fake = model_G.predict_on_batch(z)
        y_fake = np.ones([batch_size, 1])

        if self.instance_noise is not None:
            noise = np.random.randn(*x.shape) \
                * (float(self.instance_noise) / float(self._batch_updates + 1))
            noise = noise.astype(x.dtype)
            loss_D_fake = model_D.train_on_batch(x_fake + noise, y_fake)
        else:
            loss_D_fake = model_D.train_on_batch(x_fake, y_fake)

        loss_D = (0.5 * np.add(loss_D_real, loss_D_fake)).tolist()

#        im = np.concatenate((x, im_fake))
#        _y = np.zeros([2 * batch_size, 1])
#        _y[batch_size:, :] = 1

        # Train GAN model (the generator part)
        loss_GAN = None
        if (self._batch_updates + 1) % self.num_iter_discriminator == 0:

            z = np.random.normal(0.0, 1.0,
                                 size=(batch_size,) + self.input_shape)
            loss_GAN = model_GAN.train_on_batch(z, y)

            self._iterations += 1

        self._batch_updates += 1

        self.metrics_names = [model_D.metrics_names,
                              "DiscriminatorReal",
                              "DiscriminatorFake",
                              model_GAN.metrics_names]

        return loss_D, loss_D_real, loss_D_fake, loss_GAN


class WassersteinGAN(BaseModel):
    """A Wasserstein Generative Adversarial Network [1]_.

    Parameters
    ----------
    input_shape : tuple of ints
        The shape of the input to the generator network, excluding the batch
        dimension.

    output_shape : tuple of ints
        The shape of the input to the discriminator network (i.e., the training
        data), excluding the batch dimension.

    generator : Callable
        Should return an instance of the generator network as a BaseModel or
        keras.models.Model.

    discriminator : Callable
        Should return an instance of the discriminator network as a BaseModel
        or keras.models.Model. Warning! Avoid relying on the discriminator
        until you have compiled the model (using ``compile``).

    num_iter_discriminator : int, optional
        Positive integer. The number of iterations to train the discriminator
        for, for every training iteration of the adversarial model. Default is
        5.

    boost_critic : bool, optional
        Whether or not to perform extra training of the critic at the beginning
        and at each 500th iteration. Default is True, boost the critic.

    boost_batches : int, optional
        If ``boost_critic=True``, the number of batch updates to use for
        boosting the critic during a boost session. Default is 100.

    boost_init_iters : int, optional
        If ``boost_critic=True``, the number of initial iterations for which
        the critic will be boosted. Default is 25.

    boost_every_nth : int, optional
        If ``boost_critic=True``, the boost will be used every
        ``boost_every_nth``th iteration. Default is 500.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".

    device : str, optional
        A particular device to run the model on. Default is ``None``, which
        means to run on the default device (usually the first GPU). Use
        ``nethin.utils.Helper.get_device()`` to see available devices.

    name : str, optional
        The name of the network. Default is "WassersteinGAN".

    References
    ----------
    .. [1] M. Arjovsky, S. Chintala, L. Bottou (2017). "Wasserstein GAN".
       arXiv:1701.07875v2 [stat.ML], available at:
       https://arxiv.org/abs/1701.07875.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> import tensorflow as tf
    >>> tf.set_random_seed(42)
    >>> import nethin
    >>> import nethin.models as models
    >>> import nethin.constraints as constraints
    >>> import nethin.utils as utils
    >>> from keras.models import Model
    >>> from keras.layers.convolutional import Convolution2D, UpSampling2D, Convolution2DTranspose
    >>> from keras.layers import Input
    >>> from keras.layers.advanced_activations import LeakyReLU
    >>> from keras.layers import Flatten, Dense, Dropout, Activation, Reshape  # Input, Flatten, Lambda
    >>> from keras.layers.normalization import BatchNormalization
    >>> from keras.optimizers import RMSprop  # Optimizer
    >>> from keras.datasets import mnist
    >>>
    >>> (x_train, y_train), (x_test, y_test) = mnist.load_data()
    >>> x_train = np.concatenate((x_train, x_test))[..., np.newaxis]
    >>> x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    >>> y_train = np.concatenate((y_train, y_test))
    >>> learning_rate = 0.00001
    >>> dropout_rate = 0.4
    >>> alpha = 0.2
    >>> noise_shape = (100,)
    >>> data_shape = (28, 28, 1)
    >>> # batch_size = 256
    >>> batch_size = 100
    >>> device = "/device:CPU:0"
    >>> # device = "/device:GPU:0"
    >>> def discr():
    ...     inputs = Input(shape=data_shape)
    ...     x = Convolution2D(64, (5, 5), strides=2, input_shape=data_shape,
    ...                       padding="same",
    ...                       bias_constraint=constraints.BoxConstraint(0.01),
    ...                       kernel_constraint=constraints.BoxConstraint(0.01)
    ...                      )(inputs)
    ...     x = LeakyReLU(alpha)(x)
    ...     x = Dropout(dropout_rate)(x)
    ...     x = Convolution2D(128, (5, 5), strides=2,
    ...                       padding="same",
    ...                       bias_constraint=constraints.BoxConstraint(0.01),
    ...                       kernel_constraint=constraints.BoxConstraint(0.01)
    ...                      )(x)
    ...     x = LeakyReLU(alpha)(x)
    ...     x = Dropout(dropout_rate)(x)
    ...     x = Convolution2D(256, (5, 5), strides=2,
    ...                       padding="same",
    ...                       bias_constraint=constraints.BoxConstraint(0.01),
    ...                       kernel_constraint=constraints.BoxConstraint(0.01)
    ...                      )(x)
    ...     x = LeakyReLU(alpha)(x)
    ...     x = Dropout(dropout_rate)(x)
    ...     x = Convolution2D(256, (5, 5), strides=2,
    ...                       padding="same",
    ...                       bias_constraint=constraints.BoxConstraint(0.01),
    ...                       kernel_constraint=constraints.BoxConstraint(0.01)
    ...                      )(x)
    ...     x = LeakyReLU(alpha)(x)
    ...     x = Dropout(dropout_rate)(x)
    ...     x = Flatten()(x)
    ...     outputs = Dense(1,
    ...                     bias_constraint=constraints.BoxConstraint(0.01),
    ...                     kernel_constraint=constraints.BoxConstraint(0.01)
    ...                    )(x)
    ...
    ...     D = Model(inputs, outputs)
    ...
    ...     return D
    >>>
    >>> def gener():
    ...     inputs = Input(shape=noise_shape)
    ...     x = Dense(7 * 7 * 256, input_shape=noise_shape)(inputs)
    ...     x = BatchNormalization(momentum=0.9)(x)
    ...     x = Activation("relu")(x)
    ...     x = Reshape((7, 7, 256))(x)
    ...     x = Dropout(dropout_rate)(x)
    ...     x = UpSampling2D()(x)
    ...     x = Convolution2DTranspose(128, (5, 5), padding="same")(x)
    ...     x = BatchNormalization(momentum=0.9)(x)
    ...     x = Activation("relu")(x)
    ...     x = UpSampling2D()(x)
    ...     x = Convolution2DTranspose(64, (5, 5), padding="same")(x)
    ...     x = BatchNormalization(momentum=0.9)(x)
    ...     x = Activation("relu")(x)
    ...     x = Convolution2DTranspose(1, (5, 5), padding="same")(x)
    ...     outputs = Activation("tanh")(x)
    ...
    ...     G = Model(inputs, outputs)
    ...
    ...     return G
    >>>
    >>> wgan = models.WassersteinGAN(noise_shape, data_shape, gener, discr,
    ...                              num_iter_discriminator=2,
    ...                              device=device,
    ...                              boost_critic=False)
    >>> wgan.compile(optimizer=RMSprop(lr=learning_rate))
    >>> loss_d = []
    >>> # for it in range(5000):
    >>> im_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    >>> loss = wgan.train_on_batch(im_real)
    >>> loss_d.append(loss[0])
    >>> # print("it: %d, loss d: %f, all: %s" % (it, loss[0], str(loss)))
    >>> np.round(loss, 4)  # doctest: +NORMALIZE_WHITESPACE
    array([-0.0008, 0.0007, 0.0002, -0.0002])
    >>> loss = wgan.train_on_batch(im_real)
    >>> loss_d.append(loss[0])
    >>> np.round(loss, 4)  # doctest: +NORMALIZE_WHITESPACE
    array([ 0.0003, -0.0005,  0.0001, -0.0003])
    >>> # wgan = models.WassersteinGAN(noise_shape, data_shape, gener, discr,
    >>> #                              num_iter_discriminator=10,
    >>> #                              device=device,
    >>> #                              boost_critic=True,
    >>> #                              boost_batches=20,
    >>> #                              boost_every_nth=100,
    >>> #                              boost_init_iters=25)
    >>> # wgan.compile(optimizer=RMSprop(lr=learning_rate))
    >>> # loss_d = []
    >>> # for it in range(10000):  # equals 2000 iterations
    >>> #     im_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    >>> #     loss = wgan.train_on_batch(im_real)
    >>> #     if loss[-1] is not None:
    >>> #         loss_d.append(loss[0])
    >>> #     print("it: %d, iterations: %d, loss d: %f, all: %s"
    >>> #           % (it, wgan._iterations, loss[0], str(loss)))
    >>> # # print("it: %d, loss d: %f, all: %s" % (it, loss[0], str(loss)))
    >>> # # import matplotlib.pyplot as plt
    >>> # # plt.figure()
    >>> # # plt.subplot(1, 2, 1)
    >>> # # plt.plot(loss_d)
    >>> # # plt.subplot(1, 2, 2)
    >>> # # plt.plot(loss_a)
    >>> # # plt.show()
    >>> # # noise = np.random.uniform(-1.0, 1.0, size=(1,) + noise_shape)
    >>> # # fake_im = G.predict_on_batch(noise)
    >>> # # plt.figure()
    >>> # # plt.subplot(1, 2, 1)
    >>> # # plt.imshow(im_real[0, :, :, 0])
    >>> # # plt.subplot(1, 2, 2)
    >>> # # plt.imshow(fake_im[0, :, :, 0])
    >>> # # plt.show()
    """
    def __init__(self,
                 input_shape,
                 output_shape,
                 generator,
                 discriminator,
                 num_iter_discriminator=1,
                 boost_critic=True,
                 boost_batches=100,
                 boost_init_iters=25,
                 boost_every_nth=500,
                 data_format=None,
                 device=None,
                 name="WassersteinGAN"):

        super(WassersteinGAN, self).__init__("nethin.models.WassersteinGAN",
                                             data_format=data_format,
                                             device=device,
                                             name=name)

        if input_shape is not None:
            if not isinstance(input_shape, (tuple, list)):
                raise ValueError('"input_shape" must be a tuple.')
        self.input_shape = tuple([int(d) for d in input_shape])

        if output_shape is not None:
            if not isinstance(output_shape, (tuple, list)):
                raise ValueError('"output_shape" must be a tuple.')
        self.output_shape = tuple([int(d) for d in output_shape])

        self.generator = generator
        self.discriminator = discriminator
        self.num_iter_discriminator = max(1, int(num_iter_discriminator))
        self.boost_critic = bool(boost_critic)
        self.boost_batches = max(1, int(boost_batches))
        self.boost_init_iters = max(0, int(boost_init_iters))
        self.boost_every_nth = max(1, int(boost_every_nth))

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        self._model = self._with_device(self._generate_model)
        self._batch_updates = 0
        self._iterations = 0

    def _generate_model(self):

        self._D = self.discriminator()
        self._G = self.generator()

        def _generate_D():

            inputs = Input(shape=self.output_shape)
            outputs = self._D(inputs)
            model_D = Model(inputs, outputs)

            return model_D

        def _generate_G():

            inputs = Input(shape=self.input_shape)
            outputs = self._G(inputs)
            model_G = Model(inputs, outputs)

            return model_G

        def _generate_GAN(G, D):

            inputs = Input(shape=self.input_shape)
            x = G(inputs)
            D.trainable = False
            outputs = D(x)

            model_GAN = Model(inputs, outputs)

            return model_GAN

        model_G = _generate_G()
        model_D = _generate_D()

        self._model_GAN_factory = _generate_GAN

        return [model_G, model_D, None]

    def compile(self,
                optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None):
        """Configures the model for training.

        Parameters
        ----------
        optimizer : str, keras.optimizers.Optimizer or list of str or
                    keras.optimizers.Optimizer, length 2
            String (name of optimizer) or optimizer object. See
            `optimizers <https://keras.io/optimizers>`_. If a list, the first
            optimiser is used for the discriminator and the second optimiser is
            used for the adversarial model.

        loss : str or list of str, length 2
            String (name of objective function) or objective function. See
            `losses <https://keras.io/losses>`_. If a list, the first loss is
            used for the discriminator and the second loss is used for the
            adversarial model.

         metrics : list of str or list of list of str, length 2, optional
             List of metrics to be evaluated by the model during training and
             testing. Typically you will use ``metrics=["accuracy"]``. If a
             list of lists of str, the first list of metrics are used for the
             discriminator and the second list of metrics is used for the
             adversarial model.

        loss_weights : list or dict, optional
            Currently ignored by this model.

        sample_weight_mode : None, str, list or dict, optional
            Currently ignored by this model.

        weighted_metrics : list, optional
            Currently ignored by this model.

        target_tensors : Tensor, optional
            Currently ignored by this model.

        **kwargs
            When using the Theano/CNTK backends, these arguments are passed
            into ``K.function``. When using the TensorFlow backend, these
            arguments are passed into ``tf.Session.run``.

        Raises
        ------
        ValueError
            In case of invalid arguments for ``optimizer``, ``loss``,
            ``metrics`` or ``sample_weight_mode``.
        """
        if (weighted_metrics is None) and (target_tensors is None):
            # Recent additions to compile may not be available.
            self._with_device(self._compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode)
        else:
            self._with_device(self._compile,
                              optimizer,
                              loss,
                              metrics=metrics,
                              loss_weights=loss_weights,
                              sample_weight_mode=sample_weight_mode,
                              weighted_metrics=weighted_metrics,
                              target_tensors=target_tensors)

    def _compile(self,
                 optimizer,
                 loss=None,
                 metrics=None,
                 loss_weights=None,
                 sample_weight_mode=None,
                 weighted_metrics=None,
                 target_tensors=None):

        optimizer = utils.normalize_object(optimizer, 2, "optimizers")
        # loss = utils.normalize_str(loss, 2, "losses")
        if metrics is not None:
            if isinstance(metrics, list):
                if isinstance(metrics[0], str):
                    metrics = (metrics,) * 2
                elif not isinstance(metrics, list):
                    raise ValueError('The "metrics" argument must be a list '
                                     'of str or a list of a list of str.')
            else:
                raise ValueError('The "metrics" argument must be a list of '
                                 'str or a list of a list of str.')
        else:
            metrics = [metrics, metrics]

        def wasserstein_distance(y_true, y_pred):
            return K.mean(y_true * y_pred)

        model_G, model_D, model_GAN = self._model

        model_D.compile(optimizer=optimizer[0],
                        loss=wasserstein_distance,
                        metrics=metrics[0])

        if model_GAN is None:
            model_GAN = self._model_GAN_factory(model_G, model_D)

        model_GAN.compile(optimizer=optimizer[1],
                          loss=wasserstein_distance,
                          metrics=metrics[1])

        self._model = [model_G, model_D, model_GAN]

    def train_on_batch(self,
                       x,
                       y=None,
                       sample_weight=None,
                       class_weight=None):
        """Runs a single gradient update on a single batch of data.

        Arguments
        ---------
        x : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray
            Numpy array of training data, or list of Numpy arrays if the model
            has multiple inputs. If all inputs in the model are named, you can
            also pass a dictionary mapping input names to Numpy arrays.

        y : numpy.ndarray or list of numpy.ndarray or dict of numpy.ndarray,
            optional
            Numpy array of target data, or list of Numpy arrays if the model
            has multiple outputs. If all outputs in the model are named, you
            can also pass a dictionary mapping output names to Numpy arrays.

        sample_weight : numpy.ndarray, optional
            Optional array of the same length as ``x``, containing weights to
            apply to the model's loss for each sample. In the case of temporal
            data, you can pass a 2D array with shape (samples,
            sequence_length), to apply a different weight to every timestep of
            every sample. In this case you should make sure to specify
            ``sample_weight_mode="temporal"`` in ``compile()``.

        class_weight : dict, optional
            Optional dictionary mapping class indices (integers) to a weight
            (float) to apply to the model's loss for the samples from this
            class during training. This can be useful to tell the model to
            "pay more attention" to samples from an under-represented class.

        Returns
        -------
        Scalar training loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs and/or metrics).
        The attribute ``model.metrics_names`` will give you the display labels
        for the scalar outputs. Returns the discriminator loss followed by the
        adversarial model's loss.
        """
        return self._with_device(self._train_on_batch,
                                 x,
                                 y,
                                 sample_weight=sample_weight,
                                 class_weight=class_weight)

    def _train_on_batch(self,
                        x,
                        y=None,
                        sample_weight=None,
                        class_weight=None):

        model_G, model_D, model_GAN = self._model

        assert(model_GAN is not None)

        if self.boost_critic \
                and (self._iterations < self.boost_init_iters
                     or self._iterations % self.boost_every_nth == 0):
            d_iters = max(self.num_iter_discriminator, self.boost_batches)
        else:
            d_iters = self.num_iter_discriminator

        # Train discriminator
        batch_size = x.shape[0]
        y_ = np.ones([batch_size, 1])

        loss_D_real = model_D.train_on_batch(x, y_)

        noise = np.random.normal(0.0, 1.0,
                                 size=(batch_size,) + self.input_shape)
        x_ = model_G.predict_on_batch(noise)
        loss_D_fake = model_D.train_on_batch(x_, -y_)

        if isinstance(loss_D_fake, (list, tuple)):
            wasserstein_loss = -1 * loss_D_real[0] - loss_D_fake[0]
        else:
            wasserstein_loss = -1 * loss_D_real - loss_D_fake

        # Train GAN model (the generator part)
        loss_GAN = None
        if (self._batch_updates == 0) \
                or ((self._batch_updates + 1) % d_iters == 0):

            noise = np.random.normal(0.0, 1.0,
                                     size=(batch_size,) + self.input_shape)

            loss_GAN = model_GAN.train_on_batch(noise, y_)

            self._iterations += 1

        self._batch_updates += 1

        self.metrics_names = ["wasserstein",
                              model_D.metrics_names,
                              model_D.metrics_names,
                              model_GAN.metrics_names]

        return wasserstein_loss, loss_D_real, loss_D_fake, loss_GAN


if __name__ == "__main__":
    import doctest
    doctest.testmod()
