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
from six import with_metaclass

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Model
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
import keras.optimizers as optimizers
from keras.optimizers import Optimizer, Adam

from nethin.utils import with_device, Helper, to_snake_case
import nethin.padding as padding
import nethin.layers as layers

__all__ = ["BaseModel",
           "UNet"]


class BaseModel(with_metaclass(abc.ABCMeta, object)):
    # TODO: Add fit_generator, evaluate_generator and predict_generator
    # TODO: What about get_layer?
    def __init__(self, nethin_name, data_format=None, device=None, name=None):

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

    def get_model(self):

        return self.model

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
        ...     model.predict_on_batch(X)
        array([[[-0.44601449],
                [-1.13214135],
                [-0.87168205]]], dtype=float32)
        >>> if not failed:
        ...     model.save(fn)  # Creates a temporary HDF5 file
        ...     del model  # Deletes the existing model
        ...     # Returns a compiled model identical to the previous one
        ...     model = load_model(fn)
        ...     model.predict_on_batch(X)
        array([[[-0.44601449],
                [-1.13214135],
                [-0.87168205]]], dtype=float32)
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

    def compile(self, optimizer, loss,
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

    def train_on_batch(self, x, y,
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

#    def save


class UNet(BaseModel):
    """Generates the U-Net model by Ronneberger et al. (2015) [1]_.

    Parameters
    ----------
    input_shape : tuple of ints, length 3, optional
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

    use_upconvolution : bool, optional
        The use of upsampling followed by a convolution can reduce artifacts
        that appear when deconvolutions with ``stride > 1`` are used. When
        ``use_upconvolution=False``, strided deconvolutions are used instead
        unless ``use_maxunpooling=True``. Default is True, which means to use
        upsampling followed by a 2x2 convolution (denoted "upconvolution" in
        [1]_).

    use_maxunpooling : bool, optional
        If True, 2x2 unpooling will be used instead of upconvolution or strided
        deconvolutions. Default is False, which means to use the value of
        ``use_upconvolution`` to determine the upsampling procedure.

    use_maxunpooling_mask : bool, optional
        If ``use_maxunpooling=True`` and  ``use_maxunpooling_mask=True``, then
        the maxunpooling will use the sites of the maximum values from the
        maxpooling operation during the unpooling. Default is False, do not use
        the mask during unpooling.

    use_deconvolutions : bool, optional
        Use deconvolutions (transposed convolutinos) in the deconding part
        instead of convolutions. This should have no practical difference,
        since deconvolutions are also convolutions, but may be preferred in
        some cases. Default is False, do not use deconvolutions in the decoding
        part.

    use_batch_normalization : bool, optional
        Whether or not to use batch normalization after each convolution in the
        encoding part. Default is False, do not use batch normalization.

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
    >>> np.abs(np.sum(Y - X) - 1502.14844) < 5e-4
    True
    """
    def __init__(self,
                 input_shape,
                 output_channels=1,
                 num_conv_layers=[2, 2, 2, 2, 1],
                 num_filters=[64, 128, 256, 512, 1024],
                 filter_sizes=3,
                 activations="relu",
                 use_upconvolution=True,
                 use_maxunpooling=False,
                 use_maxunpooling_mask=False,
                 use_deconvolutions=False,
                 use_batch_normalization=False,
                 data_format=None,
                 device=None,
                 name="U-Net"):

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

        self.num_filters = tuple(num_filters)

        if isinstance(filter_sizes, int):
            self.filter_sizes = (filter_sizes,) * len(self.num_conv_layers)
        elif len(filter_sizes) != len(self.num_conv_layers):
                raise ValueError("``filter_sizes`` should have the same "
                                 "length as ``num_conv_layers``.")
        else:
            self.filter_sizes = tuple(filter_sizes)

        if isinstance(activations, str):
            activations_enc = [self._with_device(Activation, activations)
                               for i in range(len(num_conv_layers))]
            self.activations_enc = tuple(activations_enc)

            activations_dec = [self._with_device(Activation, activations)
                               for i in range(len(num_conv_layers))]
            self.activations_dec = tuple(activations_dec)
        elif len(activations) != len(num_conv_layers):
            raise ValueError("``activations`` should have the same length as "
                             "``num_conv_layers``.")
        elif isinstance(activations, (list, tuple)):

            activations_enc = [None] * len(activations)
            for i in range(len(activations)):
                activations_enc[i] = self._with_device(Activation,
                                                       activations[i])
            self.activations_enc = tuple(activations_enc)

            activations_dec = [None] * len(activations)
            for i in range(len(activations)):
                activations_dec[i] = self._with_device(Activation,
                                                       activations[i])
            self.activations_dec = tuple(activations_dec)
        else:
            raise ValueError("``activations`` must be a str, or a tuple of "
                             "str or ``Activation``.")

        self.use_upconvolution = bool(use_upconvolution)
        self.use_maxunpooling = bool(use_maxunpooling)
        self.use_maxunpooling_mask = bool(use_maxunpooling_mask)
        self.use_deconvolutions = bool(use_deconvolutions)
        self.use_batch_normalization = bool(use_batch_normalization)

        if self.data_format == "channels_last":
            self._axis = 3
        else:  # data_format == "channels_first":
            self._axis = 1

        self.model = self._with_device(self._generate_model)

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
        raise NotImplementedError("The save method currencly does not work in "
                                  "Keras when there are skip connections. Use "
                                  "``save_weights`` instead.")

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

            for j in range(num_conv_layers_i):
                x = Convolution2D(num_filters_i,
                                  (filter_sizes_i, filter_sizes_i),
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format)(x)
                if self.use_batch_normalization:
                    x = BatchNormalization(axis=self._axis)(x)
                x = activation_function_i(x)

            skip_connections.append(x)

            # TODO: Add alternatives here, e.g. strided convolution
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

        # Last encoding part (contractive path)
        num_conv_layers_i = self.num_conv_layers[-1]
        num_filters_i = self.num_filters[-1]
        filter_sizes_i = self.filter_sizes[-1]
        activation_function_i = self.activations_enc[-1]

        for j in range(num_conv_layers_i):
            x = Convolution2D(num_filters_i,
                              (filter_sizes_i, filter_sizes_i),
                              strides=(1, 1),
                              padding="same",
                              data_format=self.data_format)(x)
            if self.use_batch_normalization:
                x = BatchNormalization(axis=self._axis)(x)
            x = activation_function_i(x)

        # First decoding part (expansive path)
        num_conv_layers_i = self.num_conv_layers[-1]
        num_filters_i = self.num_filters[-1]
        filter_sizes_i = self.filter_sizes[-1]
        activation_function_i = self.activations_dec[0]

        for j in range(num_conv_layers_i):
            if self.use_deconvolutions:
                x = layers.Convolution2DTranspose(
                        num_filters_i,
                        (filter_sizes_i, filter_sizes_i),
                        strides=(1, 1),
                        padding="same",
                        data_format=self.data_format)(x)
            else:
                x = Convolution2D(num_filters_i,
                                  (filter_sizes_i, filter_sizes_i),
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format)(x)
            x = activation_function_i(x)

        # Build decoding part (expansive path)
        for i in range(2, len(self.num_conv_layers) + 1):
            num_conv_layers_i = self.num_conv_layers[-i]
            num_filters_i = self.num_filters[-i]
            filter_sizes_i = self.filter_sizes[-i]
            activation_function_i = self.activations_dec[i - 1]

            if self.use_upconvolution:
                x = UpSampling2D(size=(2, 2), data_format=self.data_format)(x)
                x = Convolution2D(num_filters_i,
                                  (2, 2),  # Filter size of up-convolution
                                  strides=(1, 1),
                                  padding="same",
                                  data_format=self.data_format)(x)
            else:
                if not self.use_maxunpooling:
                    # Strided deconvolution for upsampling
                    x = layers.Convolution2DTranspose(
                            num_filters_i,
                            (filter_sizes_i, filter_sizes_i),
                            strides=(2, 2),  # Upsampling
                            padding="same",
                            data_format=self.data_format)(x)
                else:
                    # Max unpooling
                    if self.use_maxunpooling_mask:
                        mask = maxpooling_layers[-(i - 1)].mask

                        # Adjust the number of channels, if necessary
                        if self.num_filters[-i] != self.num_filters[-(i - 1)]:
                            x = Convolution2D(self.num_filters[-i],
                                              (1, 1),
                                              strides=(1, 1),
                                              padding="same",
                                              data_format=self.data_format)(x)
                    else:
                        mask = None
                    unpool_name = "maxunpool_%d" \
                        % (len(self.num_conv_layers) - i,)
                    unpool_name = unpool_name + \
                        "_" + str(K.get_uid(unpool_name))
                    x = layers.MaxUnpooling2D(pool_size=(2, 2),
                                              strides=(2, 2),  # Not used
                                              padding="same",  # Not used
                                              data_format="channels_last",
                                              mask=mask,
                                              fill_zeros=True,
                                              name=unpool_name)(x)

            skip_connection = skip_connections[-(i - 1)]

            x = Concatenate(axis=self._axis)([x, skip_connection])

            for j in range(num_conv_layers_i):
                if self.use_deconvolutions:
                    x = layers.Convolution2DTranspose(
                            num_filters_i,
                            (filter_sizes_i, filter_sizes_i),
                            strides=(1, 1),
                            padding="same",
                            data_format=self.data_format)(x)
                else:
                    x = Convolution2D(num_filters_i,
                                      (filter_sizes_i, filter_sizes_i),
                                      strides=(1, 1),
                                      padding="same",
                                      data_format=self.data_format)(x)
                x = activation_function_i(x)

        # Final convolution to generate the requested output number of channels
        x = Convolution2D(self.output_channels,
                          (1, 1),  # Filter size
                          strides=(1, 1),
                          padding="same",
                          data_format=self.data_format)(x)
        outputs = x

#        # Ensure that the model takes into account
#        # any potential predecessors of `input_tensor`.
#        if self.input_tensor is not None:
#            inputs = get_source_inputs(self.input_tensor)

        # Create model
        model = Model(inputs, outputs, name=self.name)

        return model


if __name__ == "__main__":
    import doctest
    doctest.testmod()
