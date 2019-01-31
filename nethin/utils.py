# -*- coding: utf-8 -*-
"""
Contains helper functions and utility functions for use in the library.

Created on Mon Oct  9 14:05:26 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import gc
import sys
import six
import json
import base64
import numbers
import builtins
from collections.abc import Iterable
import queue

import numpy as np
import scipy.interpolate as interpolate

try:
    from pympler.asizeof import asizeof as _sizeof
    _HAS_SIZEOF = True
except (ImportError):
    try:
        from objsize import get_deep_size as _sizeof
        _HAS_SIZEOF = True
    except (ImportError):
        _HAS_SIZEOF = False

import tensorflow as tf
import keras.backend as K
import keras.engine
try:
    from keras.engine.topology import _to_snake_case as to_snake_case
except ImportError:
    from keras.engine.base_layer import _to_snake_case as to_snake_case
from keras.utils.generic_utils import serialize_keras_object, \
                                      deserialize_keras_object, \
                                      func_dump, \
                                      func_load
import keras.activations as keras_activations
import keras.layers.advanced_activations as keras_advanced_activations
from keras.layers import Activation

__all__ = ["Helper", "get_device_string", "with_device",
           "serialize_activations", "deserialize_activations",
           "to_snake_case",
           "get_json_type",
           "normalize_object", "normalize_list", "normalize_str",
           "simple_bezier", "dynamic_histogram_warping",
           "vector_median",
           "ExceedingThresholdException"]

# TODO: Make a helper module for each backend instead, as in Keras.
# TODO: Check for supported backends.

_builtin_types = [getattr(builtins, t)
                  for t in dir(builtins)
                  if isinstance(getattr(builtins, t), type)]


class TensorflowHelper(object):

    @staticmethod
    def get_scope(tensor):
        """Returns the scope of a tensor or string name of the tensor.

        If tensor.name is "Scope1/Scope2/Name:0", then this method outputs
        "Scope1/Scope2". If tensor.name is "Name:0", then this method outputs
        an empty string ("").

        Parameters
        ----------
        tensor : Tensor or str
            The Tensor or str to extract the scope of. If the Tensor does not
            have a custom scope, this method returns an empty string.
        """
        if isinstance(tensor, str):
            name = tensor
        else:
            if not hasattr(tensor, "name"):
                raise ValueError("Input does not have a name. Is it a Tensor?")

            if len(tensor.name) == 0:
                raise ValueError("Something is wrong. The Tensor does not "
                                 "have a proper name.")

            name = tensor.name

        # "Scope1/Scope2/Name:int"  => ["Scope1/Scope2/Name", "int"]
        parts = name.split(":")

        if len(parts) > 2:
            raise ValueError("The Tensor name has an unknown format!")

        # ["Scope1/Scope2/Name", "int"] => "Scope1/Scope2/Name"
        parts = parts[0]

        # "Scope1/Scope2/Name" => ["Scope1", "Scope2", "Name"]
        parts = parts.split("/")

        # ["Scope1", "Scope2", "Name"] => ["Scope1", "Scope2"]
        parts = parts[:-1]

        # ["Scope1", "Scope2"] => "Scope1/Scope2"
        scope = "/".join(parts)

        return scope

    @staticmethod
    def get_devices(device_type=None):
        """Returns a list of available computing devices.

        Warning! Uses an undocumented function "list_local_devices", that
        may break in the future.

        If broken, chech what is being used in:

            tensorflow/python/platform/test.py:is_gpu_available

        Parameters
        ----------
        device_type : str, optional
            Whether to filter the list on type of device. E.g. "CPU" or "GPU".
        """
        # TODO: We should check the validity of the device_type argument, but
        # don't know the possible device types here. Find out what they can be!

        if device_type is not None:
            device_type = str(device_type).upper()

        try:
            # Adapted from: https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
            # Note the comment by Yaroslav Bulatov: "PS, if this method ever
            # gets moved/renamed, I would look inside
            # tensorflow/python/platform/test.py:is_gpu_available since that's
            # being used quite a bit."

            from tensorflow.python.client import device_lib

            local_devices = device_lib.list_local_devices()

            if device_type is not None:
                return [d.name for d in local_devices
                        if d.device_type == device_type]
            else:
                return [d.name for d in local_devices]
        except:
            # Adapted from: https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell

            local_devices = []

            # Try to use the cpu(s), and return available devices.
            try:
                dev_num = 0
                while True:
                    device_name = "/cpu:%d" % dev_num
                    with tf.device(device_name):
                        with tf.name_scope("TensorflowHelper"):
                            a = tf.constant([1, 2], shape=[1, 2],
                                    name="get_devices.cpu_a%d" % dev_num)
                            b = tf.constant([3, 4], shape=[2, 1],
                                    name="get_devices.cpu_b%d" % dev_num)
                            c = tf.matmul(a, b,
                                    name="get_devices.cpu_b%d" % dev_num)
                    # # if self._session is None:
                    # with tf.Session() as sess:
                    #     sess.run(c)  # Try to use the device
                    # # else:
                    # #     with tf.Session(graph=self.get_graph()) as sess:
                    # #         sess.run(c)  # Try to use the device
                    with K.get_session() as sess:
                        sess.run(c)  # Try to use the device

                    local_devices.append(device_name)
                    dev_num += 1

            except:
                pass

            # Try to use the gpu(s), and return available devices.
            try:
                dev_num = 0
                while True:
                    device_name = "/gpu:%d" % dev_num
                    with tf.device(device_name):
                        with tf.name_scope("TensorflowHelper/"):  # Reuse scope
                            a = tf.constant([1, 2], shape=[1, 2],
                                    name="get_devices.gpu_a%d" % dev_num)
                            b = tf.constant([3, 4], shape=[2, 1],
                                    name="get_devices.gpu_b%d" % dev_num)
                            c = tf.matmul(a, b,
                                    name="get_devices.gpu_b%d" % dev_num)
                    # # if self._session is None:
                    # with tf.Session() as sess:
                    #     sess.run(c)  # Try to use the device
                    # # else:
                    # #     with tf.Session(graph=self.get_graph()) as sess:
                    # #         sess.run(c)  # Try to use the device
                    with K.get_session() as sess:
                        sess.run(c)  # Try to use the device

                    local_devices.append(device_name)
                    dev_num += 1

            except:
                pass

            # TODO: Other device types??

            return local_devices


Helper = TensorflowHelper


def get_device_string(cpu=None, gpu=None, num=0):

    if (cpu is None) and (gpu is None):
        gpu = True

    if cpu == gpu:
        raise ValueError('Both "cpu" and "gpu" cannot be True (or False) at '
                         'the same time.')

    num = int(num)

    if cpu:
        return "/device:CPU:" + str(num)
    if gpu:
        return "/device:GPU:" + str(num)


def with_device(__device, function, *args, **kwargs):
    """Run a given function (with given arguments) on a particular device.

    Parameters
    ----------
    device : None or str
        The device to run/construct the function/object on. None means to
        run/construct it on the default device (usually "/gpu:0"). See
        ``nethin.utils.Helper.get_devices()`` for the list of your available
        devices.

    function
        The function or class to run/construct.

    args : list, optional
        The list of arguments to ``function``.

    kwargs : list, optional
        The list of keyword arguments to ``function``.
    """
    if __device is None:
        ret = function(*args, **kwargs)
    else:
        with tf.device(__device):
            ret = function(*args, **kwargs)

    return ret


def serialize_activations(activations):

    if activations is None:
        return activations

    def serialize_one(activation):
        if isinstance(activation, six.string_types):
            return activation

        if isinstance(activation, keras.engine.Layer):  # Advanced activation
            return serialize_keras_object(activation)

        # The order matters here, since Layers are also callable.
        if callable(activation):  # A function
            return func_dump(activation)

        # Keras serialized config
        if isinstance(activation, dict) \
                and "class_name" in activation \
                and "config" in activation:
            return activation

        # Could be a marshalled function
        if isinstance(activation, (list, tuple)) \
                and len(activation) == 3 \
                and isinstance(activation[0], six.string_types):
            try:
                # TODO: Better way to check if it is a marshalled function!
                func_load(activation)  # Try to unmarshal it

                return activation

            except ValueError:
                pass

        return None

    one = serialize_one(activations)  # See if it is only one

    if (one is None) and isinstance(activations, (list, tuple)):

        _activations = []
        for activation in activations:

            one = serialize_one(activation)
            if one is not None:
                _activations.append(one)
            else:
                raise ValueError("Unable to serialize activation functions.")

        return _activations

    elif one is not None:
        return one

    else:
        raise ValueError("Unable to serialize activation functions.")


def deserialize_activations(activations, length=None, device=None):
    """Deserialize activation functions.

    Parameters
    ----------
    activations : object or list
        Serialized activation, or list of serialized activations.

    length : int, optional
        The normalized or expected length of the output list. If not None, the
        length of the input ``activations`` is compared to this number. Also,
        if an object instead of a list as input, the list will be made this
        long. Default is None, which means to not normalize the outputs.

    device : str, optional
        A particular device to create the activations on. Default is ``None``,
        which means to create on the default device (usually the first GPU
        device). Use ``nethin.utils.Helper.get_device()`` to see available
        devices.
    """
    if length is not None:
        length = max(1, int(length))

    def deserialize_one(activation):

        # Simple activation
        if (activation is None) or isinstance(activation, six.string_types):
            return with_device(device, Activation, activation)

        # Advanced activation (it has already been created, nothing we can do)
        if isinstance(activation, keras.engine.Layer):
            return activation

        # Function (it has already been created, nothing we can do)
        if callable(activation):
            return activation

        # Keras serialized config
        if isinstance(activation, dict) \
                and "class_name" in activation \
                and "config" in activation:

            # Make advanced activation functions available per default
            if activation["class_name"] in dir(keras_advanced_activations):
                custom_objects = {}
                class_name = activation["class_name"]
                for attr in dir(keras_advanced_activations):
                    if class_name == attr:
                        layer = keras_advanced_activations.__dict__[class_name]
                        custom_objects[class_name] = layer
                        break

            return with_device(device,
                               keras_activations.deserialize,
                               activation,
                               custom_objects=custom_objects)

        # Could be a marshalled function
        if isinstance(activation, (list, tuple)) \
                and len(activation) == 3 \
                and isinstance(activation[0], six.string_types):
            try:
                # TODO: Better way to check if it is a marshalled function!
                return func_load(activation)  # Try to unmarshal it

            except EOFError:
                pass  # "marshal data too short" => Not a marshalled function

            except ValueError:
                pass  # ??

        return None

    one = deserialize_one(activations)

    if one is not None:
        if length is None:
            return one
        else:
            _activations = [one]
            for i in range(1, length):
                one = deserialize_one(activations)
                _activations.append(one)
            return _activations
    else:

        if length is not None:
            if length != len(activations):
                raise ValueError("The number of activations does not match "
                                 "length.")

        _activations = []
        for activation in activations:

            one = deserialize_one(activation)

            if one is None:
                raise ValueError("Unable to deserialize activation functions.")

            _activations.append(one)

        return _activations


def serialize_array(array):
    """Serialise a numpy array to a JSON string.

    Adapted from:
        https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions

    Parameters
    ----------
    array : numpy array
        The numpy array to serialise.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> array = np.random.rand(3, 4)
    >>> string = utils.serialize_array(array)
    >>> string  # doctest: +ELLIPSIS
    '["<f8", "7FFf...Ce8/", [3, 4]]'
    >>> array2 = utils.deserialize_array(string)
    >>> np.linalg.norm(array - array2)
    0.0
    """
    string = json.dumps([array.dtype.str,
                         base64.b64encode(array).decode("utf-8"),
                         array.shape])
    return string


def deserialize_array(string):
    """De-serialise a string representation (JSON) to a numpy array.

    Adapted from:
        https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions

    Parameters
    ----------
    string : str
        JSON representation (created using ``serialize_array``) to create numpy
        array from.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> array = np.random.rand(3, 4)
    >>> string = utils.serialize_array(array)
    >>> string  # doctest: +ELLIPSIS
    '["<f8", "7FFf...Ce8/", [3, 4]]'
    >>> array2 = utils.deserialize_array(string)
    >>> np.linalg.norm(array - array2)
    0.0
    """
    dtype, array, shape = json.loads(string)
    dtype = np.dtype(dtype)
    array = np.frombuffer(base64.b64decode(array), dtype)
    array = array.reshape(shape)

    return array


def get_json_type(obj):
    """Serialize any object to a JSON-serializable structure.

    Provided since Keras has made the corresponding function an inner function
    of ``keras.models.save_model``.

    Parameters
    ----------
    obj : object
        The object to serialize.

    Returns
    -------
        JSON-serializable structure representing ``obj``.

    Raises
    ------
        TypeError: if ``obj`` cannot be serialized.
    """
    # TODO: Keep up-to-date with Keras!

    # if obj is a serializable Keras class instance
    # e.g. optimizer, layer
    if hasattr(obj, 'get_config'):
        return {'class_name': obj.__class__.__name__,
                'config': obj.get_config()}

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return {'type': type(obj),
                    'value': obj.tolist()}
        else:
            return obj.item()

    # misc functions (e.g. loss function)
    if callable(obj):
        return obj.__name__

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable:', obj)


def normalize_object(obj, n, name):
    """Transforms a single object or iterable of objects into an object tuple.

    Parameters
    ----------
    obj : object
        The value to validate. An object, or any iterable of objects. Can be
        anything, but note that tuples and other iterables will only be checked
        that they have the length ``n``, and will thus not be normalised.

    n : int
        The length of the tuple to return.

    name : str
        The name of the argument being validated, e.g. "activations" or
        "optimizers". This is only used to format error messages.
    """
    n = max(1, int(n))

    if isinstance(obj, str):
        return (obj,) * n

    if hasattr(obj, "__iter__"):
        try:
            obj = tuple(obj)
        except TypeError:
            raise ValueError('The "' + name + '" argument must be a tuple '
                             'of ' + str(n) + ' integers. Received: ' +
                             str(obj))

        if len(obj) != n:
            raise ValueError('The "' + name + '" argument must be a tuple '
                             'of ' + str(n) + ' integers. Received: ' +
                             str(obj))
    else:
        return (obj,) * n

    return obj


def normalize_list(lst, n, name):
    """Transforms a single list or iterable of lists into a tuple of lists.

    Parameters
    ----------
    lst : list or iterable of lists
        The value to validate. A list, or any iterable of lists. If a list,
        will generate a tuple of ``n`` shallow copies of that lists.

    n : int
        The length of the tuple to return.

    name : str
        The name of the argument being validated, e.g. "activations" or
        "optimizers". This is only used to format error messages.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> utils.normalize_list([1, 2, 3], 2, "list_input2")
    [[1, 2, 3], [1, 2, 3]]
    >>> utils.normalize_list([1, 2, 3], 3, "list_input3")
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    >>> utils.normalize_list(([1], [2], [3]), 3, "tuple_input")
    ([1], [2], [3])
    """
    n = max(1, int(n))

    if isinstance(lst, list):
        return [list(lst) for i in range(n)]

    try:
        lst_tuple = tuple(lst)
    except TypeError:
        raise ValueError('The "' + name + '" argument must be a tuple '
                         'of ' + str(n) + ' integers. Received: ' +
                         str(lst))

    if len(lst_tuple) != n:
        raise ValueError('The "' + name + '" argument must be a tuple '
                         'of ' + str(n) + ' integers. Received: ' +
                         str(lst))

    for i in range(len(lst_tuple)):
        if not isinstance(lst_tuple[i], list):
            raise ValueError('The "' + name + '" argument must be a tuple '
                             'of ' + str(n) + ' lists. Received: ' +
                             str(lst) + ' including element ' +
                             str(lst_tuple[i]) + ' of type ' +
                             str(type(lst_tuple[i])))

    return lst_tuple


def normalize_str(value, n, name):
    """Transforms a single str or iterable of str into a tuple of str.

    Parameters
    ----------
    value : str or iterable of str
        The value to validate. A str, or any iterable of str. If a str, will
        generate a tuple of ``n`` copies of that str.

    n : int
        The length of the tuple to return.

    name : str
        The name of the argument being validated, e.g. "activations" or
        "optimizers". This is only used to format error messages.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> utils.normalize_str("a", 2, "str_input")
    ('a', 'a')
    >>> utils.normalize_str(["a", "b"], 2, "list_input")
    ('a', 'b')
    >>> utils.normalize_str(("a", "c"), 2, "tuple_input")
    ('a', 'c')
    """
    n = max(1, int(n))

    if isinstance(value, str):
        return (value,) * n

    try:
        value_tuple = tuple(value)
    except TypeError:
        raise ValueError('The "' + name + '" argument must be a tuple '
                         'of ' + str(n) + ' str. Received: ' +
                         str(value))

    if len(value_tuple) != n:
        raise ValueError('The "' + name + '" argument must be a tuple '
                         'of ' + str(n) + ' integers. Received: ' +
                         str(value))

    for i in range(len(value_tuple)):
        if not isinstance(value_tuple[i], str):
            raise ValueError('The "' + name + '" argument must be a tuple '
                             'of ' + str(n) + ' strings. Received: ' +
                             str(value) + ' including element ' +
                             str(value_tuple[i]) + ' of type ' +
                             str(type(value_tuple[i])))

    return value_tuple


def normalize_random_state(random_state, rand_functions=[]):
    """Tests a given random_state argument.

    Parameters
    ----------
    random_state : int, float, array_like or numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If int,
        float or array_like, a new random state is created with the provided
        value as seed. If None, the default numpy random state (np.random) is
        used. Default is None, use the default numpy random state.

    rand_functions : list of str, optional
        If given, will consider the given object as a valid random number
        generator if it exposes the methods given in ``rand_functions``. I.e.,
        ``random_state=np.random`` and ``rand_functions=["rand", "randn"]`` is
        valid, but ``np.random`` and ``rand_functions=["foo", "bar"]`` is not.
        If not valid, will try to use the given object as a seed for a new
        random number generator. Will crash if this does not work.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> import numpy as np
    >>> utils.normalize_random_state(np.random, rand_functions=["rand"])  # doctest: +ELLIPSIS
    <module 'numpy.random' from ...>
    >>> utils.normalize_random_state(np.random, rand_functions=["foo"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: Cannot cast array from dtype('O') to dtype('int64') according to the rule 'safe'
    >>> utils.normalize_random_state(42)  # doctest: +ELLIPSIS
    <mtrand.RandomState ...>
    """
    if random_state is None:
        random_state = np.random.random.__self__
    else:
        if isinstance(random_state, (int, float, np.ndarray)):
            random_state = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            random_state = random_state
        elif len(rand_functions) > 0:
            has_all = True
            for rand_function in rand_functions:
                if not hasattr(random_state, rand_function):
                    has_all = False
                    break
            if has_all:
                random_state = random_state
            else:  # May crash here..
                random_state = np.random.RandomState(seed=random_state)
        else:  # May crash here..
            random_state = np.random.RandomState(seed=random_state)

    return random_state


def simple_bezier(dist,
                  controls=[0.25, 0.5, 0.75],
                  steps=256,
                  interp=None,
                  interp_kwargs=None):
    """Constructs a Bézier curve using the given control points along the
    identity line and their distance from the identity line.

    The generated curve will look something like:

        ^
       1|              ./
        |            . /
        |          .  /
        |     ______,'
        |   ,' .
        |  / .
        | /.
        |/
       0+---------------->
        0               1

    The line represents the Bézier curve and the dotted line the identity line.
    The ``dist`` vector in this case had values ``[d, 0, -d]``, for some ``d``.

    Parameters
    ----------
    dist : list of float, length 3
        The distance from the control points to the identity line. Positive
        values means above the identity line, and negative values mean below
        the identity line.

    controls : list of float, length 3, optional
        Floats in [0, 1] in increasing order (ideally, but not strictly
        required). Three control points along the identity line. [0, 0] and
        [1, 1] are also control points. Default is [0.25, 0.5, 0.75].

    steps : int, optional
        The number of point computed on the Bézier curve for use in the
        interpolation. Default is 256, in order to capture all pixel changes in
        a standard ``uint8`` image.

    interp : Callable, optional
        Any callable function that takes as arguments x and y, the coordinates
        of the interpolation points. Default is None, which means to use
        ``scipy.interpolate.PchipInterpolator``. This is a good default
        interpolator when the function is monotonic.

    interp_kwargs : dict, optional
        Keyword arguments, if any, to pass to the interpolator function.
        Default is ``None``, which means to pass ``extrapolate=True`` if
        ``interp`` is ``None`` (i.e., ``scipy.interpolate.PchipInterpolator``),
        and not to pass any arguments otherwise.

    Returns
    -------
    Callable
        The Bézier function computed from the input. It takes values in [0, 1]
        and (ideally) returns values in [0, 1] (the output depends on the input
        distances from the identity line, however).
    """
    assert(len(dist) == len(controls))

    if interp is None:
        interp = interpolate.PchipInterpolator

        if interp_kwargs is None:
            interp_kwargs = dict(extrapolate=True)

    if interp_kwargs is not None:
        if not isinstance(interp_kwargs, dict):
            raise ValueError("``interp_kwargs`` must be either ``None`` or "
                             "a ``dict``.")

    v = np.array([-1.0, 1.0])
    v /= np.linalg.norm(v)

    P0 = np.array([0.0, 0.0])
    P1 = v * dist[0] + np.array([controls[0], controls[0]])
    P2 = v * dist[1] + np.array([controls[1], controls[1]])
    P3 = v * dist[2] + np.array([controls[2], controls[2]])
    P4 = np.array([1.0, 1.0])

    def _bezier_func(t, P):

        x = P[0][0] * 1.0 * (1.0 - t)**4.0 * t**0.0 \
          + P[1][0] * 4.0 * (1.0 - t)**3.0 * t**1.0 \
          + P[2][0] * 6.0 * (1.0 - t)**2.0 * t**2.0 \
          + P[3][0] * 4.0 * (1.0 - t)**1.0 * t**3.0 \
          + P[4][0] * 1.0 * (1.0 - t)**0.0 * t**4.0
        y = P[0][1] * 1.0 * (1.0 - t)**4.0 * t**0.0 \
          + P[1][1] * 4.0 * (1.0 - t)**3.0 * t**1.0 \
          + P[2][1] * 6.0 * (1.0 - t)**2.0 * t**2.0 \
          + P[3][1] * 4.0 * (1.0 - t)**1.0 * t**3.0 \
          + P[4][1] * 1.0 * (1.0 - t)**0.0 * t**4.0

        return x, y

    pts = []
    for t in np.linspace(0, 1, steps):
        x, y = _bezier_func(t, [P0, P1, P2, P3, P4])
        pts.append([x, y])
    pts = np.array(pts)

    if interp_kwargs is not None:
        func = interp(pts[:, 0], pts[:, 1], **interp_kwargs)
    else:
        func = interp(pts[:, 0], pts[:, 1])

    return func


def dynamic_histogram_warping(I1,
                              I2,
                              bins=256,
                              max_compression1=16,
                              max_compression2=16,
                              perform_transform=True,
                              return_cost=False):
    """Computes the dynamic histogram warping function of Cox, Roy and
    Hingorani (1995) [1]_.

    Parameters
    ----------
    I1 : numpy.ndarray, dim 2
        The first (template) image.

    I2 : numpy.ndarray, dim 2
        The second image to be matched to ``I1``.

    bins : int, optional
        The number of histogram bins. Default is 256, in order to capture all
        intensities in a standard ``uint8`` image.

    max_compression1 : int, optional
        The maximum allowed compression of the histogram of image 1 (I1).
        Default is 16.

    max_compression2 : int, optional
        The maximum allowed compression of the histogram of image 2 (I2).
        Default is 16.

    perform_transform : bool, optional
        Actually perform and return the histogram warped version of image 2
        (I2). Default is True. If false, no image is returned.

    return_cost : bool, optional
        Return the cost of warping the histograms of I1 and I2. Default is
        False. If True, will return the cost as well.

    Returns
    -------
    warped_image, total_cost : If ``perform_transform=True`` and
        ``return_cost=True``.

    warped_image : If ``perform_transform=True`` and ``return_cost=False``.

    total_cost : If ``perform_transform=False`` and ``return_cost=True``.

    None : If ``perform_transform=False`` and ``return_cost=False``.

    References
    ----------
    .. [1] I. J. Cox, S. Roy and S. L. Hingorani (1995). "Dynamic Histogram
       Warping of Image Pairs for Constant Image Brightness". Proceedings of
       the International Conference on Image Processing, 23-26 October in
       Washington, DC, USA.
    """
    bins = max(1, int(bins))

    A, bins_A = np.histogram(I1.ravel(), bins=bins)
    B, bins_B = np.histogram(I2.ravel(), bins=bins)

    AC = np.cumsum(A)
    BC = np.cumsum(B)

    num_bins_A = len(A)
    num_bins_B = len(B)

    D = np.zeros((num_bins_A, num_bins_B))
    M = max(1, int(max_compression1))
    N = max(1, int(max_compression2))

    # Compute the cost function
    for m in range(num_bins_A):
        for n in range(num_bins_B):
            if m == 0 and n == 0:
                D[m, n] = 0.0
            elif m == 0 or n == 0:
                D[m, n] = np.inf
            else:
                case1 = D[m - 1, n - 1] + np.abs(A[m] - B[n])
                case2 = np.inf
                for k in range(2, M + 1):
                    if m - k < 0:
                        break
                    case2 = min(case2, D[m - k, n - 1] \
                            + abs((AC[m] - AC[m - k]) - B[n]))
                case3 = np.inf
                for l in range(2, N + 1):
                    if n - l < 0:
                        break
                    case3 = min(case3, D[m - 1, n - l] \
                            + abs(A[m] - (BC[n] - BC[n - l])))
                D[m, n] = min(case1, case2, case3)

    # Find least-cost path through the cost space
    m = num_bins_A - 1
    n = num_bins_B - 1
    x_A = [bins_A[-1]]
    x_B = [bins_B[-1]]
    total_cost = D[m, n]
    while (m > 0 and n > 0):
        x_A.insert(0, (bins_A[m] + bins_A[m + 1]) / 2.0)
        x_B.insert(0, (bins_B[n] + bins_B[n + 1]) / 2.0)
    
        min_m = m
        min_n = n
        # Select neighbour with lowest cost
        if D[m - 1, n - 1] < D[m - 1, n] and D[m - 1, n - 1] < D[m, n - 1]:
            min_m = m - 1
            min_n = n - 1
        elif D[m - 1, n] < D[m - 1, n - 1] and D[m - 1, n] < D[m, n - 1]:
            min_m = m - 1
            min_n = n
        elif D[m, n - 1] < D[m - 1, n - 1] and D[m, n - 1] < D[m - 1, n]:
            min_m = m
            min_n = n - 1
        else:  # At least two costs were equally low: Prefer to be closer to
               # the diagonal in this case (i.e., choose the one closest to the
               # diagonal).
            if D[m - 1, n - 1] == D[m - 1, n] \
                    and D[m - 1, n - 1] == D[m, n - 1]:  # All equal:
                if (float(m) / float(num_bins_A)) < (float(n) / float(num_bins_B)):
                    min_m = m
                    min_n = n - 1
                elif (float(m) / float(num_bins_A)) > (float(n) / float(num_bins_B)):
                    min_m = m - 1
                    min_n = n
                else:
                    min_m = m - 1
                    min_n = n - 1
            elif D[m - 1, n - 1] == D[m - 1, n]: # The two above are equal:
                if (float(m) / float(num_bins_A)) < (float(n) / float(num_bins_B)):
                    min_m = m - 1
                    min_n = n - 1
                else:  # (float(m) / float(num_bins_A)) > (float(n) / float(num_bins_B)):
                    min_m = m - 1
                    min_n = n
            elif D[m - 1, n - 1] == D[m, n - 1]: # The two to the left are equal:
                if (float(m) / float(num_bins_A)) < (float(n) / float(num_bins_B)):
                    min_m = m
                    min_n = n - 1
                else:  # (float(m) / float(num_bins_A)) > (float(n) / float(num_bins_B)):
                    min_m = m - 1
                    min_n = n - 1
            else:  # D[m - 1, n] == D[m, n - 1]: # The one to the left and the one above are equal:
                if (float(m) / float(num_bins_A)) < (float(n) / float(num_bins_B)):
                    min_m = m
                    min_n = n - 1
                else:  # (float(m) / float(num_bins_A)) > (float(n) / float(num_bins_B)):
                    min_m = m - 1
                    min_n = n
        m = min_m
        n = min_n
        # print("%d %d: %f" % (m, n, D[m, n]))
        total_cost += D[m, n]
    x_A.insert(0, bins_A[0])
    x_B.insert(0, bins_B[0])

    if perform_transform:
        # Use linear interpolation between bin centers.
        x_A_ = np.array(x_A)
        x_B_ = np.array(x_B)
        VB = I2.ravel()
        VB_ = np.zeros_like(VB)
        js = np.searchsorted(x_B_, VB)  # Assuming a monotonic transform

        lt = VB <= x_B[0]
        gt = VB >= x_B[-1]
        VB_[lt] = x_A[0]
        VB_[gt] = x_A[-1]

        between = np.logical_and(np.logical_not(lt), np.logical_not(gt))
        diff = x_B_[js - 1] != x_B_[js]
        steps = np.zeros_like(VB_)

        case1 = np.logical_and(between, diff)
        js_case1 = js[case1]
        x_B_js_case1_1 = x_B_[js_case1 - 1]
        steps[case1] = np.divide(VB[case1] - x_B_js_case1_1,
                                 x_B_[js_case1] - x_B_js_case1_1)

        case2 = np.logical_and(between, np.logical_not(diff))
        steps[case2] = 0.5

        js_between = js[between]
        x_A_js_between_1 = x_A_[js_between - 1]
        VB_[between] = x_A_js_between_1 \
                         + np.multiply(steps[between],
                                       x_A_[js_between] - x_A_js_between_1)

#        for i in range(len(VB_)):
#            value = VB[i]
#
#            if value <= x_B[0]:
#                mapped_value = x_A[0]
#            elif value >= x_B[-1]:
#                mapped_value = x_A[-1]
#            else:
#                # TODO: Use vectorized functions here
#                # j = np.argmax(value < x_B_)
#                # j = np.searchsorted(x_B_, value)
#                # assert(j == js[i])
#                j = js[i]
#
#                if x_B[j - 1] != x_B[j]:
#                    step = (value - x_B[j - 1]) / (x_B[j] - x_B[j - 1])
#                else:
#                    step = 0.5
#
#                mapped_value = x_A[j - 1] + step * (x_A[j] - x_A[j - 1])
#
#            VB_[i] = mapped_value

        VB_ = np.reshape(VB_, I2.shape)

        if return_cost:
            return VB_, total_cost
        else:
            return VB_
    else:
        if return_cost:
            return total_cost
        else:
            return None


def histogram_matching(A, B, return_cost=False, num_cost_interp=100):
    """Performs histogram matchning (specification) from one image to another.

    Adapted from:
        https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Parameters
    ----------
    A : numpy.ndarray
        The first (template) image.

    B : numpy.ndarray
        The second image to be matched to ``A``.

    return_cost : bool, optional
        Quantify the cost of transforming the histogram of ``B`` to match that
        of ``A``. The cost is computed as the integral of the difference of the
        inverse cumulative distribution functions, i.e., the "sum" of vertical
        distances between the histograms. Default is False. If True, will
        return the cost as well.

    num_cost_interp : int, optional
        The cost function is computed numerically as a Riemann sum. The
        ``num_cost_interp`` determines the number of discrete steps in the
        Riemann approximation. Default is 100.

    Returns
    -------
    matched_B, cost : If ``return_cost=True``.

    matched_B : If ``return_cost=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.misc import face, ascent, imresize
    >>> face = np.mean(face(), axis=2)
    >>> ascent = ascent()
    >>> try:
    ...     from skimage.transform import resize
    ...     face = resize(face, [512, 512], mode="reflect")
    ...     ascent = resize(ascent, [512, 512], mode="reflect")
    ... except:
    ...     shape = face.shape
    ...     idx = np.floor(shape[0] * np.linspace(0.0, 1.0 - np.finfo('float').eps, 512)).astype(np.int32)
    ...     face = face[idx, :]
    ...     idx = np.floor(shape[1] * np.linspace(0.0, 1.0 - np.finfo('float').eps, 512)).astype(np.int32)
    ...     face = face[:, idx]
    >>> import nethin.utils as utils
    >>> A = face.astype(np.float64)
    >>> B = ascent.astype(np.float64)
    >>> hA = np.histogram(A.ravel(), bins=10)[0]
    >>> hB = np.histogram(B.ravel(), bins=10)[0]
    >>> np.sum(np.abs(hA - hB)) < 131000
    True
    >>> B_ = np.round(utils.histogram_matching(A, B))
    >>> hB_ = np.histogram(B_.ravel(), bins=10)[0]
    >>> np.sum(np.abs(hB - hB_)) < 136000
    True
    >>> np.sum(np.abs(hA - hB_)) < 7200
    True

    References
    ----------
    .. [1] R. C. Gonzalez and R. E. Woods (2008). "Digital Image Processing",
       third edition. Pearson Education, Pearson Prentice Hall, Upper Saddle
       River, New Jersey, USA.
    """
    B_shape = B.shape
    A = A.ravel()
    B = B.ravel()

    B_values, bin_idx, B_counts = np.unique(B,
                                            return_inverse=True,
                                            return_counts=True)
    A_values, A_counts = np.unique(A, return_counts=True)

    B_quantiles = np.cumsum(B_counts).astype(np.float64)
    B_quantiles /= B_quantiles[-1]
    A_quantiles = np.cumsum(A_counts).astype(np.float64)
    A_quantiles /= A_quantiles[-1]

    interp_A_values = np.interp(B_quantiles, A_quantiles, A_values)
    matched_B = interp_A_values[bin_idx].reshape(B_shape)

    if return_cost:
        B_norm = B_values / np.max(B_values)
        A_norm = A_values / np.max(A_values)

        ind_B = np.floor(B_norm.size * np.linspace(0.0, 1.0 - np.finfo(np.float64).eps, num_cost_interp)).astype(np.int32)
        interp_B = B_norm[ind_B]
        ind_A = np.floor(A_norm.size * np.linspace(0.0, 1.0 - np.finfo(np.float64).eps, num_cost_interp)).astype(np.int32)
        interp_A = A_norm[ind_A]

        dx = 1.0 / float(num_cost_interp - 1)
        cost = np.sum(np.abs(interp_B - interp_A)) * dx

        return matched_B, cost
    else:
        return matched_B


# TODO: This function is really slow. Speed up!
def vector_median(vf, window=1, order=2):
    """Computes the vector median of an n-dimensional vector field.

    Parameters
    ----------
    vf : ndarray
        The vector field to smooth. It should have the shape ``(dim1, dim2, ...
        dimn, n)``, where the first dimensions are the spatial dimensions of
        the image, and the last dimension contains the vector dimensions.

    window : int, optional
        Positive integer. The vector median is computed in a window of size
        ``2 * window + 1``.

    order : {non-zero int, np.inf, -np.inf}, optional
        Order of the norm (see numpy.linalg.norm for details). If ``order`` is
        negative, you may encounter a RuntimeWarning message if there are
        zero-differences among the vectors.

    Examples
    --------
    >>> import nethin.utils as utils
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> vf = np.random.rand(5, 5, 2)  # doctest: +NORMALIZE_WHITESPACE
    >>> vf[..., 0]
    array([[0.37454012, 0.73199394, 0.15601864, 0.05808361, 0.60111501],
           [0.02058449, 0.83244264, 0.18182497, 0.30424224, 0.43194502],
           [0.61185289, 0.29214465, 0.45606998, 0.19967378, 0.59241457],
           [0.60754485, 0.06505159, 0.96563203, 0.30461377, 0.68423303],
           [0.12203823, 0.03438852, 0.25877998, 0.31171108, 0.54671028]])
    >>> new_vf = utils.vector_median(vf,
    ...                              window=1,
    ...                              order=1)  # doctest: +NORMALIZE_WHITESPACE
    >>> new_vf[..., 0]
    array([[0.37454012, 0.18182497, 0.30424224, 0.30424224, 0.30424224],
           [0.29214465, 0.29214465, 0.30424224, 0.30424224, 0.30424224],
           [0.29214465, 0.29214465, 0.30424224, 0.43194502, 0.43194502],
           [0.12203823, 0.25877998, 0.25877998, 0.31171108, 0.54671028],
           [0.12203823, 0.25877998, 0.25877998, 0.31171108, 0.54671028]])
    >>> new_vf = utils.vector_median(vf,
    ...                              window=1,
    ...                              order=2)  # doctest: +NORMALIZE_WHITESPACE
    >>> new_vf[..., 0]
    array([[0.73199394, 0.18182497, 0.30424224, 0.30424224, 0.30424224],
           [0.73199394, 0.29214465, 0.30424224, 0.30424224, 0.30424224],
           [0.60754485, 0.29214465, 0.30424224, 0.43194502, 0.43194502],
           [0.29214465, 0.25877998, 0.25877998, 0.31171108, 0.54671028],
           [0.03438852, 0.25877998, 0.25877998, 0.31171108, 0.54671028]])
    """
    window = max(0, abs(int(window)))

    if isinstance(order, int):
        order = int(order)
    elif isinstance(order, float) and (order not in {np.inf, -np.inf}):
        order = int(order)
    elif order not in {np.inf, -np.inf, "fro", "nuc"}:
        raise ValueError("``order`` should be one of non-zero int, np.inf, "
                         "-np.inf}.")

    ndim = len(vf.shape) - 1

    assert(ndim == vf.shape[-1])

    grid = np.linspace(-window, window, 2 * window + 1)
    mesh = np.meshgrid(*(grid for i in range(ndim)))
    for i in range(len(mesh)):
        mesh[i] = mesh[i][..., np.newaxis]
    mesh = np.concatenate(mesh, axis=ndim)

    shape = vf.shape[:-1]

    slices = tuple([slice(None, None) for i in range(ndim)]
                   + [0])
    slices_w = tuple([slice(None, None) for i in range(mesh.ndim - 1)]
                     + [0])

    new_vf = np.zeros_like(vf)
    for coord, _ in np.ndenumerate(vf[slices]):
        m = np.array(coord)
        vectors = []
        for c, _ in np.ndenumerate(mesh[slices_w]):
            d = mesh[c[0], c[1]]
            md = m + d
            inside = True
            for i in range(md.shape[0]):
                if md[i] < 0.0:
                    inside = False
                elif md[i] >= shape[0]:
                    inside = False

            if inside:
                vectors.append(vf[tuple((md + 0.5).astype(int).tolist()
                               + [slice(None)])])

        assert(len(vectors) > 0)

        min_d = np.inf
        new_vector = vectors[0]
        for v1_i in range(len(vectors)):
            v1 = vectors[v1_i]
            d_i = 0.0
            for v2_i in range(len(vectors)):
                v2 = vectors[v2_i]

                d_i += np.linalg.norm(v1 - v2, ord=order)

            if d_i < min_d:
                new_vector = v1
                min_d = d_i

        new_vf[tuple(list(coord) + [slice(None)])] = new_vector

    return new_vf


def sizeof(o, use_external=True):
    r"""Attempts to determine the size in bytes of the provided object.

    If you have Pymbler or objsize (both available through pip) installed, this
    function will use them directly (per default, and when
    ``use_external=True``). Results may therefore depend on your installed
    packages.

    Parameters
    ----------
    o : object
        The object for which the size should be determined.

    use_external : bool, optional
        Whether or not to use the external size checkers instead of the
        build-in one provided here. The external packages are either Pympler
        (tested first), or objsize (tested second). Default is True, use the
        external size packages.

    Examples
    --------
    >>> from sys import getsizeof
    >>> from nethin.utils import sizeof
    >>> import numpy as np
    >>> getsizeof(1) == sizeof(1, use_external=False)
    True
    >>> getsizeof(1.0) == sizeof(1.0, use_external=False)
    True
    >>> getsizeof("test") == sizeof("test", use_external=False)
    True
    >>> a = np.zeros((3, 3), dtype=np.float32)
    >>> getsizeof(a) == sizeof(a, use_external=False)
    True
    >>> a = np.zeros((3, 3), dtype=np.float64)
    >>> getsizeof(a) == sizeof(a, use_external=False)
    True
    """
    if _HAS_SIZEOF and use_external:
        return _sizeof(o)

    size = 0
    seen = set()
    q = queue.Queue()
    q.put(o)
    while not q.empty():
        o = q.get()
        _id = id(o)
        if (_id not in seen) and (not isinstance(o, type)):
            seen.add(_id)
            if isinstance(o, (bool, int, float, complex, numbers.Number)):
                s = sys.getsizeof(o)
                print("Basic: %d" % (s,))
                size += s
            elif isinstance(o, (str, bytearray, bytes)):
                s = sys.getsizeof(o)
                print("String: %d" % (s,))
                size += s
            else:
                size += sys.getsizeof(o)
                for o_ in gc.get_referents(o):
                    q.put(o_)

    return size


class RangeType(object):

    def __init__(self, start=None, stop=None, dtype=float, size=None):

        if start is None:
            self.start = None
        else:
            if dtype is int:
                self.start = min(int(start), int(stop))
            else:
                self.start = min(float(start), float(stop))
        if stop is None:
            self.stop = None
        else:
            if dtype is int:
                self.stop = max(int(start), int(stop))
            else:
                self.stop = max(float(start), float(stop))

        if dtype is None:
            self.dtype = None
        else:
            assert(dtype is float or dtype is int or dtype is bool)
            self.dtype = dtype

        if size is None:
            self.size = None
        else:
            self.size = max(0, int(size))


class UniformRange(RangeType):

    def __init__(self, start, stop, dtype=float, size=None):

        super(UniformRange, self).__init__(start, stop, dtype=dtype, size=size)

    def get_random(self):

        ret = []
        for i in range(1 if self.size is None else self.size):
            if self.dtype is int:
                ret.append(np.random.randint(self.start, self.stop + 1))
            else:  # self.dtype is float
                ret.append(
                    np.random.rand() * (self.stop - self.start) + self.start)

        if self.size is None:
            ret = ret[0]

        return ret


class LogRange(RangeType):

    def __init__(self, start, stop, dtype=float, base=10, size=None):

        super(LogRange, self).__init__(start, stop, dtype=dtype, size=size)

        self.base = float(base)

    def get_random(self):

        ret = []
        for i in range(1 if self.size is None else self.size):
            if self.dtype is int:
                rand = np.random.randint(self.start, self.stop + 1)
                ret.append(int((self.base**rand) + 0.5))
            else:
                rand = np.random.rand() * (self.stop - self.start) + self.start
                ret.append(self.base**rand)

        if self.size is None:
            ret = ret[0]

        return ret


class CategoryRange(RangeType):

    def __init__(self, categories, probs=None, size=None):

        super(CategoryRange, self).__init__(size=size)

        self.categories = list(categories)
        if probs is None:
            self.probs = None
        else:
            probs = [max(0.0, min(float(prob), 1.0)) for prob in probs]
            sum_probs = sum(probs)
            self.probs = [prob / sum_probs for prob in probs]
            assert(len(self.probs) == len(self.categories))

    def get_random(self):

        ret = []
        for i in range(1 if self.size is None else self.size):
            if self.probs is None:
                rand = self.categories[np.random.randint(len(self.categories))]
            else:
                csum = np.cumsum(self.probs)
                r = np.random.rand()
                for cs in range(len(csum)):
                    if r <= csum[cs]:
                        rand = self.categories[cs]
                        break

            ret.append(rand)

        if self.size is None:
            ret = ret[0]

        return ret


class LabelEncodeRange(RangeType):

    def __init__(self, num_labels, dtype=int):

        super(LabelEncodeRange, self).__init__(dtype=dtype)

        self.num_labels = max(1, int(num_labels))

    def get_random(self):

        if self.dtype is float:
            rand = [0.0] * self.num_labels
            rand[np.random.randint(self.num_labels)] = 1.0
        elif self.dtype is int:
            rand = [0] * self.num_labels
            rand[np.random.randint(self.num_labels)] = 1
        else:  # self.dtype is bool:
            rand = [False] * self.num_labels
            rand[np.random.randint(self.num_labels)] = True

        return rand


class BoolRange(RangeType):

    def __init__(self, size=None):

        super(BoolRange, self).__init__(size=size)

    def get_random(self):

        ret = []
        for i in range(1 if self.size is None else self.size):
            rand = np.random.randint(2) == 0
            ret.append(rand)

        if self.size is None:
            ret = ret[0]

        return ret


class CartesianProduct(object):

    def __init__(self, **kwargs):
        self.vars = kwargs
        self.constraints = {}

    def get_random(self):

        result = dict()
        for var in self.vars:
            if var in self.constraints:
                rel, constr = self.constraints[var]
                val = result[rel]
            else:
                val = None

                def constr(*args):
                    return True

            while True:
                res = self.vars[var].get_random()

                if constr(res, val):
                    break

            result[var] = res

        return result

    def add_constraints(self, contraints):
        for key in contraints:
            self.constraints[key[0]] = [key[1], contraints[key]]


class ExceedingThresholdException(Exception):
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
