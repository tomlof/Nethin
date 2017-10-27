# -*- coding: utf-8 -*-
"""
Contains helper functions and utility functions for use in the library.

Created on Mon Oct  9 14:05:26 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import json
import base64

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.engine.topology import _to_snake_case as to_snake_case

__all__ = ["with_device", "Helper", "to_snake_case"]

# TODO: Make a helper module for each backend instead, as in Keras.
# TODO: Check for supported backends.


def with_device(device, function, *args, **kwargs):
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
    if device is None:
        ret = function(*args, **kwargs)
    else:
        with tf.device(device):
            ret = function(*args, **kwargs)

    return ret


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
