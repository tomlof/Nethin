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
import scipy.interpolate as interpolate

import tensorflow as tf
import keras.backend as K
from keras.engine.topology import _to_snake_case as to_snake_case

__all__ = ["Helper", "with_device", "to_snake_case",
           "get_json_type",
           "normalize_object", "normalize_list", "normalize_str",
           "simple_bezier", "dynamic_histogram_warping"]

# TODO: Make a helper module for each backend instead, as in Keras.
# TODO: Check for supported backends.


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

    bins : The number of histogram bins. Default is 256, in order to capture
        all intensities in a standard ``uint8`` image.

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
        else:  # All costs are equal: Prefer to be close to the diagonal in
               # this case.
            if (float(m) / float(num_bins_A)) > (float(n) / float(num_bins_B)):
                min_m = m - 1
                min_n = n
            elif (float(n) / float(num_bins_B)) > (float(m) / float(num_bins_A)):
                min_m = m
                min_n = n - 1
            else:
                min_m = m - 1
                min_n = n - 1
        m = min_m
        n = min_n
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
