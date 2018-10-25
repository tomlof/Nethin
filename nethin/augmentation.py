# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:09:30 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import warnings

from six import with_metaclass

import numpy as np
try:
    import skimage.transform as transform
    _HAS_SKIMAGE = True
except (ImportError):
    _HAS_SKIMAGE = False

    import scipy.ndimage

try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend.common import normalize_data_format

__all__ = ["BaseAugmentation",
           "Flip",
           "ImageResize", "ImageCrop",
           "ImageHistogramShift", "ImageHistogramScale",
           "ImageHistogramAffineTransform", "ImageHistogramTransform",
           "ImageTransform",
           "Pipeline"]


class BaseAugmentation(metaclass=abc.ABCMeta):
    """Base class for data augmentation functions.

    Parameters
    ----------
    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, [depth], channels)` while
        `channels_first` corresponds to inputs with shape `(batch, channels,
        height, [depth], width)`. It defaults to the `image_data_format` value
        found in your Keras config file at `~/.keras/keras.json`. If you never
        set it, then it will be "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers. If int,
        float or array_like, a new random state is created with the provided
        value as seed. If None, the default numpy random state (np.random) is
        used. Default is None, use the default numpy random state.
    """
    def __init__(self,
                 data_format=None,
                 random_state=None):

        self.data_format = normalize_data_format(data_format)

        if random_state is None:
            self.random_state = np.random.random.__self__  # Numpy built-in

        else:
            if isinstance(random_state, (int, float, np.ndarray)):
                self.random_state = np.random.RandomState(seed=random_state)

            elif isinstance(random_state, np.random.RandomState):
                self.random_state = random_state

            elif hasattr(random_state, "rand"):  # E.g., np.random
                self.random_state = random_state
                # Note: You may need to augment this "list" of required
                # functions in your subclasses.

            else:  # May crash here..
                self.random_state = np.random.RandomState(seed=random_state)

    def __call__(self, inputs):
        """The function performing the data augmentation.

        Specialise this function in your subclasses.
        """
        return inputs


class Flip(BaseAugmentation):
    """Flips an image in any direction.

    Parameters
    ----------
    probability : float or list/tuple of floats
        The probability of a flip. If a float, flip with probability
        ``probability`` in the horizontal direction (second image dimension)
        when ``axis=1`` or ``axis=None``, and otherwise flip with probability
        ``probability`` along all axes defined by ``axis``. If a list/tuple and
        ``axis=None``, flip with ``probability[d]`` in the direction of
        dimension ``d``, and if ``axis`` is a list or tuple, flip with
        probability ``probability[i]`` along dimension ``axis[i]`` (this case
        requires that ``len(probability) == len(axis)``). If fewer
        probabilities given than axes present, only the first given axes will
        be considered. Default is 0.5, which means to flip with probability
        ``0.5`` in the horizontal direction (along ``axis=1``), or to flip with
        probability ``0.5`` along the axes defined by ``axis``.

    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip the image. If axis is a tuple
         of ints, flipping is performed with the provided probabilities on all
         of the axes specified in the tuple. If an axis is negative it counts
         from the last to the first axis. If ``axis=None``, the axes are
         determined from ``probability``. Default is ``axis=None``, which means
         to flip along the second images axis (the assumed horizontal axis), or
         to flip along the axes using indices ``0, ...,
         len(probabilities) - 1.``

    data_format : str, optional
        One of ``"channels_last"`` (default) or ``"channels_first"``. The
        ordering of the dimensions in the inputs. ``"channels_last"``
        corresponds to inputs with shape ``(batch, [image dimensions ...],
        channels)`` while ``channels_first`` corresponds to inputs with shape
        ``(batch, channels, [image dimensions ...])``. It defaults to the
        ``image_data_format`` value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers. If int,
        float or array_like, a new random state is created with the provided
        value as seed. If None, the default numpy random state (np.random) is
        used. Default is None, use the default numpy random state.

    Examples
    --------
    >>> from nethin.augmentation import Flip
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> X = np.random.rand(2, 3, 1)
    >>> X[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452]])
    >>> flip_h = Flip(probability=1.0,
    ...               random_state=42, data_format="channels_last")
    >>> flip_h(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.73199394,  0.95071431,  0.37454012],
           [ 0.15599452,  0.15601864,  0.59865848]])
    >>> flip_v = Flip(probability=[1.0, 0.0],
    ...               random_state=42, data_format="channels_last")
    >>> flip_v(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.15601864,  0.15599452],
           [ 0.37454012,  0.95071431,  0.73199394]])
    >>> flip_hv = Flip(probability=[0.5, 0.5],
    ...                random_state=42, data_format="channels_last")
    >>> flip_hv(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.15601864,  0.15599452],
           [ 0.37454012,  0.95071431,  0.73199394]])
    >>> flip_hv(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452]])
    >>> flip_hv(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.15599452,  0.15601864,  0.59865848],
           [ 0.73199394,  0.95071431,  0.37454012]])
    >>> np.random.seed(42)
    >>> X = np.random.rand(2, 3, 1)
    >>> flip = Flip(probability=1.0, random_state=42)
    >>> flip(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.73199394,  0.95071431,  0.37454012],
           [ 0.15599452,  0.15601864,  0.59865848]])
    >>> np.random.seed(42)
    >>> X = np.random.rand(1, 2, 3)
    >>> flip = Flip(probability=1.0, random_state=42)
    >>> flip(X)[:, :, 0]  # Wrong
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.37454012]])
    >>> flip(X)[0, :, :]  # Wrong
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.15601864,  0.15599452],
           [ 0.37454012,  0.95071431,  0.73199394]])
    >>> flip = Flip(probability=1.0,
    ...             random_state=42, data_format="channels_first")
    >>> flip(X)[0, :, :]  # Right
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.73199394,  0.95071431,  0.37454012],
           [ 0.15599452,  0.15601864,  0.59865848]])
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(5, 4, 3, 2)
    >>> X[:, :, 0, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[0.37454012, 0.05808361, 0.83244264, 0.43194502],
           [0.45606998, 0.60754485, 0.30461377, 0.03438852],
           [0.54671028, 0.59789998, 0.38867729, 0.14092422],
           [0.00552212, 0.35846573, 0.31098232, 0.11959425],
           [0.52273283, 0.31435598, 0.22879817, 0.63340376]])
    >>> flip = Flip(probability=1.0, random_state=42)
    >>> flip(X)[:, :, 0, 0]
    array([[0.43194502, 0.83244264, 0.05808361, 0.37454012],
           [0.03438852, 0.30461377, 0.60754485, 0.45606998],
           [0.14092422, 0.38867729, 0.59789998, 0.54671028],
           [0.11959425, 0.31098232, 0.35846573, 0.00552212],
           [0.63340376, 0.22879817, 0.31435598, 0.52273283]])
    >>> flip = Flip(probability=1.0, axis=[0, 1],
    ...             random_state=42)
    >>> flip(X)[:, :, 0, 0]
    array([[0.52273283, 0.31435598, 0.22879817, 0.63340376],
           [0.00552212, 0.35846573, 0.31098232, 0.11959425],
           [0.54671028, 0.59789998, 0.38867729, 0.14092422],
           [0.45606998, 0.60754485, 0.30461377, 0.03438852],
           [0.37454012, 0.05808361, 0.83244264, 0.43194502]])
    >>> flip = Flip(probability=[1.0, 1.0], axis=[1],
    ...             random_state=42)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: Number of probabilities suppled does not match ...
    >>> X[0, :, 0, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[0.37454012, 0.95071431],
           [0.05808361, 0.86617615],
           [0.83244264, 0.21233911],
           [0.43194502, 0.29122914]])
    >>> flip = Flip(probability=[1.0, 1.0], axis=[1, 3],
    ...             random_state=42)  # doctest: +ELLIPSIS
    >>> flip(X)[0, :, 0, :]
    array([[0.43194502, 0.29122914],
           [0.83244264, 0.21233911],
           [0.05808361, 0.86617615],
           [0.37454012, 0.95071431]])
    """
    def __init__(self,
                 probability=0.5,
                 axis=None,
                 data_format=None,
                 random_state=None):

        super().__init__(data_format=data_format,
                         random_state=random_state)

        if axis is None:
            if isinstance(probability, (float, int)):
                self.axis = [1]
            else:
                self.axis = None
        elif isinstance(axis, int):
            self.axis = [axis]
        elif isinstance(axis, (tuple, list)):
            self.axis = [int(a) for a in axis]
        else:
            raise ValueError("The value of axis must be either None, int or "
                             "list/tuple.")

        if isinstance(probability, (float, int)):  # self.axis != None here
            probability = [float(probability) for i in range(len(self.axis))]

        elif isinstance(probability, (list, tuple)):
            if self.axis is None:
                probability = [float(probability[i])
                               for i in range(len(probability))]
                self.axis = [i for i in range(len(probability))]
            else:
                if len(probability) != len(self.axis):
                    raise ValueError("Number of probabilities suppled does "
                                     "not match the number of axes.")
                else:
                    probability = [float(probability[i])
                                   for i in range(len(probability))]
        # Normalise
        for i in range(len(probability)):
            probability[i] = max(0.0, min(float(probability[i]), 1.0))
        self.probability = probability

        if self.data_format == "channels_last":
            self._axis_offset = 0
        else:  # data_format == "channels_first":
            self._axis_offset = 1

    def __call__(self, inputs):

        outputs = inputs
        for i in range(len(self.probability)):
            p = self.probability[i]
            a = self._axis_offset + self.axis[i]
            if self.random_state.rand() < p:
                outputs = np.flip(outputs, axis=a)

        return outputs


class Resize(BaseAugmentation):
    """Resizes an image.

    Parameters
    ----------
    size : list of int
        List of positive int. The size to re-size the images to.

    random_size : int or list of int, optional
        An int or a list of positive int the same length as ``size``. The upper
        bounds on the amount of extra size to randomly add to the size. If a
        single int, the same random size will be added to all axes. Default is
        0, which means to not add any extra random size.

    keep_aspect_ratio : bool, optional
        Whether or not to keep the aspect ratio of the image when resizing.
        Default is False, do not keep the aspect ratio of the original image.

    minimum_size : bool, optional
        If ``keep_aspect_ratio=True``, then ``minimum_size`` determines if the
        given size is the minimum size (scaled image is equal to or larger than
        the given ``size``) or the maximum size (scaled image is equal to or
        smaller than the given ``size``) of the scaled image. Default is True,
        the scaled image will be at least as large as ``size``. See also
        ``keep_aspect_ratio``.

    order : int, optional
        Integer in [0, 5], the order of the spline used in the interpolation.
        The order corresponds to the following interpolations:

            0: Nearest-neighbor
            1: Bi-linear (default)
            2: Bi-quadratic
            3: Bi-cubic
            4: Bi-quartic
            5: Bi-quintic

        Beware! Higher orders than 1 may cause the values to be outside of the
        allowed range of values for your data. This must be handled manually.

        Default is 1, i.e. bi-linear interpolation.

    anti_aliasing : bool, optional
        Whether or not to apply a filter to smooth the image prior to
        down-scaling. When down-sampling the image, filtering helps to avoid
        aliasing artifacts. When scikit-image is available, a Gaussian filter
        is applied; otherwise, scipy applies a spline filter. Default is True,
        use anti-aliasing.

    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers. If int,
        float or array_like, a new random state is created with the provided
        value as seed. If None, the default numpy random state (np.random) is
        used. Default is None, use the default numpy random state.

    Examples
    --------
    >>> from nethin.augmentation import Resize
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, [2, 2, 1])
    >>> X[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 2],
           [2, 3]])
    >>> resize = Resize([4, 4], order=1, data_format="channels_last")
    >>> Y = resize(X)
    >>> Y[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  1.25,  1.75,  2.  ],
           [ 1.25,  1.5 ,  2.  ,  2.25],
           [ 1.75,  2.  ,  2.5 ,  2.75],
           [ 2.  ,  2.25,  2.75,  3.  ]])
    >>> resize = Resize([2, 2], order=1, data_format="channels_last")
    >>> X_ = resize(Y)
    >>> X_[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.25,  2.  ],
           [ 2.  ,  2.75]])
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]]).reshape((1, 2, 2))
    >>> X[0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 2],
           [2, 3]])
    >>> resize = Resize([4, 4], order=1, data_format="channels_first")
    >>> Y = resize(X)
    >>> Y[0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  1.25,  1.75,  2.  ],
           [ 1.25,  1.5 ,  2.  ,  2.25],
           [ 1.75,  2.  ,  2.5 ,  2.75],
           [ 2.  ,  2.25,  2.75,  3.  ]])
    >>>
    >>> X = np.random.rand(10, 20, 1)
    >>> resize = Resize([5, 5], keep_aspect_ratio=False, order=1)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (5, 5, 1)
    >>> resize = Resize([5, 5], keep_aspect_ratio=True,
    ...                 minimum_size=True, order=1)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (5, 10, 1)
    >>> resize = Resize([5, 5], keep_aspect_ratio=True,
    ...                 minimum_size=False, order=1)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (3, 5, 1)
    >>>
    >>> X = np.random.rand(10, 20, 30, 1)
    >>> resize = Resize([5, 5, 5],
    ...                 keep_aspect_ratio=True,
    ...                 minimum_size=False,
    ...                 order=1)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (2, 3, 5, 1)
    >>> resize = Resize([5, 5, 5],
    ...                 keep_aspect_ratio=True,
    ...                 minimum_size=True,
    ...                 order=1)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (5, 10, 15, 1)
    >>> X = np.arange(27).reshape((3, 3, 3, 1))
    >>> resize = Resize([5, 5, 5],
    ...                 keep_aspect_ratio=True)
    >>> Y = resize(X)
    >>> Y.shape  # doctest: +NORMALIZE_WHITESPACE
    (5, 5, 5, 1)
    >>> X[:5, :5, 0, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0,  3,  6],
           [ 9, 12, 15],
           [18, 21, 24]])
    >>> Y[:5, :5, 0, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ,  1.2,  3. ,  4.8,  6. ],
           [ 3.6,  4.8,  6.6,  8.4,  9.6],
           [ 9. , 10.2, 12. , 13.8, 15. ],
           [14.4, 15.6, 17.4, 19.2, 20.4],
           [18. , 19.2, 21. , 22.8, 24. ]])
    >>> X[0, :5, :5, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> Y[0, :5, :5, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[0. , 0.4, 1. , 1.6, 2. ],
           [1.2, 1.6, 2.2, 2.8, 3.2],
           [3. , 3.4, 4. , 4.6, 5. ],
           [4.8, 5.2, 5.8, 6.4, 6.8],
           [6. , 6.4, 7. , 7.6, 8. ]])
    """
    def __init__(self,
                 size,
                 random_size=0,
                 keep_aspect_ratio=False,
                 minimum_size=True,
                 order=1,
                 anti_aliasing=False,  # TODO: Toggle when using skimage 0.15.
                 data_format=None,
                 random_state=None):

        super().__init__(data_format=data_format,
                         random_state=random_state)

        self.size = [max(1, int(size[i])) for i in range(len(list(size)))]

        if not _HAS_SKIMAGE:
            raise warnings.warn("scikit-image (skimage) is not available. "
                                "Will use scipy.ndimage.zoom instead. The "
                                "result may differ.")

        if isinstance(random_size, int):
            self.random_size = max(0, int(random_size))
        elif isinstance(random_size, (list, tuple)):
            self.random_size = [max(0, int(random_size[i]))
                                for i in range(len(list(random_size)))]
            if len(self.random_size) != len(self.size):
                raise ValueError("random_size and size must have the same "
                                 "lengths.")
        else:
            raise ValueError("random_size must be an int, or a list/tuple of "
                             "int.")

        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.minimum_size = bool(minimum_size)

        if int(order) not in [0, 1, 2, 3, 4, 5]:
            raise ValueError('``order`` must be in [0, 5].')
        self.order = int(order)

        self.anti_aliasing = bool(anti_aliasing)

        if self.data_format == "channels_last":
            self._axis_offset = 0
        else:  # data_format == "channels_first":
            self._axis_offset = 1

    def __call__(self, inputs):

        shape = inputs.shape
        if self.data_format == "channels_last":
            shape = shape[:-1]
        else:  # self.data_format == "channels_first"
            shape = shape[1:]
        ndim = len(shape)  # inputs.ndim - 1

        size_ = [0] * len(self.size)
        if len(size_) < ndim:
            size_.extend(shape[len(size_):ndim])  # Add dims from the data
        elif len(size_) > ndim:
            raise ValueError("The given size specifies more dimensions than "
                             "what is present in the data.")

        if isinstance(self.random_size, int):
            random_size = np.random.randint(0, self.random_size + 1)
            for i in range(len(self.size)):  # Recall: May be fewer than ndim
                size_[i] = self.size[i] + random_size
        else:  # List or tuple
            for i in range(len(self.size)):  # Recall: May be fewer than ndim
                random_size = np.random.randint(0, self.random_size[i] + 1)
                size_[i] = self.size[i] + random_size

        if self.keep_aspect_ratio:
            if self.minimum_size:
                val_i = np.argmin(shape)
            else:
                val_i = np.argmax(shape)
            factor = size_[val_i] / shape[val_i]

            new_size = [int((shape[i] * factor) + 0.5)
                        for i in range(len(shape))]
            new_factor = [new_size[i] / shape[i] for i in range(len(shape))]
        else:
            new_size = size_
            new_factor = [size_[i] / shape[i] for i in range(len(shape))]

        if self.data_format == "channels_last":
            num_channels = inputs.shape[-1]
            outputs = None  # np.zeros(new_size + [num_channels])
            for c in range(num_channels):
                if _HAS_SKIMAGE:
                    im = transform.resize(inputs[..., c],
                                          new_size,
                                          order=self.order,
                                          mode="edge",  # TODO: Opt!
                                          cval=0.0,  # TODO: Opt!
                                          clip=False,
                                          preserve_range=True,
                                          anti_aliasing=self.anti_aliasing)
                else:
                    im = scipy.ndimage.zoom(inputs[..., c],
                                            new_factor,
                                            order=self.order,
                                            # = "edge"
                                            mode="nearest",  # TODO: Opt!
                                            cval=0.0,  # TODO: Opt!
                                            prefilter=self.anti_aliasing)
                if outputs is None:
                    outputs = np.zeros(list(im.shape) + [num_channels])
                outputs[..., c] = im

        else:  # data_format == "channels_first":
            num_channels = inputs.shape[0]
            outputs = None
            for c in range(num_channels):
                if _HAS_SKIMAGE:
                    im = transform.resize(inputs[c, ...],
                                          new_size,
                                          order=self.order,
                                          mode="edge",  # TODO: Opt?
                                          clip=False,
                                          preserve_range=True,
                                          anti_aliasing=self.anti_aliasing)
                else:
                    im = scipy.ndimage.zoom(inputs[c, ...],
                                            new_factor,
                                            order=self.order,
                                            # = "edge"
                                            mode="nearest",  # TODO: Opt!
                                            cval=0.0,  # TODO: Opt!
                                            prefilter=self.anti_aliasing)
                if outputs is None:
                    outputs = np.zeros([num_channels] + list(im.shape))
                outputs[c, ...] = im

        return outputs


class ImageCrop(BaseAugmentation):
    """Crops an image.

    Parameters
    ----------
    crop : list of int
        A subimage size to crop from the image. If any images are smaller than
        crop in any direction, no cropping will be performed in that direction.

    crop_random : bool, optional
        Whether or not to select a random crop position, or the middle portion
        of the image when cropping. Default is True, select a random crop.

    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers. If int,
        float or array_like, a new random state is created with the provided
        value as seed. If None, the default numpy random state (np.random) is
        used. Default is None, use the default numpy random state.

    Examples
    --------
    >>> from nethin.augmentation import ImageCrop
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(4, 4, 1)
    >>> X[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394,  0.59865848],
           [ 0.15601864,  0.15599452,  0.05808361,  0.86617615],
           [ 0.60111501,  0.70807258,  0.02058449,  0.96990985],
           [ 0.83244264,  0.21233911,  0.18182497,  0.18340451]])
    >>> crop = ImageCrop([2, 2], crop_random=True)
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.15599452,  0.05808361],
           [ 0.70807258,  0.02058449]])
    >>> crop = ImageCrop([2, 2], crop_center=False)
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.15599452,  0.05808361],
           [ 0.70807258,  0.02058449]])
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431],
           [ 0.15601864,  0.15599452]])
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.73199394,  0.59865848],
           [ 0.05808361,  0.86617615]])
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(3, 3, 1)
    >>> X[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452],
           [ 0.05808361,  0.86617615,  0.60111501]])
    >>> crop = ImageCrop([2, 2], crop_center=True)
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431],
           [ 0.59865848,  0.15601864]])
    >>> crop = ImageCrop([2, 2], crop_center=False)
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.15601864],
           [ 0.05808361,  0.86617615]])
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.59865848,  0.15601864],
           [ 0.05808361,  0.86617615]])
    >>> crop(X)[:, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.15601864,  0.15599452],
           [ 0.86617615,  0.60111501]])
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(3, 3, 3)
    >>> X  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.37454012,  0.95071431,  0.73199394],
            [ 0.59865848,  0.15601864,  0.15599452],
            [ 0.05808361,  0.86617615,  0.60111501]],
           [[ 0.70807258,  0.02058449,  0.96990985],
            [ 0.83244264,  0.21233911,  0.18182497],
            [ 0.18340451,  0.30424224,  0.52475643]],
           [[ 0.43194502,  0.29122914,  0.61185289],
            [ 0.13949386,  0.29214465,  0.36636184],
            [ 0.45606998,  0.78517596,  0.19967378]]])
    >>> crop = ImageCrop([2, 2], crop_center=True)
    >>> crop(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.37454012,  0.95071431,  0.73199394],
            [ 0.59865848,  0.15601864,  0.15599452]],
           [[ 0.70807258,  0.02058449,  0.96990985],
            [ 0.83244264,  0.21233911,  0.18182497]]])
    >>> crop = ImageCrop([2, 2], crop_center=False)
    >>> crop(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.59865848,  0.15601864,  0.15599452],
            [ 0.05808361,  0.86617615,  0.60111501]],
           [[ 0.83244264,  0.21233911,  0.18182497],
            [ 0.18340451,  0.30424224,  0.52475643]]])
    >>> crop = ImageCrop([2, 2], crop_center=True, data_format="channels_last")
    >>> crop(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.37454012,  0.95071431,  0.73199394],
            [ 0.59865848,  0.15601864,  0.15599452]],
           [[ 0.70807258,  0.02058449,  0.96990985],
            [ 0.83244264,  0.21233911,  0.18182497]]])
    >>> np.all(crop(X) == X[0:2, 0:2, :])
    True
    >>> crop(X).shape == X[0:2, 0:2, :].shape
    True
    >>> crop = ImageCrop([2, 2], crop_center=True,
    ...                  data_format="channels_first")
    >>> crop(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.37454012,  0.95071431],
            [ 0.59865848,  0.15601864]],
           [[ 0.70807258,  0.02058449],
            [ 0.83244264,  0.21233911]],
           [[ 0.43194502,  0.29122914],
            [ 0.13949386,  0.29214465]]])
    >>> np.all(crop(X) == X[:, 0:2, 0:2])
    True
    >>> crop(X).shape == X[:, 0:2, 0:2].shape
    True
    """
    def __init__(self,
                 crop,
                 crop_random=True,
                 data_format=None,
                 random_state=None):

        super(ImageCrop, self).__init__(data_format=data_format,
                                        random_state=random_state)

        self.crop = [max(0, int(crop[i])) for i in range(len(list(crop)))]
        self.crop_random = bool(crop_random)

        # Not checked in base class
        assert(hasattr(self.random_state, "randint"))

        if self.data_format == "channels_last":
            self._axis_offset = 0
        else:  # data_format == "channels_first":
            self._axis_offset = 1

    def __call__(self, inputs):

        crop = [None] * len(self.crop)
        for i in range(len(self.crop)):
            crop[i] = min(inputs.shape[self._axis_offset + i], self.crop[i])

        coord = [None] * len(self.crop)
        slices = []
        if self._axis_offset > 0:
            slices.append(slice(None))
        for i in range(len(self.crop)):
            if self.crop_random:
                coord[i] = int(round((inputs.shape[self._axis_offset + i] / 2)
                                     - (crop[i] / 2)) + 0.5)
            else:
                coord[i] = self.random_state.randint(
                    0,
                    max(1, inputs.shape[self._axis_offset + i] - crop[i] + 1))

            slices.append(slice(coord[i], coord[i] + crop[i]))

        outputs = inputs[tuple(slices)]

        return outputs


class ImageHistogramShift(BaseAugmentation):
    """Shifts the histogram of the inputs.

    Parameters
    ----------
    shift : float or Callable
        Either the amount to shift, or a image-wise shift function (a function
        that takes a whole image as input).

    min_value : float, optional
        The minimum possible or allowed value of the image. If None, no lower
        clipping will be performed. Default is None.

    max_value : float, optional
        The maximum possible or allowed value of the image. If None, no upper
        clipping will be performed. Default is None.

    Examples
    --------
    >>> from nethin.augmentation import ImageHistogramShift
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(3, 3)
    >>> X  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452],
           [ 0.05808361,  0.86617615,  0.60111501]])
    >>> shift = ImageHistogramShift(1)
    >>> shift(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.37454012,  1.95071431,  1.73199394],
           [ 1.59865848,  1.15601864,  1.15599452],
           [ 1.05808361,  1.86617615,  1.60111501]])
    >>> shift = ImageHistogramShift(-0.5, min_value=0.0)
    >>> shift(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.        ,  0.45071431,  0.23199394],
           [ 0.09865848,  0.        ,  0.        ],
           [ 0.        ,  0.36617615,  0.10111501]])
    >>> shift = ImageHistogramShift(0.5, max_value=1.0)
    >>> shift(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.87454012,  1.        ,  1.        ],
           [ 1.        ,  0.65601864,  0.65599452],
           [ 0.55808361,  1.        ,  1.        ]])
    """
    def __init__(self,
                 shift,
                 min_value=None,
                 max_value=None,
                 data_format=None,
                 random_state=None):

        super(ImageHistogramShift, self).__init__(data_format=data_format,
                                                  random_state=random_state)

        if isinstance(shift, (int, float)):
            self.shift = lambda I: I + float(shift)
        elif callable(shift):
            self.shift = shift
        else:
            raise ValueError("``shift`` must either be a scalar or a "
                             "callable.")

        if min_value is None:
            self.min_value = min_value
        else:
            self.min_value = float(min_value)

        if max_value is None:
            self.max_value = max_value
        else:
            self.max_value = float(max_value)

    def __call__(self, inputs):

        outputs = self.shift(inputs)

        if (self.min_value is not None) or (self.max_value is not None):
            outputs = np.clip(outputs, self.min_value, self.max_value)

        return outputs


class ImageHistogramScale(BaseAugmentation):
    """Scales the histogram of the inputs.

    Parameters
    ----------
    scale : float or Callable
        Either the scale factor, or a image-wise scale function (a function
        that takes a whole image as input).

    min_value : float, optional
        The minimum possible or allowed value of the image. If None, no lower
        clipping will be performed. Default is None.

    max_value : float, optional
        The maximum possible or allowed value of the image. If None, no upper
        clipping will be performed. Default is None.

    Examples
    --------
    >>> from nethin.augmentation import ImageHistogramScale
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(3, 3)
    >>> X  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452],
           [ 0.05808361,  0.86617615,  0.60111501]])
    >>> scale = ImageHistogramScale(2)
    >>> scale(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.74908024,  1.90142861,  1.46398788],
           [ 1.19731697,  0.31203728,  0.31198904],
           [ 0.11616722,  1.73235229,  1.20223002]])
    >>> scale = ImageHistogramScale(20.0, max_value=10.0)
    >>> scale(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[  7.49080238,  10.        ,  10.        ],
           [ 10.        ,   3.12037281,   3.11989041],
           [  1.16167224,  10.        ,  10.        ]])
    >>> scale = ImageHistogramScale(0.5, max_value=0.25)
    >>> scale(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.18727006,  0.25      ,  0.25      ],
           [ 0.25      ,  0.07800932,  0.07799726],
           [ 0.02904181,  0.25      ,  0.25      ]])
    """
    def __init__(self,
                 scale,
                 min_value=None,
                 max_value=None,
                 data_format=None,
                 random_state=None):

        super(ImageHistogramScale, self).__init__(data_format=data_format,
                                                  random_state=random_state)

        if isinstance(scale, (int, float)):
            self.scale = lambda I: I * float(scale)

        elif callable(scale):
            self.scale = scale

        else:
            raise ValueError("``scale`` must either be a scalar or a "
                             "callable.")

        if min_value is None:
            self.min_value = min_value
        else:
            self.min_value = float(min_value)

        if max_value is None:
            self.max_value = max_value
        else:
            self.max_value = float(max_value)

    def __call__(self, inputs):

        outputs = self.scale(inputs)

        if (self.min_value is not None) or (self.max_value is not None):
            outputs = np.clip(outputs, self.min_value, self.max_value)

        return outputs


class ImageHistogramAffineTransform(BaseAugmentation):
    """Performs an affine transformation of the histogram of the inputs.

    This means that the following affine transformation is applied:

        I' = scale * I + shift.

    Parameters
    ----------
    scale : float or Callable, optional
        Either the scale factor, or a image-wise scale function (a function
        that takes a whole image as input).

    shift : float or Callable, optional
        Either the amount to shift, or a image-wise shift function (a function
        that takes a whole image as input).

    min_value : float, optional
        The minimum possible or allowed value of the image. If None, no lower
        clipping will be performed. Default is None.

    max_value : float, optional
        The maximum possible or allowed value of the image. If None, no upper
        clipping will be performed. Default is None.

    Examples
    --------
    >>> from nethin.augmentation import ImageHistogramAffineTransform
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(3, 3)
    >>> X  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.37454012,  0.95071431,  0.73199394],
           [ 0.59865848,  0.15601864,  0.15599452],
           [ 0.05808361,  0.86617615,  0.60111501]])
    >>> affine_transform = ImageHistogramAffineTransform(shift=2, scale=3)
    >>> affine_transform(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 3.12362036,  4.85214292,  4.19598183],
           [ 3.79597545,  2.46805592,  2.46798356],
           [ 2.17425084,  4.59852844,  3.80334504]])
    >>> np.linalg.norm((3 * X + 2) - affine_transform(X))
    0.0
    >>> affine_transform = ImageHistogramAffineTransform(shift=5.0,
    ...                                                  scale=10.0,
    ...                                                  max_value=10.0)
    >>> affine_transform(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[  8.74540119,  10.        ,  10.        ],
           [ 10.        ,   6.5601864 ,   6.5599452 ],
           [  5.58083612,  10.        ,  10.        ]])
    >>> affine_transform = ImageHistogramAffineTransform(shift=-0.5, scale=1.0,
    ...                                                  min_value=-0.25,
    ...                                                  max_value=0.25)
    >>> affine_transform(X)  # doctest: +NORMALIZE_WHITESPACE
    array([[-0.12545988,  0.25      ,  0.23199394],
           [ 0.09865848, -0.25      , -0.25      ],
           [-0.25      ,  0.25      ,  0.10111501]])
    """
    def __init__(self,
                 scale=1.0,
                 shift=0.0,
                 min_value=None,
                 max_value=None,
                 data_format=None,
                 random_state=None):

        super(ImageHistogramAffineTransform,
              self).__init__(data_format=data_format,
                             random_state=random_state)

        if isinstance(shift, (int, float)):
            shift = float(shift)
            if shift != 0.0:
                self.shift = lambda I: I + shift
            else:
                self.shift = lambda I: I
        elif callable(shift):
            self.shift = shift
        else:
            raise ValueError("``shift`` must either be a scalar or a "
                             "callable.")

        if isinstance(scale, (int, float)):
            scale = float(scale)
            if scale != 1.0:
                self.scale = lambda I: scale * I
            else:
                self.scale = lambda I: I
        elif callable(scale):
            self.scale = scale
        else:
            raise ValueError("``scale`` must either be a scalar or a "
                             "callable.")

        if min_value is None:
            self.min_value = min_value
        else:
            self.min_value = float(min_value)

        if max_value is None:
            self.max_value = max_value
        else:
            self.max_value = float(max_value)

    def __call__(self, inputs):

        outputs = self.shift(self.scale(inputs))

        if (self.min_value is not None) or (self.max_value is not None):
            outputs = np.clip(outputs, self.min_value, self.max_value)

        return outputs


class ImageHistogramTransform(BaseAugmentation):
    """Transforms the histogram of the input image (i.e., intensity transform).

    Parameters
    ----------
    transform : Transform or Callable
        Any scalar function defined on the domain of the input image. Ideally,
        this function should be monotonically increasing, but this is not a
        formal requirement by this class.

    min_value : float, optional
        The minimum possible or allowed value of the image. If None, no lower
        clipping will be performed. Default is None.

    max_value : float, optional
        The maximum possible or allowed value of the image. If None, no upper
        clipping will be performed. Default is None.

    Examples
    --------
    >>> from nethin.augmentation import ImageHistogramTransform
    >>> from nethin.utils import simple_bezier
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> x = 0.125 * np.random.randn(10000, 1) + 0.5
    >>> float("%.12f" % np.mean(x))
    0.499733002079
    >>> hist, _ = np.histogram(x, bins=15)
    >>> hist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [4, 19, 75, 252, 630, 1208, 1787, 2071, 1817, 1176, 626, 239, 73, 20, 3]
    >>> transf = simple_bezier([-0.45, 0.0, 0.45], controls=[0.15, 0.5, 0.85],
    ...                        steps=100)
    >>> transform = ImageHistogramTransform(transf,
    ...                                     min_value=0.0, max_value=1.0)
    >>> x_trans = transform(x)
    >>> float("%.12f" % np.mean(x_trans))
    0.49932746958
    >>> hist, _ = np.histogram(x_trans, bins=15)
    >>> hist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [656, 621, 699, 690, 696, 649, 652, 674, 657, 678, 676, 684, 666, 644, 658]
    >>> transf = simple_bezier([-0.45, 0.2, 0.4], controls=[0.15, 0.6, 0.85],
    ...                        steps=100)
    >>> transform = ImageHistogramTransform(transf,
    ...                                     min_value=0.0, max_value=1.0)
    >>> x_trans = transform(x)
    >>> float("%.12f" % np.mean(x_trans))
    0.572189192312
    >>> hist, _ = np.histogram(x_trans, bins=15)
    >>> hist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [345, 398, 489, 566, 587, 607, 637, 678, 688, 709, 807, 843, 850, 856, 940]
    >>>
    >>> np.random.seed(42)
    >>> x = 125.0 * np.random.randn(10000, 1) + 1500
    >>> float("%.12f" % np.mean(x))
    1499.733002078947
    >>> hist, _ = np.histogram(x, bins=15)
    >>> hist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [4, 19, 75, 252, 630, 1208, 1787, 2071, 1817, 1176, 626, 239, 73, 20, 3]
    >>> transf_ = simple_bezier([-0.45, 0.0, 0.45], controls=[0.15, 0.5, 0.85],
    ...                         steps=100)
    >>> transf = lambda x: (transf_((x - 1000.0) / 1000.0) * 1000.0 + 1000.0)
    >>> transform = ImageHistogramTransform(transf,
    ...                                     min_value=1000.0, max_value=2000.0)
    >>> x_trans = transform(x)
    >>> float("%.12f" % np.mean(x_trans))
    1499.327469579897
    >>> hist, _ = np.histogram(x_trans, bins=15)
    >>> hist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [656, 621, 699, 690, 696, 649, 652, 674, 657, 678, 676, 684, 666, 644, 658]
    """
    def __init__(self,
                 transform,
                 min_value=None,
                 max_value=None,
                 data_format=None,
                 random_state=None):

        super(ImageHistogramTransform, self).__init__(data_format=data_format,
                                                      random_state=random_state)

        if not callable(transform):
            raise ValueError('``transform`` must be callable.')
        self.transform = transform

        if min_value is None:
            self.min_value = min_value
        else:
            self.min_value = float(min_value)

        if max_value is None:
            self.max_value = max_value
        else:
            self.max_value = float(max_value)

        self._vec_trans = np.vectorize(self.transform)

    def __call__(self, inputs):

        if isinstance(self.transform, Transform):
            self.transform.prepare()

        outputs = self._vec_trans(inputs)

        if (self.min_value is not None) or (self.max_value is not None):
            outputs = np.clip(outputs, self.min_value, self.max_value)

        return outputs


class ImageTransform(BaseAugmentation):
    """Transforms an entire input image.

    Parameters
    ----------
    transform : Transform or Callable
        Any function defined on the domain of the input image mapping to
        another image.

    min_value : float, optional
        The minimum possible or allowed value of the image. If None, no lower
        clipping will be performed. Default is None.

    max_value : float, optional
        The maximum possible or allowed value of the image. If None, no upper
        clipping will be performed. Default is None.

    Examples
    --------
    >>> from nethin.augmentation import ImageTransform
    >>> from nethin.utils import simple_bezier
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    """
    def __init__(self,
                 transform,
                 min_value=None,
                 max_value=None,
                 data_format=None,
                 random_state=None):

        super(ImageTransform, self).__init__(data_format=data_format,
                                             random_state=random_state)

        if not callable(transform):
            raise ValueError('``transform`` must be callable.')
        self.transform = transform

        if min_value is None:
            self.min_value = min_value
        else:
            self.min_value = float(min_value)

        if max_value is None:
            self.max_value = max_value
        else:
            self.max_value = float(max_value)

        if (self.min_value is not None) and (self.max_value is not None):
            assert(self.min_value < self.max_value)

    def __call__(self, inputs):

        if isinstance(self.transform, Transform):
            self.transform.prepare()

        outputs = self.transform(inputs)

        if (self.min_value is not None) or (self.max_value is not None):
            outputs = np.clip(outputs, self.min_value, self.max_value)

        return outputs


# Deprecated names!
ImageFlip = Flip
ImageResize = Resize


class Pipeline(object):
    """Applies a data augmentation/preprocessing pipeline.

    Parameters
    ----------
    pipeline : list of BaseAugmentation or Callable, optional
        The pipeline. A list of data augmentation functions, that will be
        applied (chained) one at the time starting with the first element of
        ``pipeline`` and ending with the last element of ``pipeline``. Default
        is an empty list.

    Returns
    -------
    outputs : object
        The augmented data. Often of the same type as the input.
    """
    def __init__(self, pipeline=[]):

        try:
            _pipeline = []
            for p in pipeline:
                if isinstance(p, BaseAugmentation):
                    _pipeline.append(p)
                elif callable(p):
                    _pipeline.append(p)
                else:
                    raise RuntimeError()
            self.pipeline = _pipeline
        except (TypeError, RuntimeError):
            raise ValueError('"pipeline" must be a list of '
                             '"BaseAugmentation" or "Callable".')

    def __call__(self, inputs):

        outputs = inputs
        for p in self.pipeline:
            outputs = p(outputs)
        return outputs

    def add(self, p):
        """Add an data autmentation/preprocessing step to the pipeline.
        """
        self.pipeline.append(p)


class Transform(object):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def __call__(self, x):
        return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
