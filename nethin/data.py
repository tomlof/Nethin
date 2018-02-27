# -*- coding: utf-8 -*-
"""
Contains means to read, generate and handle data.

Created on Tue Oct  3 08:20:52 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import os
import re
import abc
import datetime

import numpy as np
from six import with_metaclass
from scipy.misc import imread, imresize

from keras.utils import conv_utils

import nethin
import nethin.utils as utils

try:
    from collections import Generator
    _HAS_GENERATOR = True
except (ImportError):
    _HAS_GENERATOR = False

try:
    import dicom
    _HAS_DICOM = True
except (ImportError):
    _HAS_DICOM = False

__all__ = ["BaseGenerator",
           "ImageGenerator", "ArrayGenerator",
           "Dicom3DGenerator", "DicomGenerator",
           "Numpy3DGenerator",
           "Dicom3DSaver"]


if _HAS_GENERATOR:
    class BaseGenerator(with_metaclass(abc.ABCMeta, Generator)):  # Python 3
        pass
else:
    class BaseGenerator(with_metaclass(abc.ABCMeta, object)):  # Python 2
        """Abstract base class for generators.

        Adapted from:

            https://github.com/python/cpython/blob/3.6/Lib/_collections_abc.py
        """
        def __iter__(self):
            return self

        def __next__(self):
            """Return the next item from the generator.

            When exhausted, raise StopIteration.
            """
            return self.send(None)

        def close(self):
            """Raise GeneratorExit inside generator.
            """
            try:
                self.throw(GeneratorExit)
            except (GeneratorExit, StopIteration):
                pass
            else:
                raise RuntimeError("generator ignored GeneratorExit")

        def __subclasshook__(cls, C):

            if cls is Generator:
                methods = ["__iter__", "__next__", "send", "throw", "close"]
                mro = C.__mro__
                for method in methods:
                    for B in mro:
                        if method in B.__dict__:
                            if B.__dict__[method] is None:
                                return NotImplemented
                            break
                    else:
                        return NotImplemented

                return True

            return NotImplemented

        @abc.abstractmethod
        def send(self, value):
            """Send a value into the generator.

            Return next yielded value or raise StopIteration.
            """
            raise StopIteration

        @abc.abstractmethod
        def throw(self, typ, val=None, tb=None):
            """Raise an exception in the generator.
            """
            if val is None:
                if tb is None:
                    raise typ
                val = typ()

            if tb is not None:
                val = val.with_traceback(tb)

            raise val


class ImageGenerator(BaseGenerator):
    """A generator over the images in a given directory.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the images.

    recursive : bool, optional
        Whether or not to traverse the given directory recursively. Default is
        False, do not traverse the directory recursively.

    batch_size : int, optional
        The number of images to return at each yield. Default is 1, return only
        one image at the time. If there are not enough images to return in one
        batch, the source directory is considered exhausted, and StopIteration
        will be thrown.

    num_training : int or float
        The total number of training samples to return (or None, to read all),
        or a fraction in [0, 1] thereof. If a fraction is given, the generator
        will first count the total number of images in the directory (and its
        subdirectories if ``recursive=True``); note that if the number of files
        is substantial, this may require some time before the first image (or
        batch of images) is yielded.

    crop : tuple of int, length 2, optional
        A subimage size to crop randomly from the read image. If any images are
        smaller than crop in any direction, no cropping will be performed in
        that direction. Default is None, do not perform any cropping.

    size : tuple of int, length 2, optional
        The (possibly cropped image) will be resized to this absolute size.
        Default is None, do not resize the images. See also
        ``keep_aspect_ratio`` and ``minimum_size``.

    flip : float, optional
        The probability of flipping the image in the left-right direction.
        Default is None, which means to not flip the image (equivalent to
        ``flip=0.0``.

    crop_center : bool, optional
        Whether or not to select the middle portion of the image when cropping,
        or to select random crop positions. Default is True, select the center
        of the image when cropping.

    keep_aspect_ratio : bool, optional
        Whether or not to keep the aspect ratios of the images when resizing.
        Only used if size it not None. Default is True, keep the aspect ratio
        of the original image. See also ``minimum_size``.

    minimum_size : bool, optional
        If ``keep_aspect_ratio=True``, then ``minimum_size`` determines if the
        given size is the minimum size (scaled image is equal to or larger than
        the given ``size``) or the maximum size (scaled image is equal to or
        smaller than the given ``size``) of the scaled image. Default is True,
        the scaled image will be at least as large as ``size``. See also
        ``keep_aspect_ratio``.

    interp : str, optional
        Interpolation to use for re-sizing ("nearest", "lanczos", "bilinear",
        "bicubic" or "cubic"). Default is "bilinear".

    restart_generation : bool, optional
        Whether or not to start over from the first file again after the
        generator has finished. Default is False, do not start over again.

    bias : float, optional
        A bias to add to the generated images. Use this in conjunction with
        ``scale`` in order to scale and center the images to a particular
        range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale image to the
        range [-1, 1], you would have ``bias=-127.5`` and ``scale=1.0 / 127.5``
        and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not add a bias.

    scale : float, optional
        A factor to use to scale the generated images. Use this in conjunction
        with ``bias`` in order to scale and center the images to a particular
        range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale image to the
        range [-1, 1], you would have ``bias=-127.5`` and ``scale=1.0 / 127.5``
        and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not scale the images.

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers (for the
        flip). If int, float or array_like, a new random state is created with
        the provided value as seed. If None, the default numpy random state
        (np.random) is used. Default is None, use the default numpy random
        state.

    Examples
    --------
    >>> import numpy as np
    >>> from nethin.data import ImageGenerator
    """
    def __init__(self, dir_path, recursive=False, batch_size=1,
                 num_training=None, crop=None, size=None, flip=None,
                 crop_center=True, keep_aspect_ratio=True, minimum_size=True,
                 interp="bilinear", restart_generation=False, bias=None,
                 scale=None, random_state=None):

        # TODO: Handle recursive and num_training!

        self.dir_path = str(dir_path)
        self.recursive = bool(recursive)
        self.batch_size = max(1, int(batch_size))
        if num_training is None:
            self.num_training = num_training
        else:
            if isinstance(num_training, float):
                self.num_training = max(0.0, min(float(num_training), 1.0))
            else:
                self.num_training = max(1, int(num_training))

        if crop is None:
            self.crop = crop
        else:
            self.crop = (max(1, int(crop[0])), max(1, int(crop[1])))

        if size is None:
            self.size = size
        else:
            self.size = (max(1, int(size[0])), max(1, int(size[1])))

        if flip is None:
            self.flip = flip
        else:
            self.flip = max(0.0, min(float(flip), 1.0))

        self.crop_center = bool(crop_center)
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.minimum_size = bool(minimum_size)
        self.interp = str(interp)
        self.bias = float(bias) if (bias is not None) else bias
        self.scale = float(scale) if (scale is not None) else scale
        self.restart_generation = bool(restart_generation)

        if random_state is None:
            self.random_state = np.random.random.__self__
        else:
            if isinstance(random_state, (int, float, np.ndarray)):
                self.random_state = np.random.RandomState(seed=random_state)
            elif isinstance(random_state, np.random.RandomState):
                self.random_state = random_state
            elif hasattr(random_state, "rand") and \
                    hasattr(random_state, "randint"):  # E.g., np.random
                self.random_state = random_state
            else:  # May crash here..
                self.random_state = np.random.RandomState(seed=random_state)

        self.walker = None
        self._restart_walker()

        self.left_files = []

    def _restart_walker(self):

        if self.walker is not None:
            self.walker.close()
        self.walker = os.walk(self.dir_path)

    def _update_left_files(self):

        try_again = True
        tries = 0
        while try_again and (tries <= 1):
            try_again = False
            try:
                dir_name, sub_dirs, files = next(self.walker)

                for i in range(len(files)):
                    file = os.path.join(dir_name, files[i])
                    self.left_files.append(file)

                if not self.recursive:
                    self.walker.close()

            except StopIteration as e:
                if self.restart_generation:
                    self._restart_walker()
                    try_again = True
                    tries += 1  # Only try again once
                else:
                    self.throw(e)

            except Exception as e:
                self.throw(e)

    def _read_image(self, file_name):

        try:
            image = imread(file_name)

            if len(image.shape) != 3:
                return None
            else:
                return image

        except FileNotFoundError:
            return None

    def _process_image(self, image):

        if self.size is not None:
            if self.keep_aspect_ratio:
                im_size = image.shape[:2]
                factors = [float(im_size[0]) / float(self.size[0]),
                           float(im_size[1]) / float(self.size[1])]
                factor = min(factors) if self.minimum_size else max(factors)
                new_size = list(im_size[:])
                new_size[0] = int((new_size[0] / factor) + 0.5)
                new_size[1] = int((new_size[1] / factor) + 0.5)
            else:
                new_size = self.size

            image = imresize(image, new_size, interp=self.interp)

        if self.crop is not None:
            crop0 = min(image.shape[0], self.crop[0])
            crop1 = min(image.shape[1], self.crop[1])
            if self.crop_center:
                top = int(round((image.shape[0] / 2) - (crop0 / 2)) + 0.5)
                left = int(round((image.shape[1] / 2) - (crop1 / 2)) + 0.5)
            else:
                top = self.random_state.randint(0, max(1, image.shape[0] - crop0))
                left = self.random_state.randint(0, max(1, image.shape[1] - crop1))
            image = image[top:top + crop0, left:left + crop1]

        if self.flip is not None:
            if self.random_state.rand() < self.flip:
                image = image[:, ::-1, :]

        if self.bias is not None:
            image = image + self.bias

        if self.scale is not None:
            image = image * self.scale

        return image

    def throw(self, typ, **kwargs):
        """Raise an exception in the generator.
        """
        super(ImageGenerator, self).throw(typ, **kwargs)

    def send(self, value):
        """Send a value into the generator.

        Return next yielded value or raise StopIteration.
        """
        return_images = []
        while len(return_images) < self.batch_size:
            if len(self.left_files) < 1:
                self._update_left_files()

            file_name = self.left_files.pop()

            image = self._read_image(file_name)
            if image is not None:
                image = self._process_image(image)
                return_images.append(image)

        return return_images


class ArrayGenerator(BaseGenerator):
    """Generates batches from the data in the input array.

    Parameters
    ----------
    X : numpy.ndarray, shape (batch_dim, ...)
        The input array with the data.

    batch_size : int, optional
        The number of samples to return at each yield. If there are not enough
        samples to return in one batch, the source is considered exhausted, and
        StopIteration will be thrown unless other parameters tell us to
        continue. See ``wrap_around`` and ``restart_generation``. Default is
        32.

    wrap_around : bool, optional
        Whether or not to wrap around for the last batch, if the batch
        dimension is not divisible by the batch size. Otherwise, the last
        ``X.shape[0] % batch_size`` samples will be discarded. Default is
        False, do not wrap around.

    restart_generation : bool, optional
        Whether or not to start over from the first sample again after the
        generator has finished. Default is False, do not start over again.

    Examples
    --------
    >>> from nethin.data import ArrayGenerator
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    >>> X = np.random.randn(5, 2)
    >>> gen = ArrayGenerator(X, batch_size=2,
    ...                      wrap_around=False, restart_generation=False)
    >>> for batch in gen:
    >>>     print(batch)
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    [[ 0.20666447 -2.01126457]
     [-0.55725071  0.33721701]]
    >>> gen = ArrayGenerator(X, batch_size=2,
    ...                      wrap_around=True, restart_generation=False)
    >>> for batch in gen:
    >>>     print(batch)
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    [[ 0.20666447 -2.01126457]
     [-0.55725071  0.33721701]]
    [[ 1.54883597 -1.37073656]
     [-0.70318731 -0.49028236]]
    >>> gen = ArrayGenerator(X, batch_size=2,
    ...                      wrap_around=False, restart_generation=True)
    >>> counter = 1
    >>> for batch in gen:
    >>>     print(batch)
    >>>     if counter >= 5:
    >>>         break
    >>>     counter += 1
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    [[ 0.20666447 -2.01126457]
     [-0.55725071  0.33721701]]
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    [[ 0.20666447 -2.01126457]
     [-0.55725071  0.33721701]]
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    >>> gen = ArrayGenerator(X, batch_size=2,
    ...                      wrap_around=True, restart_generation=True)
    >>> counter = 1
    >>> for batch in gen:
    >>>     print(batch)
    >>>     if counter >= 6:
    >>>         break
    >>>     counter += 1
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    [[ 0.20666447 -2.01126457]
     [-0.55725071  0.33721701]]
    [[ 1.54883597 -1.37073656]
     [-0.70318731 -0.49028236]]
    [[-0.32181433 -1.75507872]
     [ 0.20666447 -2.01126457]]
    [[-0.55725071  0.33721701]
     [ 1.54883597 -1.37073656]]
    [[-0.70318731 -0.49028236]
     [-0.32181433 -1.75507872]]
    """
    def __init__(self, X, batch_size=32, wrap_around=False,
                 restart_generation=False):

        self.X = np.atleast_1d(X)
        self.batch_size = max(1, int(batch_size))
        self.wrap_around = bool(wrap_around)
        self.restart_generation = bool(restart_generation)

        self._done = False
        self._sample = 0
        if self.wrap_around:
            self._num_samples = self.X.shape[0]
        else:  # Ignore the last odd samples
            self._num_samples = self.X.shape[0] \
                    - (self.X.shape[0] % self.batch_size)

    def throw(self, typ, **kwargs):
        """Raise an exception in the generator.
        """
        super(ArrayGenerator, self).throw(typ, **kwargs)

    def send(self, value):
        """Send a value into the generator.

        Return next yielded value or raise StopIteration.
        """
        if self._done:
            self.throw(StopIteration)

        num_samples = self.X.shape[0]
        num_dims = len(self.X.shape) - 1
        slicer = [slice(None)] * num_dims
        return_samples = []
        while len(return_samples) < self.batch_size:
            if self._sample >= self._num_samples:  # First thing that happens
                if not self.wrap_around and not self.restart_generation:
                    self.throw(StopIteration)
                elif self.wrap_around and not self.restart_generation:
                    if self._sample >= num_samples:
                        self._sample = 0  # Start from the first sample
                    self._done = True
                elif not self.wrap_around and self.restart_generation:
                    self._sample = 0  # Start from the first sample
                    return_samples = []  # Don't use the last few odd samples
                else:  # self.wrap_around and self.restart_generation:
                    if self._sample >= num_samples:
                        self._sample = 0  # Start from the first sample

            sample = self.X[[self._sample] + slicer]
            return_samples.append(sample)

            self._sample += 1

        return np.array(return_samples)


class Dicom3DGenerator(BaseGenerator):
    """A generator over 3D Dicom images in a given directory.

    The images are organised in a directory for each image, a subdirectory
    for each channel, and the third-dimension slices for each channel are
    in those subdirectories.

    It will be assumed that the subdirectories (channels) of the given
    directory (image) contains different "channels" (different image modes,
    for instance), and they will be returned as such. The subdirectories and
    their order is determined by the list ``channel_names``.

    It will be assumed that the Dicom files have some particular tags. It will
    be assumed that they have: "RescaleSlope", "RescaleIntercept", "Rows",
    "Columns".

    This generator requires that the ``dicom`` package be installed.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the images.

    image_names : list of str
        The subdirectories to extract files from below ``dir_path``. Every
        element of this list corresponds to an image.

    channel_names : list of str or list of str
        The inner strings or lists corresponds to directory names or regular
        expressions defining the names of the subdirectories under
        ``image_names`` that corresponds to channels of this image. Every outer
        element of this list corresponds to a channel of the images defined by
        ``image_names``. The elements of the inner lists are alternative names
        for the subdirectories. If more than one subdirectory name matches,
        only the first one found will be used.

    batch_size : int or None, optional
        The number of images to return at each yield. If None, all images will
        be returned. If there are not enough images to return in one batch, the
        source directory is considered exhausted, and StopIteration will be
        thrown. Default is 1, which means to return only one image at the time.

    crop : tuple of int, length 2, optional
        A subimage size to crop randomly from the read image. If any images are
        smaller than crop in any direction, no cropping will be performed in
        that direction. Default is None, do not perform any cropping.

    size : tuple of int, length 2, optional
        The (possibly cropped image) will be resized to this absolute size.
        Default is None, do not resize the images. See also
        ``keep_aspect_ratio`` and ``minimum_size``.

    flip : float, optional
        The probability of flipping the image in the left-right direction.
        Default is None, which means to not flip the image (equivalent to
        ``flip=0.0``.

    crop_center : bool, optional
        Whether or not to select the middle portion of the image when cropping,
        or to select random crop positions. Default is True, select the center
        of the image when cropping.

    keep_aspect_ratio : bool, optional
        Whether or not to keep the aspect ratios of the images when resizing.
        Only used if size it not None. Default is True, keep the aspect ratio
        of the original image. See also ``minimum_size``.

    minimum_size : bool, optional
        If ``keep_aspect_ratio=True``, then ``minimum_size`` determines if the
        given size is the minimum size (scaled image is equal to or larger than
        the given ``size``) or the maximum size (scaled image is equal to or
        smaller than the given ``size``) of the scaled image. Default is True,
        the scaled image will be at least as large as ``size``. See also
        ``keep_aspect_ratio``.

    interp : str, optional
        Interpolation to use for re-sizing ("nearest", "lanczos", "bilinear",
        "bicubic" or "cubic"). Default is "bilinear".

    restart_generation : bool, optional
        Whether or not to start over from the first file again after the
        generator has finished. Default is False, do not start over again.

    bias : float or list of float, optional
        A bias to add to the generated images. If a list of float, each value
        is the bias for the corresponding channel. Use this in conjunction with
        ``scale`` in order to scale and center the images to a particular
        range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale image to the
        range [-1, 1], you would have ``bias=-127.5`` and ``scale=1.0 / 127.5``
        and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not add a bias.

    scale : float or list of float, optional
        A factor to use to scale the generated images. If a list of float, each
        value is the scale for the corresponding channel. Use this in
        conjunction with ``bias`` in order to scale and center the images to a
        particular range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale
        image to the range [-1, 1], you would have ``bias=-127.5`` and
        ``scale=1.0 / 127.5`` and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not scale the images.

    randomize_order : bool, optional
        Whether or not to randomize the order of the images as they are read.
        The order will be completely random if there is only one image
        sub-folder, however, when there are multiple, they will be read one
        folder at the time and only randomized on a per-folder basis. Use
        ``random_pool_size`` in order to achieve inter-subfolder mixing.
        Default is False, do not randomise the order of the images.

    random_pool_size : int, optional
        Since the data are read one directory at the time, the slices can only
        be randomised on a per-image basis. A random pool can therefore be
        used to achieve inter-image mixing, and from which slices are selected
        one mini-batch at the time. The value of ``random_pool_size``
        determines how many images will be read and kept in the pool at the
        same time. When the number of slices in the pool falls below the
        average per-image number of slices times ``random_pool_size - 1``, a
        new image will be automatically read into the pool, and the pool will
        be shuffled again, to improve the mixing. If the
        ``random_pool_size`` is small, only a few image will be kept in the
        pool, and mini-batches may not be independent. If possible, for a
        complete mixing of all slices, the value of ``random_pool_size``
        should be set to ``len(image_names)``. Default is None, which means to
        not use the random pool. In this case, when ``randomize_order=True``,
        the images will only be randomised within each subfolder. If
        ``randomize_order=False``, the pool will not be used at all.

    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers (for the
        flip). If int, float or array_like, a new random state is created with
        the provided value as seed. If None, the default numpy random state
        (np.random) is used. Default is None, use the default numpy random
        state.

    Examples
    --------
    >>> import numpy as np
    >>> from nethin.data import DicomGenerator
    """
    def __init__(self,
                 dir_path,
                 image_names,
                 channel_names,
                 batch_size=1,
                 crop=None,
                 size=None,
                 flip=None,
                 crop_center=True,
                 keep_aspect_ratio=True,
                 minimum_size=True,
                 interp="bilinear",
                 restart_generation=False,
                 bias=None,
                 scale=None,
                 randomize_order=False,
                 random_pool_size=None,
                 data_format=None,
                 random_state=None):

        if not _HAS_DICOM:
            raise RuntimeError('The "dicom" package is not available.')

        self.dir_path = str(dir_path)
        self.image_names = [str(name) for name in image_names]
        
        self.channel_names = []
        for channel in channel_names:
            if isinstance(channel, str):
                self.channel_names.append([str(channel)])
            elif isinstance(channel, (list, tuple)):
                self.channel_names.append([str(name) for name in channel])
            else:
                raise ValueError('``channel_names`` must be a list of either '
                                 'strings or lists of strings.')

        if batch_size is None:
            self.batch_size = batch_size
        else:
            self.batch_size = max(1, int(batch_size))

        if crop is None:
            self.crop = crop
        else:
            self.crop = (max(1, int(crop[0])), max(1, int(crop[1])))

        if size is None:
            self.size = size
        else:
            self.size = (max(1, int(size[0])), max(1, int(size[1])))

        if flip is None:
            self.flip = flip
        else:
            self.flip = max(0.0, min(float(flip), 1.0))

        self.crop_center = bool(crop_center)
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.minimum_size = bool(minimum_size)
        self.interp = str(interp)
        self.restart_generation = bool(restart_generation)

        if bias is None:
            self.bias = None
        else:
            if isinstance(bias, (float, int)):
                self.bias = [float(bias) for i in range(len(image_names))]
            else:
                self.bias = [float(bias_) for bias_ in bias]

        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, (float, int)):
                self.scale = [float(scale) for i in range(len(image_names))]
            else:
                self.scale = [float(scale_) for scale_ in scale]

        self.randomize_order = bool(randomize_order)

        if random_pool_size is None:
            self.random_pool_size = None
        else:
            self.random_pool_size = max(1, int(random_pool_size))

        self.data_format = conv_utils.normalize_data_format(data_format)

        if random_state is None:
            self.random_state = np.random.random.__self__
        else:
            if isinstance(random_state, (int, float, np.ndarray)):
                self.random_state = np.random.RandomState(seed=random_state)
            elif isinstance(random_state, np.random.RandomState):
                self.random_state = random_state
            elif hasattr(random_state, "rand") and \
                    hasattr(random_state, "randint") and \
                    hasattr(random_state, "choice"):  # E.g., np.random
                self.random_state = random_state
            else:  # May crash here..
                self.random_state = np.random.RandomState(seed=random_state)

        self._image_i = 0
        self._file_queue = []
        for i in range(len(self.channel_names)):
            self._file_queue.append([])

        # Fill the queue with slices from random_pool_size images
        if self.random_pool_size is None:
            num_pool_updates = 1
        else:
            num_pool_updates = self.random_pool_size
        actual_pool_updates = 0
        for i in range(num_pool_updates):
            update_done = self._file_queue_update()
            if update_done:
                actual_pool_updates += 1

        self._average_num_slices = int(
                (self._file_queue_len() / actual_pool_updates) + 0.5)

        self._throw_stop_iteration = False

    def _read_next_image(self):
        """Extracts the file names for all channels of the next image.
        """
        dir_path = self.dir_path  # "~/data"
        image_names = self.image_names  # ["Patient 1", "Patient 2"]
        channel_names = self.channel_names  # [["CT.*", "[CT].*"], ["MR.*"]]

        if self._image_i >= len(image_names):
            if self.restart_generation:
                self._image_i = 0
            else:
                return None

        image_name = image_names[self._image_i]  # "Patient 1"
        image_path = os.path.join(dir_path, image_name)  # "~/data/Patient 1"
        possible_channel_dirs = os.listdir(image_path)  # ["CT", "MR"]
        channel_dirs = []
        for channel_name in channel_names:  # channel_name = ["CT.*", "[CT].*"]
            found = False
            for channel_re in channel_name:  # channel_re = "CT.*"
                regexp = re.compile(channel_re)
                for channel in possible_channel_dirs:  # channel = "CT"
                    if regexp.match(channel):
                        channel_dirs.append(channel)  # channel_dirs = ["CT"]
                        found = True
                    if found:
                        break
                if found:
                    break
            else:
                raise RuntimeError("Channel %s was not found for image %s"
                                   % (channel_re, image_name))
        # channel_dirs = ["CT", "MR"]

        all_channel_files = []
        channel_length = None
        for channel_dir_i in range(len(channel_dirs)):  # 0
            channel_dir = channel_dirs[channel_dir_i]  # channel_dir = "CT"
            channel_path = os.path.join(image_path, channel_dir)  # "~/data/Patient 1/CT"
            dicom_files = os.listdir(channel_path)  # ["im1.dcm", "im2.dcm"]

            # Check that channels have the same length
            if channel_length is None:
                channel_length = len(dicom_files)
            else:
                if channel_length != len(dicom_files):
                    raise RuntimeError("The number of slices for channel %s "
                                       "and channel %d does not agree."
                                       % (channel_dir, channel_dirs[0]))

            # Create full relative or absolute path for all slices
            full_file_names = []
            for file in dicom_files:
                dicom_file = os.path.join(channel_path, file)  # "~/data/Patient 1/CT/im1.dcm"

                full_file_names.append(dicom_file)

            all_channel_files.append(full_file_names)

        self._image_i += 1

        return all_channel_files

    def _file_queue_update(self):

        image = self._read_next_image()
        # None if there are no more images to read from the list of images
        if image is not None:

            if self.randomize_order:
                indices = None

            for channels_i in range(len(self._file_queue)):
                files = self._file_queue[channels_i]
                files.extend(image[channels_i])

                if self.randomize_order:
                    if indices is None:
                        # Randomize using same random order for all channels
                        indices = self.random_state.choice(len(files),
                                                           size=len(files),
                                                           replace=False).tolist()

                    new_files = [None] * len(files)
                    for i in range(len(files)):
                        new_files[i] = files[indices[i]]
                    files = new_files

                self._file_queue[channels_i] = files

            return True
        return False

    def _file_queue_len(self):

        if len(self._file_queue) == 0:
            return 0

        return len(self._file_queue[0])

    def _file_queue_pop(self):

        file_names = []
        for files_channels in self._file_queue:
            file_names.append(files_channels.pop())

        return file_names

    def _read_dicom(self, file_name):
        """Read a single channel slice for a particular image.
        """
        data = dicom.read_file(file_name)
        image = data.pixel_array.astype(float)

        # Convert to original units
        image = image * data.RescaleSlope + data.RescaleIntercept

        return image

    def _read_image(self, file_names):
        """Read all channels for a particular slice in an image.
        """
        try:
            images = []
            for file_name in file_names:
                image = self._read_dicom(file_name)

                images.append(image)

            return images

        except FileNotFoundError:
            return None

    def _process_images(self, images):
        """Process all channels of a slice.
        """
        for i in range(len(images)):
            image = images[i]
            image = self._process_image(image, i)
            images[i] = image

        return images

    def _process_image(self, image, channel_index):
        """Process all channels for a slice in an image.
        """
        if self.size is not None:
            if self.keep_aspect_ratio:
                im_size = image.shape[:2]
                factors = [float(im_size[0]) / float(self.size[0]),
                           float(im_size[1]) / float(self.size[1])]
                factor = min(factors) if self.minimum_size else max(factors)
                new_size = list(im_size[:])
                new_size[0] = int((new_size[0] / factor) + 0.5)
                new_size[1] = int((new_size[1] / factor) + 0.5)
            else:
                new_size = self.size

            image = imresize(image, new_size, interp=self.interp)

        if self.crop is not None:
            crop0 = min(image.shape[0], self.crop[0])
            crop1 = min(image.shape[1], self.crop[1])
            if self.crop_center:
                top = int(round((image.shape[0] / 2) - (crop0 / 2)) + 0.5)
                left = int(round((image.shape[1] / 2) - (crop1 / 2)) + 0.5)
            else:
                top = self.random_state.randint(
                        0, max(1, image.shape[0] - crop0))
                left = self.random_state.randint(
                        0, max(1, image.shape[1] - crop1))
            image = image[top:top + crop0, left:left + crop1]

        if self.flip is not None:
            if self.random_state.rand() < self.flip:
                image = image[:, ::-1, :]

        if self.bias is not None:
            image = image + self.bias[channel_index]

        if self.scale is not None:
            image = image * self.scale[channel_index]

        return image

    def throw(self, typ, **kwargs):
        """Raise an exception in the generator.
        """
        super(Dicom3DGenerator, self).throw(typ, **kwargs)

    def send(self, value):
        """Send a value into the generator.

        Return next yielded value or raise StopIteration.
        """
        if (self.batch_size is None) and self._throw_stop_iteration:
            self._throw_stop_iteration = False
            self.throw(StopIteration)

        return_images = []
        while (self.batch_size is None) \
                or (len(return_images) < self.batch_size):

            if self.random_pool_size is None:
                queue_lim = 1
            else:
                queue_lim = max(1, (self.random_pool_size - 1) * self._average_num_slices)
            if self._file_queue_len() < queue_lim:
                update_done = self._file_queue_update()

            file_names = None
            try:
                file_names = self._file_queue_pop()
            except (IndexError) as e:
                if not update_done:  # Images not added, no images in queue
                    if (self.batch_size is None) \
                            and (not self._throw_stop_iteration):
                        self._throw_stop_iteration = True
                        break
                    else:
                        self.throw(StopIteration)

            if file_names is not None:
                images = self._read_image(file_names)
                if images is not None:
                    images = self._process_images(images)
                    images = np.array(images)
                    if self.data_format == "channels_last":
                        images = np.transpose(images, (1, 2, 0))  # Channels last
                    return_images.append(images)

        return return_images


class DicomGenerator(BaseGenerator):
    """A generator over Dicom images in a given directory.

    It will be assumed that the Dicom files have some particular tags. They are
    assumed to have: "RescaleSlope", "RescaleIntercept", "Rows", "Columns".

    This generator requires that the ``dicom`` package be installed.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the dicom files.

    recursive : bool, optional
        Whether or not to traverse the given directory recursively. Default is
        False, do not traverse the directory recursively.

    batch_size : int, optional
        The number of images to return at each yield. Default is 1, return only
        one image at the time. If there are not enough images to return in one
        batch, the source directory is considered exhausted, and StopIteration
        will be thrown.

    crop : tuple of int, length 2, optional
        A subimage size to crop randomly from the read image. If any images are
        smaller than crop in any direction, no cropping will be performed in
        that direction. Default is None, do not perform any cropping.

    size : tuple of int, length 2, optional
        The (possibly cropped image) will be resized to this absolute size.
        Default is None, do not resize the images. See also
        ``keep_aspect_ratio`` and ``minimum_size``.

    flip : float, optional
        The probability of flipping the image in the left-right direction.
        Default is None, which means to not flip the image (equivalent to
        ``flip=0.0``.

    crop_center : bool, optional
        Whether or not to select the middle portion of the image when cropping,
        or to select random crop positions. Default is True, select the center
        of the image when cropping.

    keep_aspect_ratio : bool, optional
        Whether or not to keep the aspect ratios of the images when resizing.
        Only used if size it not None. Default is True, keep the aspect ratio
        of the original image. See also ``minimum_size``.

    minimum_size : bool, optional
        If ``keep_aspect_ratio=True``, then ``minimum_size`` determines if the
        given size is the minimum size (scaled image is equal to or larger than
        the given ``size``) or the maximum size (scaled image is equal to or
        smaller than the given ``size``) of the scaled image. Default is True,
        the scaled image will be at least as large as ``size``. See also
        ``keep_aspect_ratio``.

    interp : str, optional
        Interpolation to use for re-sizing ("nearest", "lanczos", "bilinear",
        "bicubic" or "cubic"). Default is "bilinear".

    restart_generation : bool, optional
        Whether or not to start over from the first file again after the
        generator has finished. Default is False, do not start over again.

    bias : float, optional
        A bias to add to the generated images. Use this in conjunction with
        ``scale`` in order to scale and center the images to a particular
        range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale image to the
        range [-1, 1], you would have ``bias=-127.5`` and ``scale=1.0 / 127.5``
        and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not add a bias.

    scale : float, optional
        A factor to use to scale the generated images. Use this in
        conjunction with ``bias`` in order to scale and center the images to a
        particular range, e.g. to [-1, 1]. E.g., to scale a 8-bit grey-scale
        image to the range [-1, 1], you would have ``bias=-127.5`` and
        ``scale=1.0 / 127.5`` and the operation would thus be

            I = (I + bias) * scale = (I - 127.5) / 127.5

        Default is None, which means to not scale the images.

    randomize_order : bool, optional
        Whether or not to randomize the order of the images as they are read.
        The order will be completely random if there are no sub-folders or
        ``recursive=False``. When there are sub-folders, and they are read
        recursively, they will be read one folder at the time and only
        randomized on a per-folder basis. Use ``random_pool_size`` in order to
        achieve inter-subfolder mixing. Default is False, do not randomise the
        order of the images.

    random_pool_size : int, optional
        Since the data are read one sub-folder at the time, the images can only
        be randomised on a per-folder basis. A random pool can therefore be
        used to achieve inter-folder mixing, and from which images are selected
        one mini-batch at the time. The value of ``random_pool_size``
        determines how many images will be read and kept in the pool at the
        same time. When the number of iamges in the pool falls below the given
        value, new images will be automatically read into the pool, and the
        pool will be shuffled again to improve the mixing. If the
        ``random_pool_size`` is small, only a few image will be kept in the
        pool, and mini-batches may not be independent. If possible, for a
        complete mixing of all images, the value of ``random_pool_size``
        should be set equal to the total number of images in ``dir_path`` and
        its subfolders (if ``recursive=True``). Default is None, which means to
        not use the random pool. In this case, when ``randomize_order=True``,
        the images will only be randomised within each sub-folder. If
        ``randomize_order=False``, the pool will not be used at all.

    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers (for the
        flip and for the random order). If int, float or array_like, a new
        random state is created with the provided value as seed. If None, the
        default numpy random state (np.random) is used. Default is None, use
        the default numpy random state.

    Examples
    --------
    >>> import numpy as np
    >>> from nethin.data import DicomGenerator
    """
    def __init__(self,
                 dir_path,
                 recursive=False,
                 batch_size=1,
                 crop=None,
                 size=None,
                 flip=None,
                 crop_center=True,
                 keep_aspect_ratio=True,
                 minimum_size=True,
                 interp="bilinear",
                 restart_generation=False,
                 bias=None,
                 scale=None,
                 randomize_order=False,
                 random_pool_size=None,
                 data_format=None,
                 random_state=None):

        if not _HAS_DICOM:
            raise RuntimeError('The "dicom" package is not available.')

        self.dir_path = str(dir_path)
        self.recursive = bool(recursive)
        self.batch_size = max(1, int(batch_size))

        if crop is None:
            self.crop = crop
        else:
            self.crop = (max(1, int(crop[0])), max(1, int(crop[1])))

        if size is None:
            self.size = size
        else:
            self.size = (max(1, int(size[0])), max(1, int(size[1])))

        if flip is None:
            self.flip = flip
        else:
            self.flip = max(0.0, min(float(flip), 1.0))

        self.crop_center = bool(crop_center)
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.minimum_size = bool(minimum_size)

        allowed_interp = ("nearest", "lanczos", "bilinear", "bicubic", "cubic")
        self.interp = str(interp).lower()
        if self.interp not in allowed_interp:
            raise ValueError("The ``interp`` parameter must be one of " +
                             str(allowed_interp))

        self.restart_generation = bool(restart_generation)

        if bias is None:
            self.bias = None
        else:
            self.bias = float(bias)

        if scale is None:
            self.scale = None
        else:
            self.scale = float(scale)

        self.randomize_order = bool(randomize_order)

        if random_pool_size is None:
            self.random_pool_size = None
        else:
            self.random_pool_size = max(self.batch_size, int(random_pool_size))

        self.data_format = conv_utils.normalize_data_format(data_format)

        if random_state is None:
            self.random_state = np.random.random.__self__
        else:
            if isinstance(random_state, (int, float, np.ndarray)):
                self.random_state = np.random.RandomState(seed=random_state)
            elif isinstance(random_state, np.random.RandomState):
                self.random_state = random_state
            elif hasattr(random_state, "rand") and \
                    hasattr(random_state, "randint") and \
                    hasattr(random_state, "shuffle"):  # E.g., np.random
                self.random_state = random_state
            else:  # May crash here..
                self.random_state = np.random.RandomState(seed=random_state)

        self._walker = None
        self._restart_walker()

        self._image_i = 0
        self._file_queue = []

        # Fill the queue with random_pool_size images, if possible
        if self.random_pool_size is None:
            pool_size = 1
        else:
            pool_size = self.random_pool_size
        while self._file_queue_len() < pool_size:
            if not self._file_queue_update(throw=False):  # No more files
                break

    def _restart_walker(self):

        if self._walker is not None:
            self._walker.close()
        self._walker = os.walk(self.dir_path)

    def _file_queue_len(self):

        return len(self._file_queue)

    def _file_queue_push(self, file):

        # Append on the right
        self._file_queue.append(file)

    def _file_queue_pop(self):

        # Pop on the left
        file_name = self._file_queue[0]
        del self._file_queue[0]

        return file_name

    def _file_queue_randomize(self):

        self.random_state.shuffle(self._file_queue)  # Shuffle in-place

    def _file_queue_update(self, throw=True):

        try_again = True
        tries = 0
        while try_again and (tries <= 1):
            try_again = False
            try:
                dir_name, sub_dirs, files = next(self._walker)

                for i in range(len(files)):
                    file = os.path.join(dir_name, files[i])
                    self._file_queue_push(file)

                if self.randomize_order:
                    self._file_queue_randomize()

                if not self.recursive:
                    self._walker.close()

                return True

            except StopIteration as e:
                if self.restart_generation:
                    self._restart_walker()
                    try_again = True
                    tries += 1  # Only try to restart again once
                else:
                    if throw:
                        self.throw(e)

            except Exception as e:
                if throw:
                    self.throw(e)

        return False  # An exception was raised

    def _read_image(self, file_name):
        """Extracts the file names for all channels of the next image.
        """
        image = self._read_dicom(file_name)

        return image

    def _read_dicom(self, file_name):
        """Read a single dicom image or return None if not a dicom file.
        """
        try:
            data = dicom.read_file(file_name)
        except (dicom.filereader.InvalidDicomError, FileNotFoundError) as e:
            return None

        image = data.pixel_array.astype(float)

        # Convert to original units
        image = image * data.RescaleSlope + data.RescaleIntercept

        return image

    def _process_image(self, image):
        """Process an image.
        """
        if self.size is not None:
            if self.keep_aspect_ratio:
                im_size = image.shape[:2]
                factors = [float(im_size[0]) / float(self.size[0]),
                           float(im_size[1]) / float(self.size[1])]
                factor = min(factors) if self.minimum_size else max(factors)
                new_size = list(im_size[:])
                new_size[0] = int((new_size[0] / factor) + 0.5)
                new_size[1] = int((new_size[1] / factor) + 0.5)
            else:
                new_size = self.size

            image = imresize(image, new_size, interp=self.interp)

        if self.crop is not None:
            crop0 = min(image.shape[0], self.crop[0])
            crop1 = min(image.shape[1], self.crop[1])
            if self.crop_center:
                top = int(round((image.shape[0] / 2) - (crop0 / 2)) + 0.5)
                left = int(round((image.shape[1] / 2) - (crop1 / 2)) + 0.5)
            else:
                top = self.random_state.randint(
                        0, max(1, image.shape[0] - crop0))
                left = self.random_state.randint(
                        0, max(1, image.shape[1] - crop1))
            image = image[top:top + crop0, left:left + crop1]

        if self.flip is not None:
            if self.random_state.rand() < self.flip:
                image = image[:, ::-1, :]

        if self.bias is not None:
            image = image + self.bias

        if self.scale is not None:
            image = image * self.scale

        return image

    def throw(self, typ, **kwargs):
        """Raise an exception in the generator.
        """
        super(DicomGenerator, self).throw(typ, **kwargs)

    def send(self, value):
        """Send a value into the generator.

        Return next yielded value or raise StopIteration.
        """
        return_images = []
        while len(return_images) < self.batch_size:
            if self.random_pool_size is None:
                queue_lim = 1
            else:
                queue_lim = self.random_pool_size
            if self._file_queue_len() < queue_lim:
                update_done = self._file_queue_update(throw=False)

            file_name = None
            try:
                file_name = self._file_queue_pop()
            except (IndexError) as e:
                if not update_done:  # Images not added, no images in queue
                    self.throw(StopIteration)

            if file_name is not None:
                image = self._read_image(file_name)
                if image is not None:
                    image = self._process_image(image)

                    return_images.append(image)

        return return_images


class Numpy3DGenerator(BaseGenerator):
    """A generator over 3D images in a given directory, saved as npz files.

    It will be assume that all npz files in the directory are to be read. The
    returned data will have shape ``(batch_size, height, width, channels)`` or
    ``(batch_size, channels, height, width)``, depending on the value of
    ``data_format``.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the images.

    image_key : str
        The string name of the key in the loaded ``NpzFile`` that contains the
        image.

    file_names : list of str or None, optional
        A list of npz filenames to read or None, which means to read all npz
        files. Default is None, read all npz files in the given directory.

    batch_size : int or None, optional
        The number of images to return at each yield. If None, all images will
        be returned. If there are not enough images to return in one batch, the
        source directory is considered exhausted, and StopIteration will be
        thrown. Default is 1, which means to return only one image at the time.

    restart_generation : bool, optional
        Whether or not to start over from the first file again after the
        generator has finished. Default is False, do not start over again.

    randomize_order : bool, optional
        Whether or not to randomise the order of the images as they are read.
        The order will be completely random if ``random_pool_size`` is the same
        size as the number of npz files in the given directory.
        Otherwise, only ``random_pool_size`` npz files will be loaded at any
        given time and only randomized within the loaded set of files. Use
        ``random_pool_size`` in order to control the amount of randomness in
        the outputs. Default is False, do not randomise the order of the
        images.

    random_pool_size : int or None, optional
        Since the data will be read one npz file at the time, the slices can
        only be randomised on a per-file basis. A random pool can therefore be
        used to achieve inter-file mixing, and from which slices are selected
        one mini-batch at the time. The value of ``random_pool_size``
        determines how many files will be read and kept in the pool at the
        same time. When the number of slices in the pool falls below the
        average per-image number of slices times ``random_pool_size - 1``, a
        new image will be automatically loaded into the pool, and the pool will
        be reshuffled, to improve the mixing. If the ``random_pool_size`` is
        small, only a few image will be kept in the pool at any given time, and
        mini-batches may not be independent. If possible, for a complete mixing
        of all slices, the value of ``random_pool_size`` should be set to
        the number of files in the given directory. Default is None, which
        means to not use the random pool. In this case, when
        ``randomize_order=True``, the slices will only be randomised within
        each file. If ``randomize_order=False``, the pool will not be used at
        all.

    data_format : str, optional
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    random_state : int, float, array_like or numpy.random.RandomState, optional
        A random state to use when sampling pseudo-random numbers (for the
        flip). If int, float or array_like, a new random state is created with
        the provided value as seed. If None, the default numpy random state
        (np.random) is used. Default is None, use the default numpy random
        state.

    Examples
    --------
    >>> import numpy as np
    >>> from nethin.data import DicomGenerator
    """
    def __init__(self,
                 dir_path,
                 image_key,
                 file_names=None,
                 batch_size=1,
                 restart_generation=False,
                 randomize_order=False,
                 random_pool_size=None,
                 data_format=None,
                 random_state=None):

        self.dir_path = str(dir_path)
        self.image_key = str(image_key)

        if file_names is None:
            self.file_names = None
        else:
            self.file_names = [str(file_name) for file_name in file_names]

        if batch_size is None:
            self.batch_size = batch_size
        else:
            self.batch_size = max(1, int(batch_size))

        self.restart_generation = bool(restart_generation)
        self.randomize_order = bool(randomize_order)

        if random_pool_size is None:
            self.random_pool_size = None
        else:
            self.random_pool_size = max(1, int(random_pool_size))

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.random_state = utils.normalize_random_state(random_state,
                                                         rand_functions=["rand",
                                                                         "randint",
                                                                         "choice"])

        self._all_files = self._list_dir()  # ["image1.npz", "image2.npz"]
        self._file_i = 0
        self._average_num_slices = 1.0
        self._num_updates = 0
        self._slice_queue = []

        # Attempt to fill the queue
        for file_i in range(len(self._all_files)):
            self._file_i = file_i

            if self._slice_queue_update(self._all_files[file_i]):
                if len(self._slice_queue) > 0:
                    break  # There are valid files with slices
        if len(self._slice_queue) == 0:
            raise ValueError("The given directory does not contain any valid "
                             "npz files.")

        self._throw_stop_iteration = False

    def _list_dir(self):
        all_files = os.listdir(self.dir_path)
        all_images = []
        for file in all_files:
            if file.endswith(".npz"):
                if os.path.isfile(os.path.join(self.dir_path, file)):
                    if self.file_names is None:
                        all_images.append(file)
                    elif file in self.file_names:
                        all_images.append(file)

        return all_images

    def _read_file(self, file):

        image = np.load(file)

        return image[self.image_key]

    def _slice_queue_update(self, file):

        try:
            full_file = os.path.join(self.dir_path, file)
            image = self._read_file(full_file)
        except (KeyError) as e:
            raise ValueError('Key "%s" does not exist in image file "%s".'
                             % (self.image_key, file))
        except:
            return False

        num_slices = image.shape[0]
        for i in range(num_slices):
            self._slice_queue_push(image[i, ...])

        self._average_num_slices \
            = (self._average_num_slices * self._num_updates + num_slices) \
            / float(self._num_updates + 1)

        self._num_updates += 1

        if self.randomize_order:
            self._slice_queue_shuffle()

        return True

    def _slice_queue_push(self, value):

        self._slice_queue.append(value)

    def _slice_queue_pop(self):

        return self._slice_queue.pop(0)

    def _slice_queue_shuffle(self):

        indices = self.random_state.choice(len(self._slice_queue),
                                           size=len(self._slice_queue),
                                           replace=False).tolist()

        new_queue = [None] * len(self._slice_queue)
        for i in range(len(self._slice_queue)):
            new_queue[i] = self._slice_queue[indices[i]]
        self._slice_queue = new_queue

    def throw(self, typ, **kwargs):
        """Raise an exception in the generator.
        """
        super(Numpy3DGenerator, self).throw(typ, **kwargs)

    def send(self, value):
        """Send a value into the generator.

        Return next yielded value or raise StopIteration.
        """
        if self._throw_stop_iteration:
            if self.restart_generation:
                self._throw_stop_iteration = False  # Should not happen
            else:
                if (self.batch_size is None) \
                        or (len(self._slice_queue) < self.batch_size):
                    self.throw(StopIteration)
                else:
                    pass  # We are not ready to stop just yet

        if self.random_pool_size is None:
            pool_size = 1
        else:
            pool_size = self.random_pool_size

        return_images = []
        while (self.batch_size is None) \
                or (len(return_images) < self.batch_size):

            while not self._throw_stop_iteration:
                # If we have enough slices, don't update
                if len(self._slice_queue) \
                        >= self._average_num_slices * pool_size:
                    break

                self._file_i += 1
                if (self.batch_size is None) and self.restart_generation:

                    if self._file_i >= len(self._all_files):
                        self._file_i = 0  # Restart next time
                        break  # Do not read any more files now

                elif (self.batch_size is None) and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                elif (self.batch_size is not None) and self.restart_generation:

                    if self._file_i >= len(self._all_files):
                        self._file_i = 0  # Restart from first file

                else:  # (self.batch_size is not None) and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                self._slice_queue_update(self._all_files[self._file_i])

            try:
                image = self._slice_queue_pop()
                return_images.append(image)
            except (IndexError) as e:  # Empty queue
                if (self.batch_size is None):
                    break  # Done, return images
                elif (not self.restart_generation):
                    # We did not end on a full batch
                    if len(return_images) != self.batch_size:
                        self._throw_stop_iteration = True
                        self.throw(StopIteration)

        return return_images


class Dicom3DSaver(object):

    def __init__(self, path):

        self.path = str(path)

    def save(self, images, image_name, slice_name=None, tags=dict()):

        if not isinstance(images, dict) or len(images) == 0:
            raise ValueError('The "images" must be a dict, mapping channel '
                             'names to numpy arrays.')

        image = images[list(images.keys())[0]]
        if len(image.shape) != 3 and len(image.shape) != 4:
            raise ValueError('The "images" must have shape (B, R, C) or '
                             '(B, R, C, 1).')
        batch_dim = image.shape[0]

        image_name = str(image_name)

        if slice_name is None:
            num_digits = int(np.log10(batch_dim)) + 1
            slice_name = "Image%%0%dd.dcm" % num_digits
        else:
            slice_name = str(slice_name)
            try:
                slice_name % (1,)
            except TypeError:
                raise ValueError('The "slice_name" must contain an integer '
                                 'formatter ("%d").')

        if not isinstance(tags, dict):
            raise ValueError('The "tags" must be a dict.')
        for tag in tags.keys():
            if not isinstance(tag, dicom.tag.Tag):
                raise ValueError('The keys of "tags" must be of type '
                                 '"Tag".')

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        image_dir = os.path.join(self.path, image_name)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        uid = 0
        channel_id = 0
        for channel in images.keys():

            image = images[channel]

            intercept = np.min(image)
            image = image - intercept
            max_value = np.max(image)
            bits_stored = 16
            dicom_max_value = int(2**bits_stored - 1)
            if max_value <= dicom_max_value:
                slope = 1.0
            else:
                slope = float(dicom_max_value) / max_value
                image = image / slope
            image = np.round(image).astype(np.int16)

            if np.random.rand() < 0.5:
                patient_name = "Doe^John"
            else:
                patient_name = "Doe^Jane"

            for slice_i in range(batch_dim):

                filename = os.path.join(image_dir, slice_name % (slice_i,))

                file_meta = dicom.dataset.Dataset()
                file_meta.MediaStorageSOPClassUID = "CT Image Storage"
                file_meta.MediaStorageSOPInstanceUID = "1.%d.%d.%d" \
                                                       % (channel_id,
                                                          slice_i,
                                                          uid)
                uid += 1
                file_meta.ImplementationClassUID = "2.3.7.6.1.2.%s" \
                                                   % (nethin.__version__,)

                data = dicom.dataset.FileDataset(filename, {},
                                                 file_meta=file_meta,
                                                 preamble=b"\0" * 128)
                data.PatientName = patient_name
                data.PatientID = image_name

                data.is_little_endian = True
                data.is_implicit_VR = True

                dt = datetime.datetime.now()
                data.ContentDate = dt.strftime('%Y%m%d')
                data.ContentTime = dt.strftime('%H%M%S.%f')  # Long format with micro seconds

                data.InstanceNumber = slice_i
                data.RescaleIntercept = intercept
                data.RescaleSlope = slope
                # data.RescaleType = "HU"
                if len(image.shape) == 3:
                    data.PixelData = image[slice_i, :, :].tostring()
                else:
                    data.PixelData = image[slice_i, :, :, 0].tostring()

                data.Rows = image.shape[1]
                data.Columns = image.shape[2]
                data.SamplesPerPixel = 1
                data.PhotometricInterpretation = "MONOCHROME2"
                data.PlanarConfiguration = 0
                data.BitsAllocated = bits_stored  # 16
                data.BitsStored = bits_stored  # 16
                data.HighBit = bits_stored - 1  # 15

                data.PixelRepresentation = 0  # Unsigned
                data.NumberOfFrames = 1

                for tag in tags.keys():
                    data[tag] = tags[tag]

                data.save_as(filename)

            channel_id += 1
