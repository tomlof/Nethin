# -*- coding: utf-8 -*-
"""
This module contains ready-to-use functions to view images and networks.

Created on Fri Apr  6 15:55:02 2018

Copyright (c) 2017-2018, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import os
import six
import pickle

import numpy as np

import matplotlib.pyplot as plt

import keras.backend as K


__all__ = ["image_3d"]


def image_3d(images, cmap=None, vmin=None, vmax=None, title=None,
             channel_names=None, data_format=None):
    """Show a 3-dimensional image. It will show the channels in different
    subplots and slices can be flipped through by scrolling.

    Parameters
    ----------
    images : numpy.ndarray, list of numpy.ndarray, or str
        A numpy array of shape (B, H, W, C) if ``data_format="channels_last"``,
        and (B, C, H, W) if ``data_format="channels_first"``. A list of numpy
        arrays of shape (B, H, W, C)  if ``data_format="channels_last"``, and
        a list of numpy arrays of shape (B, C, H, W) if
        ``data_format="channels_first"``. If a str, it is the path to a file
        containing an image of the specified format. The file must either be a
        numpy npz file or a pickled file (binary).

    cmap : `~matplotlib.colors.Colormap`, optional, default: None
        If None, default to rc `image.cmap` value.

    vmin, vmax : scalar, or list of scalar, optional, default: None
        `vmin` and `vmax` are used to normalize the image to the
        ``[vmin, vmax]`` range. If a scalar, the same range is used for all
        channels. If a list, the number of values must be the same as the
        number of channels, with one value for each channel (can be None).

    title : str, optional, default: None
        A title string for the plot.

    channel_names : str, list of str, optional, default: None
        If given, gives subplot titles for the channels of the image.

    data_format : str, optional
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape ``(batch, channels,
        height, width)``. It defaults to the ``image_data_format`` value found
        in your Keras config file at ``~/.keras/keras.json``. If you never set
        it, then it will be "channels_last".
    """
    if hasattr(K, "normalize_data_format"):
        data_format = K.normalize_data_format(data_format)
    else:
        # TODO: Change since d40656e5799dfd22e96b8bab217638b2934a6894 (#10690).
        #       Remove when appropriate ...
        from keras.utils import conv_utils
        data_format = conv_utils.normalize_data_format(data_format)

    if data_format == "channels_last":
        channel_axis = 3
    else:  # data_format == "channels_first":
        channel_axis = 1

    if isinstance(images, six.string_types):
        filename = images

        if not os.path.exists(filename):
            raise ValueError('No such file "%s"' % (filename,))

        found = False
        try:
            images_ = np.load(filename)
            images = None
            for key in images_.keys():
                try:
                    if images_[key].ndim == 4:
                        images = images_[key]
                        del images_
                        found = True
                        break
                except:
                    pass
        except:
            pass

        if not found:
            try:
                with open(filename, "rb") as fp:
                    images_ = pickle.load(fp)
                images = None
                for val in images_:
                    try:
                        if val.ndim == 4:
                            images = val
                            del images_
                            found = True
                            break
                    except:
                        pass
            except:
                pass

        if not found:
            raise ValueError("Unable to open file! Is it an npz or pkl file?")

    elif isinstance(images, (list, tuple)):
        for i in range(0, len(images)):
            if images[i].ndim != 4:
                raise ValueError("Input images must have shape (B, H, W, C) "
                                 "or (B, C, H, W).")

        images = np.concatenate(images, axis=channel_axis)

    if isinstance(vmin, (list, tuple)):
        vmin_ = []
        for v in vmin:
            if v is not None:
                vmin_.append(float(v))
            else:
                vmin_.append(v)
        vmin = vmin_
        if len(vmin) != images.shape[channel_axis]:
            raise ValueError("vmin must have the same number of elements as "
                             "there are channels in images.")
    else:
        vmin = float(vmin) if vmin is not None else vmin
        vmin = [vmin] * images.shape[channel_axis]

    if isinstance(vmax, (list, tuple)):
        vmax_ = []
        for v in vmax:
            if v is not None:
                vmax_.append(float(v))
            else:
                vmax_.append(v)
        vmax = vmax_
        if len(vmax) != images.shape[channel_axis]:
            raise ValueError("vmax must have the same number of elements as "
                             "there are channels in images.")
    else:
        vmax = float(vmax) if vmax is not None else vmax
        vmax = [vmax] * images.shape[channel_axis]

    if channel_names is not None:
        if isinstance(channel_names, six.string_types):
            channel_names = [str(channel_names)]
        else:
            channel_names = list(channel_names)

    if images.ndim != 4:
        raise ValueError("Input image must have shape (B, H, W, C) or "
                         "(B, C, H, W).")

    if channel_names is not None:
        if len(channel_names) < images.shape[channel_axis]:
            channel_names.extend(
                    [None] * (images.shape[channel_axis] - len(channel_names)))

    ny = int(np.floor(np.sqrt(images.shape[channel_axis])) + 0.5)
    nx = int(np.ceil(images.shape[channel_axis] / float(ny)) + 0.5)

    # Format: Slice index, total number of frames, running
    handler_data = [0, images.shape[0], False]

    fig, axs = plt.subplots(nrows=ny, ncols=nx, sharex=True, sharey=True)
    if (ny == 1) and (nx == 1):
        axs = np.asarray([axs])
    axs = axs.reshape(ny, nx)

    def draw(slice_i, first_run=False):
        for i in range(images.shape[channel_axis]):
            y = i // nx
            x = i % nx
            if data_format == "channels_last":
                axs[y, x].imshow(images[slice_i, :, :, i],
                                 cmap=cmap, vmin=vmin[i], vmax=vmax[i])
            else:
                axs[y, x].imshow(images[slice_i, i, :, :],
                                 cmap=cmap, vmin=vmin[i], vmax=vmax[i])

        for i in range(images.shape[channel_axis]):
            y = i // nx
            x = i % nx
            if (channel_names is None) or (channel_names[i] is None):
                axs[y, x].set_title("Channel %d (slice %d)"
                                    % (i + 1, slice_i))
            else:
                axs[y, x].set_title(str(channel_names[i]) + " (slice %d)"
                                    % (slice_i,))

        if first_run:
            for i in range(images.shape[channel_axis], ny * nx):
                y = i // nx
                x = i % nx
                axs[y, x].axis("off")

            if title is not None:
                plt.suptitle(title)

        plt.pause(0.0001)

    def scroll_handler(event):
        if handler_data[2]:
            return
        if event.button == "up":
            handler_data[0] = min(handler_data[0] + 1, handler_data[1] - 1)
            try:
                handler_data[2] = True
                draw(handler_data[0])
                handler_data[2] = False
            except:
                handler_data[2] = False
        elif event.button == "down":
            handler_data[0] = max(0, handler_data[0] - 1)
            try:
                handler_data[2] = True
                draw(handler_data[0])
                handler_data[2] = False
            except:
                handler_data[2] = False
        else:
            pass

    draw(0, first_run=True)
    plt.show()

    fig.canvas.mpl_connect("scroll_event", scroll_handler)
