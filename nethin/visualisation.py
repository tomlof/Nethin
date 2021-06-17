#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for visualisation of deep neural networks.

Created on Thu Apr 30 09:53:31 2020

Copyright (c) 2017-2021, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["occlusion"]


def occlusion(images,
              targets,
              model,
              window=16,
              iterations=1000,
              value=None,
              relative_score=True,
              metric="loss"):

    if len(images.shape) < 4:
        images = images[np.newaxis, ...]  # Add batch dimension
        batch_dim = False
    elif len(images.shape) == 4:
        batch_dim = True
    if len(images.shape) != 4:
        raise ValueError("The input image(s) should have shape (N, W, H, C), "
                         "(N, C, W, H), (W, H, C), or (C, W, H).")

    from tensorflow.keras import backend as K
    if K.image_data_format() == "channels_last":
        channels_last = True
        H, W = images.shape[1:3]
    else:
        channels_last = False
        H, W = images.shape[2:4]

    if isinstance(window, (list, tuple)):
        if len(window) != 2:
            raise ValueError("The 'window' should be an int or a list of "
                             "two ints.")
        window = [max(1, int(window[0])), max(1, int(window[1]))]
    else:
        window = max(1, int(window))
        window = [window, window + 1]

    iterations = max(0, int(iterations))

    if value is None:
        value = np.mean(images, axis=(1, 2, 3))
    elif isinstance(value, (int, float)):
        value = [float(value)] * images.shape[0]
    else:
        value = np.asarray(value).ravel()
        if value.size != images.shape[0]:
            raise ValueError("The 'value' should be an int, float, list of "
                             "int or float, or a 1-dimensional numpy array "
                             "of ints or floats.")

    if metric is not None:
        metric = str(metric)
    # Find metric
    metric_idx = -1
    for idx, name in enumerate(model.metrics_names):
        if name == metric:
            metric_idx = idx
    if metric_idx < 0:
        raise ValueError(f"The metric '{metric}' was not found. The "
                         f"available metrics are: {model.metrics_names}.")

    n = images.shape[0]
    baseline_score = model.evaluate(images, targets, verbose=0)
    baseline_score = baseline_score[metric_idx]
    output = np.zeros(images.shape, dtype=np.float32)
    output_counts = np.zeros(images.shape, dtype=np.int32)
    for it in range(iterations):
        w = np.random.randint(*window)
        y = np.random.randint(0, H - w + 1)
        x = np.random.randint(0, W - w + 1)
        for i in range(n):
            if channels_last:
                original_values = np.copy(images[i, y:y + w, x:x + w, :])
            else:
                original_values = np.copy(images[i, :, y:y + w, x:x + w])
            try:
                if channels_last:
                    images[i, y:y + w, x:x + w, :] = value[i]
                else:
                    images[i, :, y:y + w, x:x + w] = value[i]
                score = model.evaluate(images, targets, verbose=0)
            finally:
                if channels_last:
                    images[i, y:y + w, x:x + w, :] = original_values
                else:
                    images[i, :, y:y + w, x:x + w] = original_values

            score = score[metric_idx]

            score = score - baseline_score
            if relative_score:
                score /= baseline_score + np.finfo(np.float32).eps

            if channels_last:
                output[i, y:y + w, x:x + w, :] += score
                output_counts[i, y:y + w, x:x + w, :] += 1
            else:
                output[i, :, y:y + w, x:x + w] += score
                output_counts[i, :, y:y + w, x:x + w] += 1

    output_ = output / (output_counts + np.finfo(np.float32).eps)

    if batch_dim:
        return output_
    else:
        return output_[0, ...]
