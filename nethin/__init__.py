# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:02:33 2017

Copyright (c) 2017-2019, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from nethin import consts  # no keras
from nethin import utils  # tf.keras only

from nethin import augmentation  # tf.keras only
from nethin import data  # tf.keras only
from nethin import hyper  # no keras

from nethin import constraints  # tf.keras only
from nethin import layers  # tf.keras only
from nethin import models  # tf.keras only, but needs restructuring!
from nethin import normalization  # tf.keras only
from nethin import padding  # tf.keras only

from nethin import trainers  # tf.keras only

# from nethin import optimizers

__version__ = "0.0.1"

__all__ = ["consts", "utils",

           "augmentation", "data", "hyper",

           "constraints", "layers", "models", "normalization", "padding",

           # "optimizers",

           "trainers"
           ]
