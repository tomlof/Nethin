# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:02:33 2017

Copyright (c) 2017-2019, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from nethin import augmentation
from nethin import consts
from nethin import data
from nethin import hyper
from nethin import models
from nethin import normalization
# from nethin import optimizers
from nethin import padding
from nethin import trainers
from nethin import utils

# models = utils.LazyImport("nethin.models")
# normalization = utils.LazyImport("nethin.normalization")
# padding = utils.LazyImport("nethin.padding")

__version__ = "0.0.1"

__all__ = ["augmentation", "consts", "data", "hyper", "models",
           "normalization",
           # "optimizers",
           "padding", "trainers",
           "utils"]
