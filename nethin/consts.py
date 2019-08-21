# -*- coding: utf-8 -*-
"""
Contains constants used in the package.

Copyright (c) 2017-2019, Tommy Löfstedt. All rights reserved.

Created on Sat Dec  2 22:02:23 2017

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["NETHIN_SAVE_KEY", "MACHINE_EPSILON"]

NETHIN_SAVE_KEY = "_nethin_model_config"

MACHINE_EPSILON = np.finfo(np.float64).eps
