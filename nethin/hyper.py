# -*- coding: utf-8 -*-
"""
This module contains functionality for hyperparameter optimisation.

Created on Thu Feb 21 15:28:16 2019

Copyright (c) 2017-2019, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import copy
import inspect

import numpy as np
import scipy.optimize
import scipy.stats as stats

import nethin.utils as utils
import nethin.consts as consts

try:
    import skopt
    _HAS_SKOPT = True
except ModuleNotFoundError:
    skopt = None
    _HAS_SKOPT = False


__all__ = ["UniformPrior", "LogUniformPrior",
           "Real", "Integer",
           "Space",
           "GaussianProcessRegression",
           "HyperParameterOptimization"]


class BasePrior(abc.ABC):

    @abc.abstractmethod
    def rvs(self, size=1, random_state=None):
        pass

    @abc.abstractmethod
    def transform(self, x):
        pass

    @abc.abstractmethod
    def inverse_transform(self, x):
        pass


class UniformPrior(BasePrior):

    # TODO: Handle closed, open, and half-open intervals.

    def __init__(self, low, high):
        self.low = min(float(low), float(high))
        self.high = max(float(low), float(high))

    def rvs(self, size=1, random_state=None):

        if isinstance(size, (int, float)):
            size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            size = tuple([max(0, int(s + 0.5)) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

        random_state = utils.normalize_random_state(random_state)

        samples = stats.uniform.rvs(loc=0.0,
                                    scale=1.0,
                                    size=size,
                                    random_state=random_state)

        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform the prior's values to the default range [0, 1].
        """
        return np.clip(
                ((X - self.low) / (self.high - self.low)).astype(np.float),
                0.0,
                1.0)

    def inverse_transform(self, Y):
        """Transform values from [0, 1] back to the prior's range.
        """
        return np.clip(Y * (self.high - self.low) + self.low,
                       self.low,
                       self.high)


class LogUniformPrior(BasePrior):

    # TODO: Handle closed, open, and half-open intervals.

    def __init__(self, low, high):
        self.low = max(consts.MACHINE_EPSILON, min(float(low), float(high)))
        self.high = max(float(low), float(high))

        self._log_low = np.log10(self.low)
        self._log_high = np.log10(self.high)

    def rvs(self, size=1, random_state=None):

        if isinstance(size, (int, float)):
            size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            size = tuple([max(0, int(s + 0.5)) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

        random_state = utils.normalize_random_state(random_state)

        samples = stats.uniform.rvs(loc=0.0,
                                    scale=1.0,
                                    size=size,
                                    random_state=random_state)

        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform the prior's values to the default range [0, 1].
        """
        return np.clip(
                ((np.log10(X) - self._log_low)
                 / (self._log_high - self._log_low)).astype(np.float),
                0.0,
                1.0)

    def inverse_transform(self, Y):
        """Transform values from [0, 1] back to the prior's range.
        """
        return np.clip(
                10.0 ** (Y * (self._log_high - self._log_low) + self._log_low),
                self.low,
                self.high)


class Dimension(abc.ABC):
    """Base class for hyper-parameters.
    """
    def __init__(self,
                 name,
                 prior,
                 size=1):

        self.name = str(name)

        if not isinstance(prior, BasePrior):
            raise ValueError("The 'prior' must be of type 'BasePrior'.")
        self.prior = prior

        if isinstance(size, (int, float)):
            self.size = (max(0, int(size)),)
        elif isinstance(size, (list, tuple)):
            self.size = tuple([int(s) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

    def rvs(self, size=1, random_state=None):
        return self.prior.rvs(size=size, random_state=random_state)

    def transform(self, X):
        """Transform the prior's values to the default range [0, 1].
        """
        return self.prior.transform(X)

    def inverse_transform(self, Y):
        """Transform values from [0, 1] back to the prior's range.
        """
        return self.prior.inverse_transform(Y)

    @abc.abstractmethod
    def _to_skopt(self):
        pass


class Real(Dimension):

    def __init__(self,
                 name,
                 prior,
                 size=1):

        super(Real, self).__init__(name,
                                   prior,
                                   size=size)

    def _to_skopt(self):
        if not _HAS_SKOPT:
            raise RuntimeError("This operation requires 'scikit-optimize'.")

        if isinstance(self.prior, UniformPrior):
            low = self.prior.low
            high = self.prior.high
            return skopt.space.Real(low,
                                    high,
                                    prior="uniform",
                                    transform="normalize",
                                    name=self.name)
        elif isinstance(self.prior, LogUniformPrior):
            low = self.prior.low
            high = self.prior.high
            return skopt.space.Real(low,
                                    high,
                                    prior="log-uniform",
                                    transform="normalize",
                                    name=self.name)
        else:
            raise RuntimeError("This operation uses 'scikit-optimize', and it "
                               "does not work with the given prior.")


class Integer(Dimension):

    def __init__(self,
                 name,
                 prior,
                 size=1):

        super(Integer, self).__init__(name,
                                      prior,
                                      size=size)

    def rvs(self, size=1, random_state=None):
        value = super().rvs(size=size, random_state=random_state)
        return int(value + 0.5)

    def _to_skopt(self):
        if not _HAS_SKOPT:
            raise RuntimeError("This operation requires 'scikit-optimize'.")

        if isinstance(self.prior, UniformPrior):
            low = self.prior.low
            high = self.prior.high
            return skopt.space.Integer(int(low + 0.5),
                                       int(high + 0.5),
                                       transform="normalize",
                                       name=self.name)

        elif isinstance(self.prior, LogUniformPrior):
            low = self.prior.low
            high = self.prior.high
            value = skopt.space.Real(low,
                                     high,
                                     prior="log-uniform",
                                     transform="normalize",
                                     name=self.name)
            return int(value + 0.5)

        else:
            raise RuntimeError("This operation uses 'scikit-optimize', and it "
                               "does not work with the given prior.")


class Space(object):

    def __init__(self, dimensions):
        if not isinstance(dimensions, (list, tuple)):
            raise ValueError("The 'dimensions' should be a list/tuple with "
                             "elements of type 'Dimension'.")
        self.dimensions = []
        names = set()
        for dim in dimensions:
            if not isinstance(dim, Dimension):
                raise ValueError("The 'dimensions' should be a list/tuple "
                                 "with elements of type 'Dimension'.")
            else:
                if dim.name in names:
                    raise ValueError("Two dimensions can not have the same "
                                     "name (%s)." % (dim.name,))
                else:
                    names.add(dim.name)

                self.dimensions.append(dim)

        if len(names) != len(self.dimensions):
            raise ValueError()

    def rvs(self, size=1, random_state=None):

        random_state = utils.normalize_random_state(random_state)

        dims = []
        for dim in self.dimensions:
            dims.append(dim.rvs(size=size, random_state=random_state))

        return dims

    def _to_skopt(self):

        if not _HAS_SKOPT:
            raise RuntimeError("This operation requires 'scikit-optimize'.")

        dims = []
        for dim in self.dimensions:
            dims.append(dim._to_skopt())

        return dims


class BaseAcquisitionFunction(abc.ABC):

    def __init__(self, optimizer):

        if not isinstance(optimizer, BaseAcquisitionOptimizer):
            raise ValueError("The 'optimizer' must be of type "
                             "'BaseAcquisitionOptimizer'.")

        self.optimizer = optimizer

    @abc.abstractmethod
    def _to_skopt(self):
        pass


class ExpectedImprovement(BaseAcquisitionFunction):

    def _to_skopt(self):
        return "EI"


class BaseAcquisitionOptimizer(abc.ABC):

    @abc.abstractmethod
    def _to_skopt(self):
        pass


class LBFGS(BaseAcquisitionOptimizer):

    def _to_skopt(self):
        return "lbfgs"


class BaseMinimizer(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, f, space, random_state=None):
        pass


class GaussianProcessRegression(BaseMinimizer):

    def __init__(self,
                 space,
                 noise=None,
                 acquisition_function=None,
                 result=None,
                 random_state=None):

        if not _HAS_SKOPT:
            raise ValueError("This minimiser requires 'scikit-optimize'.")

        if not isinstance(space, Space):
            raise ValueError("The 'space' must be of type 'Space'.")
        self.dimensions = skopt.utils.normalize_dimensions(space._to_skopt())

        if noise is None:
            self.noise = None
        else:
            self.noise = max(np.sqrt(consts.MACHINE_EPSILON), float(noise))

        if not isinstance(acquisition_function, BaseAcquisitionFunction):
            raise ValueError("The 'acquisition_function' must be of type "
                             "'BaseAcquisitionFunction'.")
        self.acquisition_function = acquisition_function._to_skopt()

        self.acquisition_optimizer = \
            acquisition_function.optimizer._to_skopt()

        if result is None:
            self.result = None
        elif isinstance(result, scipy.optimize.optimize.OptimizeResult):
            self.result = result
        elif isinstance(result, (tuple, list)) \
                and len(result) == 2 \
                and isinstance(result[0], np.ndarray) \
                and isinstance(result[1], (np.ndarray, list)):
            self.result = skopt.utils.create_result(result[0], result[1])
        else:
            raise ValueError("The 'result' must either be an 'OptimizeResult' "
                             "or a tuple of two numpy arrays or lists of "
                             "lists.")

        self.random_state = utils.normalize_random_state(random_state)

        self.base_estimator_ = None
        self.optimizer_ = None

        # Default parameters
        n_points = 10000
        n_restarts_optimizer = 5
        n_jobs = 1
        xi = 0.01  # Used with "EI" and "PI"
        kappa = 1.96  # Used with "LCB"
        # n_random_starts = 1
        self.n_initial_points_ = 1

        self.acq_optimizer_kwargs_ = {
                "n_points": n_points,
                "n_restarts_optimizer": n_restarts_optimizer,
                "n_jobs": n_jobs}
        self.acq_func_kwargs_ = {"xi": xi,
                                 "kappa": kappa}

    def step(self, f):

        specs = {"args": copy.copy(inspect.currentframe().f_locals),
                 "function": inspect.currentframe().f_code.co_name}

        import skopt.utils

        if self.base_estimator_ is None:
            self.base_estimator_ = skopt.utils.cook_estimator(
                "GP",
                space=self.dimensions,
                random_state=self.random_state.randint(0,
                                                       np.iinfo(np.int32).max),
                noise=self.noise)

        # Create optimizer
        if self.optimizer_ is None:
            self.optimizer_ = skopt.optimizer.Optimizer(
                    self.dimensions,
                    self.base_estimator_,
                    n_initial_points=self.n_initial_points_,
                    acq_func=self.acquisition_function,
                    acq_optimizer=self.acquisition_optimizer,
                    random_state=self.random_state,
                    acq_optimizer_kwargs=self.acq_optimizer_kwargs_,
                    acq_func_kwargs=self.acq_func_kwargs_)

            if self.result is not None:
                X0 = self.result.x_iters
                y0 = self.result.func_vals

                for i in range(len(X0)):
                    result = self.optimizer.tell(X0[i], y0[i])
                    result.specs = specs

        next_x = self.optimizer_.ask()
        next_y = f(next_x)
        result = self.optimizer_.tell(next_x, next_y)
        result.specs = specs

        return result


class HyperParameterOptimization(object):
    """General class for hyperparameter optimization.

    Examples
    --------
    >>> import nethin.hyper as hyper
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> noise_level = 0.1
    >>> def f(x, noise_level=noise_level):
    ...     return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) \
    ...                   + np.random.randn() * noise_level
    >>> x = hyper.Real("x", hyper.UniformPrior(-2, 2))
    >>> space = hyper.Space([x])
    >>> af_minimizer = hyper.LBFGS()
    >>> acquisition_function = hyper.ExpectedImprovement(af_minimizer)
    >>> minimizer = hyper.GaussianProcessRegression(
    ...         space,
    ...         acquisition_function=acquisition_function,
    ...         result=None,  # Previous results
    ...         random_state=np.random.randint(2**31))
    >>> hyperopt = hyper.HyperParameterOptimization(minimizer)
    >>>
    >>> result = hyperopt.step(f)
    >>> np.round(result["x_iters"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([[-1.40655]])
    >>> np.round(result["func_vals"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([-0.08059])
    >>>
    >>> result = hyperopt.step(f)
    >>> np.round(result["x_iters"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([[-1.40655],
           [ 2.     ]])
    >>> np.round(result["func_vals"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([-0.08059,  0.05118])
    """
    def __init__(self, minimizer, callback=None):
        if not isinstance(minimizer, BaseMinimizer):
            raise ValueError("The 'minimizer' must be of type "
                             "'BaseMinimizer'.")
        self.minimizer = minimizer

        self.callbacks = utils.normalize_callables(callback)

    def step(self, f):

        if not callable(f):
            raise ValueError("The 'f' must be callable.")

        result = self.minimizer.step(f)

        return result

    def minimize(self, f, num_iter=10):

        if not callable(f):
            raise ValueError("The 'f' must be callable.")

        num_iter = max(1, int(num_iter))

        for i in range(num_iter):

            result = self.step(f)

            if utils.apply_callables(self.callbacks, result):
                break

        return result





#class LogRange(RangeType):
#
#    def __init__(self, start, stop, dtype=float, base=10, size=None):
#
#        super(LogRange, self).__init__(start, stop, dtype=dtype, size=size)
#
#        self.base = float(base)
#
#    def get_random(self):
#
#        ret = []
#        for i in range(1 if self.size is None else self.size):
#            if self.dtype is int:
#                rand = np.random.randint(self.start, self.stop + 1)
#                ret.append(int((self.base**rand) + 0.5))
#            else:
#                rand = np.random.rand() * (self.stop - self.start) + self.start
#                ret.append(self.base**rand)
#
#        if self.size is None:
#            ret = ret[0]
#
#        return ret
#
#
#class CategoryRange(RangeType):
#
#    def __init__(self, categories, probs=None, size=None):
#
#        super(CategoryRange, self).__init__(size=size)
#
#        self.categories = list(categories)
#        if probs is None:
#            self.probs = None
#        else:
#            probs = [max(0.0, min(float(prob), 1.0)) for prob in probs]
#            sum_probs = sum(probs)
#            self.probs = [prob / sum_probs for prob in probs]
#            assert(len(self.probs) == len(self.categories))
#
#    def get_random(self):
#
#        ret = []
#        for i in range(1 if self.size is None else self.size):
#            if self.probs is None:
#                rand = self.categories[np.random.randint(len(self.categories))]
#            else:
#                csum = np.cumsum(self.probs)
#                r = np.random.rand()
#                for cs in range(len(csum)):
#                    if r <= csum[cs]:
#                        rand = self.categories[cs]
#                        break
#
#            ret.append(rand)
#
#        if self.size is None:
#            ret = ret[0]
#
#        return ret
#
#
#class LabelEncodeRange(RangeType):
#
#    def __init__(self, num_labels, dtype=int):
#
#        super(LabelEncodeRange, self).__init__(dtype=dtype)
#
#        self.num_labels = max(1, int(num_labels))
#
#    def get_random(self):
#
#        if self.dtype is float:
#            rand = [0.0] * self.num_labels
#            rand[np.random.randint(self.num_labels)] = 1.0
#        elif self.dtype is int:
#            rand = [0] * self.num_labels
#            rand[np.random.randint(self.num_labels)] = 1
#        else:  # self.dtype is bool:
#            rand = [False] * self.num_labels
#            rand[np.random.randint(self.num_labels)] = True
#
#        return rand
#
#
#class BoolRange(RangeType):
#
#    def __init__(self, size=None):
#
#        super(BoolRange, self).__init__(size=size)
#
#    def get_random(self):
#
#        ret = []
#        for i in range(1 if self.size is None else self.size):
#            rand = np.random.randint(2) == 0
#            ret.append(rand)
#
#        if self.size is None:
#            ret = ret[0]
#
#        return ret
#
#
#class CartesianProduct(object):
#
#    def __init__(self, **kwargs):
#        self.vars = kwargs
#        self.constraints = {}
#
#    def get_random(self):
#
#        result = dict()
#        for var in self.vars:
#            if var in self.constraints:
#                rel, constr = self.constraints[var]
#                val = result[rel]
#            else:
#                val = None
#
#                def constr(*args):
#                    return True
#
#            while True:
#                res = self.vars[var].get_random()
#
#                if constr(res, val):
#                    break
#
#            result[var] = res
#
#        return result
#
#    def add_constraints(self, contraints):
#        for key in contraints:
#            self.constraints[key[0]] = [key[1], contraints[key]]
