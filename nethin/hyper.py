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
import numbers

import numpy as np


def _import_skopt(message="This operation requires 'scikit-optimize'."):
    try:
        import skopt
    except ModuleNotFoundError:
        raise RuntimeError(message)

    return skopt


__all__ = ["UniformPrior", "LogUniformPrior",
           "Boolean", "Real", "Integer", "Categorical", "Constant",
           "Space",
           "GaussianProcessRegression",
           "HyperParameterOptimization"]


class BasePrior(abc.ABC):

    @abc.abstractmethod
    def rvs(self, size=None, random_state=None):
        pass

    @abc.abstractmethod
    def transform(self, x):
        pass

    @abc.abstractmethod
    def inverse_transform(self, x):
        pass


class DiscretePrior(BasePrior):

    def __init__(self, probabilities):
        if not isinstance(probabilities, (list, tuple)):
            raise ValueError("The probabilities should be a list/tuple of "
                             "positive floats.")
        probabilities = [max(0.0, float(p)) for p in probabilities]
        sum_probs = sum(probabilities)
        self.probabilities = [p / sum_probs for p in probabilities]

        import scipy.stats

        self._rv_discrete = scipy.stats.rv_discrete(
                values=(range(len(self.probabilities)), self.probabilities))

    def rvs(self, size=None, random_state=None):

        if size is None:
            size = 1
        elif isinstance(size, (int, float)):
            size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            size = tuple([max(0, int(s + 0.5)) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

        import nethin.utils

        random_state = nethin.utils.normalize_random_state(random_state)

        samples = self._rv_discrete.rvs(size=size,
                                        random_state=random_state)

        return samples

    def transform(self, X):
        """Transform the prior's values to the default range [0, 1].
        """
        return np.clip((X / (len(self.probabilities) - 1)).astype(np.float),
                       0.0,
                       1.0)

    def inverse_transform(self, Y):
        """Transform values from [0, 1] back to the prior's range.
        """
        # TODO: "Only" works with less than 2**31 categories ...
        return (np.clip(Y * (len(self.probabilities) - 1),
                        0,
                        len(self.probabilities) - 1) + 0.5).astype(np.int)


class UniformPrior(BasePrior):

    # TODO: Handle closed, open, and half-open intervals.

    def __init__(self, low, high):
        self.low = min(float(low), float(high))
        self.high = max(float(low), float(high))

    def rvs(self, size=None, random_state=None):

        if size is None:
            size = 1
        elif isinstance(size, (int, float)):
            size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            size = tuple([max(0, int(s + 0.5)) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

        import nethin.utils
        import scipy.stats

        random_state = nethin.utils.normalize_random_state(random_state)

        samples = scipy.stats.uniform.rvs(loc=0.0,
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
        import nethin.consts

        self.low = max(nethin.consts.MACHINE_EPSILON,
                       min(float(low), float(high)))
        self.high = max(float(low), float(high))

        self._log_low = np.log10(self.low)
        self._log_high = np.log10(self.high)

    def rvs(self, size=None, random_state=None):

        if size is None:
            size = 1
        elif isinstance(size, (int, float)):
            size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            size = tuple([max(0, int(s + 0.5)) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

        import nethin.utils
        import scipy.stats

        random_state = nethin.utils.normalize_random_state(random_state)

        samples = scipy.stats.uniform.rvs(loc=0.0,
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

    Parameters
    ----------
    name : str
        The name of this dimension.

    prior : BasePrior
        The prior to use when sampling from this dimension.

    size : int or tuple of int, optional
        Non-negative integer(s). The shape of the output.
    """
    def __init__(self,
                 name,
                 prior,
                 size=None):

        self.name = str(name)

        if not isinstance(prior, BasePrior):
            raise ValueError("The 'prior' must be of type 'BasePrior'.")
        self.prior = prior

        if size is None:
            self.size = (1,)
        elif isinstance(size, (int, float)):
            self.size = (max(0, int(size + 0.5)),)
        elif isinstance(size, (list, tuple)):
            self.size = tuple([int(s + 0.5) for s in size])
        else:
            raise ValueError("Argument 'size' must be an int or a list/tuple "
                             "of int.")

    def rvs(self, size=None, random_state=None):

        if size is None:
            size = self.size

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


class Boolean(Dimension):
    """Represents a boolean variable.
    """
    def __init__(self,
                 name,
                 prior=None,
                 size=None):

        if prior is None:
            prior = UniformPrior(0.0, 1.0)

        if isinstance(prior, DiscretePrior) and len(prior.probabilities) != 2:
            raise ValueError("A DiscretePrior can only have two categories "
                             "here.")

        super(Boolean, self).__init__(name,
                                      prior,
                                      size=size)

    def rvs(self, size=None, random_state=None):

        value = super().rvs(size=size, random_state=random_state)

        if isinstance(self.prior, DiscretePrior):
            ret = value == 0  # There should only be two categories here!
        elif isinstance(self.prior, (UniformPrior, LogUniformPrior)):
            ret = value < ((self.prior.high - self.prior.low) / 2.0)
        else:
            raise RuntimeError("Unknown prior distribution.")

        return ret

    def _to_skopt(self):

        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if self.size is not None:
            size = np.prod(self.size)
        else:
            size = 1

        dims = []
        for i in range(size):
            if isinstance(self.prior, DiscretePrior):
                dims.append(skopt.space.Categorical(
                        [0, 1],
                        prior=self.prior.probabilities,
                        transform="identity",
                        name=self.name))

            elif isinstance(self.prior, UniformPrior):
                dims.append(skopt.space.Categorical(
                        [0, 1],
                        prior=None,  # None == uniform here
                        transform="identity",
                        name=self.name))
            else:
                raise RuntimeError("This operation uses 'scikit-optimize', "
                                   "and booleans do not work with the given "
                                   "prior (%s)." % (str(self.prior),))
        return dims


class Real(Dimension):

    def __init__(self,
                 name,
                 prior,
                 size=None):

        super(Real, self).__init__(name,
                                   prior,
                                   size=size)

    def rvs(self, size=None, random_state=None):

        value = super().rvs(size=size, random_state=random_state)

        return value.astype(np.float)

    def _to_skopt(self):
        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if self.size is not None:
            size = np.prod(self.size)
        else:
            size = 1

        dims = []
        for i in range(size):
            if isinstance(self.prior, UniformPrior):
                low = self.prior.low
                high = self.prior.high
                dims.append(skopt.space.Real(low,
                                             high,
                                             prior="uniform",
                                             transform="normalize",
                                             name=self.name))

            elif isinstance(self.prior, LogUniformPrior):
                low = self.prior.low
                high = self.prior.high
                dims.append(skopt.space.Real(low,
                                             high,
                                             prior="log-uniform",
                                             transform="normalize",
                                             name=self.name))
            else:
                raise RuntimeError("This operation uses 'scikit-optimize', "
                                   "and reals do not work with the given "
                                   "prior (%s)." % (str(self.prior),))
        return dims


class Integer(Dimension):

    def __init__(self,
                 name,
                 prior,
                 size=None):

        super(Integer, self).__init__(name,
                                      prior,
                                      size=size)

    def rvs(self, size=None, random_state=None):

        value = super().rvs(size=size, random_state=random_state)

        if isinstance(self.prior, DiscretePrior):
            return value
        else:
            return (value + 0.5).astype(np.int)

    def _to_skopt(self):
        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if self.size is not None:
            size = np.prod(self.size)
        else:
            size = 1

        dims = []
        for i in range(size):
            if isinstance(self.prior, UniformPrior):
                low = self.prior.low
                high = self.prior.high
                dims.append(skopt.space.Integer(int(low + 0.5),
                                                int(high + 0.5),
                                                transform="normalize",
                                                name=self.name))

            # elif isinstance(self.prior, LogUniformPrior):
            #     low = self.prior.low
            #     high = self.prior.high
            #     dims.append(skopt.space.Real(low,
            #                                  high,
            #                                  prior="log-uniform",
            #                                  transform="normalize",
            #                                  name=self.name))
            else:
                raise RuntimeError("This operation uses 'scikit-optimize', "
                                   "and integers do not work with the given "
                                   "prior (%s)." % (str(self.prior),))
        return dims


class Categorical(Dimension):
    """Represents a categorical variable.
    """
    def __init__(self,
                 name,
                 categories,
                 prior=None,
                 size=None):

        if prior is None:
            prior = UniformPrior(0.0, 1.0)

        super(Categorical, self).__init__(name,
                                          prior,
                                          size=size)

        if not isinstance(categories, (list, tuple)):
            raise ValueError("The categories should be a list or tuple.")
        self.categories = categories

    def rvs(self, size=None, random_state=None):

        value = super().rvs(size=size, random_state=random_state)
        shape = value.shape
        ret = [None] * value.size

        value = value.reshape(-1)
        num_categories = len(self.categories)
        for i in range(value.size):
            index = int(value[i] * num_categories)
            val = self.categories[index]
            ret[i] = val
        ret = np.reshape(ret, shape)

        return ret

    def _to_skopt(self):
        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if self.size is not None:
            size = np.prod(self.size)
        else:
            size = 1

        dims = []
        for i in range(size):
            if isinstance(self.prior, DiscretePrior):
                dims.append(skopt.space.Categorical(
                        self.categories,
                        prior=self.prior.probabilities,
                        transform="onehot",
                        name=self.name))

            elif isinstance(self.prior, UniformPrior):
                dims.append(skopt.space.Categorical(
                        self.categories,
                        prior=None,  # None == uniform here
                        transform="onehot",
                        name=self.name))
            else:
                raise RuntimeError("This operation uses 'scikit-optimize', "
                                   "and categorials do not work with the "
                                   "given prior (%s)." % (str(self.prior),))
        return dims


class Constant(Dimension):
    """Represents a constant value.
    """
    def __init__(self,
                 name,
                 value,
                 prior=None,
                 size=None,
                 dtype=None,
                 **kwargs):

        if prior is None:
            prior = UniformPrior(0.0, 0.0)

        super(Constant, self).__init__(name,
                                       prior,
                                       size=size)

        if not isinstance(value, (bool, int, float)):
            raise ValueError("The value should be an int or a float.")
        self.dtype = dtype
        if self.dtype is None:
            self.value = value
        else:
            self.value = self.dtype(value)

    def rvs(self, size=None, random_state=None):

        if isinstance(self.value, bool):
            ret = np.ones(self.size, dtype=bool) * self.value

        elif isinstance(self.value, numbers.Integral):
            ret = np.ones(self.size, dtype=int) * self.value

        elif isinstance(self.value, numbers.Real):
            ret = np.ones(self.size, dtype=float) * self.value

        else:
            ret = np.ones(self.size, dtype=type(self.value)) * self.value

        return ret

    def _to_skopt(self):
        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if self.size is not None:
            size = np.prod(self.size)
        else:
            size = 1

        dims = []
        for i in range(size):
            dims.append(skopt.space.Categorical([self.value],
                                                transform="identity",
                                                name=self.name))

        return dims


class Space(object):

    def __init__(self, dimensions=[], unflatten_dict=False):

        if not isinstance(dimensions, (list, tuple)):
            raise ValueError("The 'dimensions' should be a list/tuple with "
                             "elements of type 'Dimension'.")
        self.dimensions = {}
        for dim in dimensions:
            self.add(dim)

        self.unflatten_dict = bool(unflatten_dict)

    def rvs(self, size=None, random_state=None):

        import nethin.utils

        random_state = nethin.utils.normalize_random_state(random_state)

        dims = {}
        for dim_name in self.dimensions.keys():
            dims[dim_name] = self.dimensions[dim_name].rvs(
                    size=size, random_state=random_state)

        return dims

    def is_categorical(self):

        return all([isinstance(dim, Categorical)
                    for dim in self.dimensions.values()])

    def num_dim_normalized(self):

        return sum([1 if dim.size is None else np.prod(dim.size)
                    for dim in self.dimensions.values()])

    def _to_skopt(self):

        _import_skopt("This operation requires 'scikit-optimize'.")

        dims = []
        for dim in self.dimensions.values():
            dim_ = dim._to_skopt()
            if isinstance(dim_, list):
                dims.extend(dim_)
            else:
                dims.append(dim_)

        return dims

    def add(self, dim):

        if not isinstance(dim, Dimension):
            raise ValueError("The 'dim' should be of type 'Dimension'.")
        else:
            if dim.name in self.dimensions:
                raise ValueError("Two dimensions can not have the same name "
                                 "(%s)." % (dim.name,))

            self.dimensions[dim.name] = dim

    def unflatten(self, dims):
        """
        Examples
        --------
        >>> import nethin.hyper as hyper
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> #
        >>> a = hyper.Real("a", hyper.UniformPrior(-2, 2), size=(3, 2))
        >>> b = hyper.Integer("b", hyper.UniformPrior(-2, 2))
        >>> c = hyper.Boolean("c", hyper.UniformPrior(-2, 2), size=2)
        >>> space = hyper.Space([a, b, c])
        >>> #
        >>> minimizer = hyper.GaussianProcessRegression(
        ...     space,
        ...     acquisition_function=hyper.ExpectedImprovement(hyper.LBFGS()),
        ...     result=None,  # Previous results
        ...     random_state=np.random.randint(2**31))
        >>> hyperopt = hyper.HyperParameterOptimization(minimizer)
        >>> #
        >>> result = hyperopt.step(lambda x: 0.0)
        >>> vals = np.round(result["x"], 5)  # doctest: +NORMALIZE_WHITESPACE
        array([-1.40655,  0.54935, -1.65037, -0.99839,  0.78373,  0.04561,
                2.     ,  1.     ,  0.     ])
        >>> unflattened = space.unflatten(vals)
        >>> unflattened  # doctest: +NORMALIZE_WHITESPACE
        [array([[-1.40655,  0.54935],
                [-1.65037, -0.99839],
                [ 0.78373,  0.04561]]), array([2.]), array([1., 0.])]
        >>> unflattened[0].shape
        (3, 2)
        >>> unflattened[1].shape
        (1,)
        >>> unflattened[2].shape
        (2,)
        """
        if self.unflatten_dict:
            unflat_dims = {}
        else:
            unflat_dims = []
        i = 0
        for dim in self.dimensions.values():
            if dim.size is None:
                numel = 1
                shape = (1,)
            else:
                numel = int(np.prod(dim.size) + 0.5)
                if isinstance(dim.size, int):
                    shape = (dim.size,)
                else:
                    shape = tuple(dim.size)

            vals = dims[i:i + numel]
            i += numel
            vals = np.asarray(vals)
            vals = vals.reshape(*shape)

            if self.unflatten_dict:
                unflat_dims[dim.name] = vals
            else:
                unflat_dims.append(vals)

        return unflat_dims


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
    """The acquisition function is minimised using the limited memory BFGS
    algorithm.
    """
    def _to_skopt(self):
        return "lbfgs"


class Sampling(BaseAcquisitionOptimizer):
    """The minimum is found as the smallest value found when evaluating the
    acquisition function at randomly sampled points in the search space.
    """
    def _to_skopt(self):
        return "sampling"


class BaseMinimizer(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, f, **kwargs):
        pass


class GaussianProcessRegression(BaseMinimizer):

    def __init__(self,
                 space,
                 noise=None,
                 acquisition_function=None,
                 result=None,
                 random_state=None):

        skopt = _import_skopt("This operation requires 'scikit-optimize'.")

        if not isinstance(space, Space):
            raise ValueError("The 'space' must be of type 'Space'.")
        self.dimensions = skopt.utils.normalize_dimensions(space._to_skopt())
        self._space = space

        if noise is None:
            self.noise = None
        else:
            import nethin.consts

            self.noise = max(np.sqrt(nethin.consts.MACHINE_EPSILON),
                             float(noise))

        if not isinstance(acquisition_function, BaseAcquisitionFunction):
            raise ValueError("The 'acquisition_function' must be of type "
                             "'BaseAcquisitionFunction'.")
        if space.is_categorical() \
                and (not isinstance(acquisition_function.optimizer, Sampling)):
            raise ValueError("The space contains only categorical variables, "
                             "use the Sampling optimizer for the acquisition "
                             "function.")
        self.acquisition_function = acquisition_function._to_skopt()

        self.acquisition_optimizer = \
            acquisition_function.optimizer._to_skopt()

        import scipy.optimize

        result_done = False
        if result is None:
            self.result = None
            result_done = True
        elif isinstance(result, scipy.optimize.optimize.OptimizeResult):
            self.result = result
            result_done = True
        elif isinstance(result, (tuple, list)) \
                and len(result) == 2 \
                and isinstance(result[0], (np.ndarray, list)) \
                and isinstance(result[1], (np.ndarray, list)):

            result = [result[0], result[1]]

            ok_0 = True
            if isinstance(result[0], np.ndarray):
                if len(result[0].shape) == 2:
                    result[0] = result[0].tolist()
                else:
                    ok_0 = False
            elif isinstance(result[0], list):
                for val in result[0]:
                    if not isinstance(val, list):
                        ok_0 = False
                        break

            ok_1 = True
            if isinstance(result[1], np.ndarray):
                if result[1].ndim == 2:
                    if result[1].shape[0] == 1 or result[1].shape[1] == 1:
                        result[1] = result[1].ravel()
                    else:
                        ok_1 = False
                elif result[1].ndim != 1:
                    ok_1 = False
            elif isinstance(result[1], list):
                for val in result[1]:
                    if isinstance(val, (list, tuple)):
                        ok_1 = False
                        break

            if ok_0 and ok_1:
                self.result = skopt.utils.create_result(result[0], result[1])
                result_done = True

        if not result_done:
            raise ValueError("The 'result' must either be an 'OptimizeResult' "
                             "or a tuple of two numpy arrays or lists of "
                             "lists.")

        import nethin.utils

        self.random_state = nethin.utils.normalize_random_state(random_state)

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

    def _cook_estimator(self, space, **kwargs):
        """
        Cook a default estimator.

        For the special base_estimator called "DUMMY" the return value is None.
        This corresponds to sampling points at random, hence there is no need
        for an estimator.

        Parameters
        ----------
        space : Space
            The space to sample over.

        kwargs : dict, optional
            Extra parameters provided to the base estimator at init time.
        """
        from skopt.learning import GaussianProcessRegressor

        from skopt.learning.gaussian_process.kernels import ConstantKernel
        from skopt.learning.gaussian_process.kernels import HammingKernel
        from skopt.learning.gaussian_process.kernels import Matern

        if isinstance(space, Space):
            n_dims = space.num_dim_normalized()
            is_cat = space.is_categorical()
        else:
            n_dims = space.transformed_n_dims
            is_cat = space.is_categorical

        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        # Only use a special kernel if all dimensions are categorical
        if is_cat:
            other_kernel = HammingKernel(length_scale=np.ones(n_dims))
        else:
            other_kernel = Matern(
                length_scale=np.ones(n_dims),
                length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)

        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, noise="gaussian",
            n_restarts_optimizer=2)

        base_estimator.set_params(**kwargs)

        return base_estimator

    def step(self, f, **kwargs):

        specs = {"args": copy.copy(inspect.currentframe().f_locals),
                 "function": inspect.currentframe().f_code.co_name}

        import skopt.utils

        if self.base_estimator_ is None:
            self.base_estimator_ = self._cook_estimator(
                self.dimensions,
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
                    result = self.optimizer_.tell(X0[i], y0[i])
                    result.specs = specs

        next_x = self.optimizer_.ask()
        next_y = f(self._space.unflatten(next_x), **kwargs)
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
    >>> #
    >>> noise_level = 0.1
    >>> def f(x, noise_level=noise_level):
    ...     return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) \
    ...                   + np.random.randn() * noise_level
    >>> x = hyper.Real("x", hyper.UniformPrior(-2, 2), size=2)
    >>> space = hyper.Space([x])
    >>> af_minimizer = hyper.LBFGS()
    >>> acquisition_function = hyper.ExpectedImprovement(af_minimizer)
    >>> minimizer = hyper.GaussianProcessRegression(
    ...         space,
    ...         acquisition_function=acquisition_function,
    ...         result=None,  # Previous results
    ...         random_state=np.random.randint(2**31))
    >>> hyperopt = hyper.HyperParameterOptimization(minimizer)
    >>> #
    >>> result = hyperopt.step(f)
    >>> np.round(result["x_iters"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([[-1.40655,  0.54935]])
    >>> np.round(result["func_vals"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([-0.08059])
    >>>
    >>> result = hyperopt.step(f)
    >>> np.round(result["x_iters"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([[-1.40655,  0.54935],
           [ 2.     , -2.     ]])
    >>> np.round(result["func_vals"], 5)  # doctest: +NORMALIZE_WHITESPACE
    array([-0.08059,  0.05118])
    """
    def __init__(self, minimizer, callback=None):
        if not isinstance(minimizer, BaseMinimizer):
            raise ValueError("The 'minimizer' must be of type "
                             "'BaseMinimizer'.")
        self.minimizer = minimizer

        import nethin.utils

        self.callbacks = nethin.utils.normalize_callables(callback)

    def step(self, f, **kwargs):

        if not callable(f):
            raise ValueError("The 'f' must be callable.")

        result = self.minimizer.step(f, **kwargs)

        return result

    def minimize(self, f, num_iter=10):

        if not callable(f):
            raise ValueError("The 'f' must be callable.")

        num_iter = max(1, int(num_iter))

        import nethin.utils

        for i in range(num_iter):

            result = self.step(f)

            if nethin.utils.apply_callables(self.callbacks, result):
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
