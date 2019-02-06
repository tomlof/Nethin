# -*- coding: utf-8 -*-
"""
Contains means to read, generate and handle data.

Created on Tue Oct  3 08:20:52 2017

Copyright (c) 2017-2018, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import os
import re
import abc
import six
import datetime
import bisect
import warnings
import sqlite3
import struct
import queue
import traceback
import collections

import numpy as np
from six import with_metaclass
from scipy.misc import imread, imresize

try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend.common import normalize_data_format

import nethin
import nethin.utils as utils

try:
    from collections import Generator
    _HAS_GENERATOR = True
except (ImportError):
    _HAS_GENERATOR = False

try:
    import pydicom
    _HAS_PYDICOM = True
except (ImportError):
    try:
        import dicom as pydicom
        _HAS_PYDICOM = True
    except (ImportError):
        _HAS_PYDICOM = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except (ImportError):
    _HAS_PANDAS = False


__all__ = ["BaseGenerator",
           "ImageGenerator", "ArrayGenerator",
           "Dicom3DGenerator", "DicomGenerator",
           "Numpy2DGenerator", "Numpy3DGenerator",
           "Dicom3DSaver"]


try:
    from torch.utils.data import Dataset
    from torch.utils.data import ConcatDataset
    from torch.utils.data import DataLoader
    _HAS_PYTORCH = True

except (ImportError):  # Copy of PyTorch code
    _HAS_PYTORCH = False

    class Dataset(object):
        r"""This abstract class is a copy of torch.utils.data.Dataset, used as
        the base class for all datasets. PyTorch pydoc follows:

        An abstract class representing a Dataset.

        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and
        ``__getitem__``, supporting integer indexing in range from 0 to
        len(self) exclusive.
        """
        def __getitem__(self, index):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError

        def __add__(self, other):
            return ConcatDataset([self, other])

    class ConcatDataset(Dataset):
        r"""This class is a copy of torch.utils.data.ConcatDataset. PyTorch
        pydoc follows:

        Dataset to concatenate multiple datasets.

        Purpose: useful to assemble different existing datasets, possibly
        large-scale datasets as the concatenation operation is done in an
        on-the-fly manner.

        Arguments:
            datasets (sequence): List of datasets to be concatenated
        """
        @staticmethod
        def cumsum(sequence):
            r, s = [], 0
            for e in sequence:
                l_ = len(e)
                r.append(l_ + s)
                s += l_
            return r

        def __init__(self, datasets):
            super(ConcatDataset, self).__init__()
            assert len(datasets) > 0, "datasets should not be an empty " + \
                "iterable"
            self.datasets = list(datasets)
            self.cumulative_sizes = self.cumsum(self.datasets)

        def __len__(self):
            return self.cumulative_sizes[-1]

        def __getitem__(self, idx):
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]

        @property
        def cummulative_sizes(self):
            warnings.warn("cummulative_sizes attribute is renamed to "
                          "cumulative_sizes", DeprecationWarning, stacklevel=2)

            return self.cumulative_sizes

    class _DataLoaderIter(object):
        r"""This class is a copy of
        torch.utils.data.dataloader._DataLoaderIter. PyTorch pydoc follows:

        Iterates once over the DataLoader's dataset, as specified by the
        sampler.
        """
        def __init__(self, loader):
            self.dataset = loader.dataset
            self.collate_fn = loader.collate_fn
            self.batch_sampler = loader.batch_sampler

            # Nethin addition: We don't allow multiprocess loading of images
            if loader.num_workers > 0:
                warnings.warm("PyTorch is not available. Multithreaded "
                              "loading is not available.")
            self.num_workers = 0  # loader.num_workers

            # Nethin addition: We don't allow to pin on GPU
            if loader.pin_memory:
                warnings.warm("PyTorch is not available. Tensors can not be "
                              "pinned on the GPU.")
            self.pin_memory = False
            # self.pin_memory = loader.pin_memory and torch.cuda.is_available()

            self.timeout = loader.timeout

            self.sample_iter = iter(self.batch_sampler)

            # Nethin addition! Everything below here was removed:

            # base_seed = torch.LongTensor(1).random_().item()

            # if self.num_workers > 0:
            #     ...

        def __len__(self):
            return len(self.batch_sampler)

        def _get_batch(self):
            # In the non-timeout case, worker exit is covered by SIGCHLD
            # handler. But if `pin_memory=True`, we still need account for the
            # possibility that `pin_memory_thread` dies.
            if self.timeout > 0:
                try:
                    return self.data_queue.get(timeout=self.timeout)
                except queue.Empty:
                    raise RuntimeError("DataLoader timed out after {} "
                                       "seconds".format(self.timeout))
            # Nethin addition: self.pin_memory is False!
            # elif self.pin_memory:
            #     while self.pin_memory_thread.is_alive():
            #         try:
            #             return self.data_queue.get(
            #                 timeout=MP_STATUS_CHECK_INTERVAL)
            #         except queue.Empty:
            #             continue
            #     else:
            #         # while condition is false, i.e., pin_memory_thread died.
            #         raise RuntimeError('Pin memory thread exited '
            #                            'unexpectedly')
            #     # In this case, `self.data_queue` is a `queue.Queue`,. But we
            #     # don't need to call `.task_done()` because we don't use
            #     # `.join()`.
            else:
                return self.data_queue.get()

        def __next__(self):
            if self.num_workers == 0:  # same-process loading
                indices = next(self.sample_iter)  # may raise StopIteration
                batch = self.collate_fn([self.dataset[i] for i in indices])
                # Nethin addition: self.pin_memory is False!
                # if self.pin_memory:
                #     batch = pin_memory_batch(batch)
                return batch

            # Nethin addition: self.num_workers is always 0, so the below code
            # was removed.
            # # check if the next sample has already been generated
            # if self.rcvd_idx in self.reorder_dict:
            #     batch = self.reorder_dict.pop(self.rcvd_idx)
            #     return self._process_next_batch(batch)
            #
            # if self.batches_outstanding == 0:
            #     self._shutdown_workers()
            #     raise StopIteration
            #
            # while True:
            #     assert (not self.shutdown and self.batches_outstanding > 0)
            #     idx, batch = self._get_batch()
            #     self.batches_outstanding -= 1
            #     if idx != self.rcvd_idx:
            #         # store out-of-order samples
            #         self.reorder_dict[idx] = batch
            #         continue
            #     return self._process_next_batch(batch)

        # Nethin addition: We do not support Python 2.
        # next = __next__  # Python 2 compatibility

        def __iter__(self):
            return self

        # Nethin addition: self.num_workers is always 0, so the below two
        # methods were removed.
        # def _put_indices(self):
        #     assert self.batches_outstanding < 2 * self.num_workers
        #     indices = next(self.sample_iter, None)
        #     if indices is None:
        #         return
        #     self.index_queues[self.worker_queue_idx].put((self.send_idx,
        #                                                   indices))
        #     self.worker_queue_idx \
        #         = (self.worker_queue_idx + 1) % self.num_workers
        #     self.batches_outstanding += 1
        #     self.send_idx += 1
        #
        # def _process_next_batch(self, batch):
        #     self.rcvd_idx += 1
        #     self._put_indices()
        #     if isinstance(batch, ExceptionWrapper):
        #         raise batch.exc_type(batch.exc_msg)
        #     return batch

        def __getstate__(self):
            # TODO: add limited pickling support for sharing an iterator
            # across multiple threads for HOGWILD.
            # Probably the best way to do this is by moving the sample pushing
            # to a separate thread and then just sharing the data queue
            # but signalling the end is tricky without a non-blocking API
            raise NotImplementedError("_DataLoaderIter cannot be pickled")

        # Nethin addition: self.num_workers is always 0, so the below code
        # will never be run, and was therefore removed.
        # def _shutdown_workers(self):
        #     # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for
        #     # details on the logic of this function.
        #     if not self.shutdown:
        #         self.shutdown = True
        #         # Removes pids from the C side data structure first so worker
        #         # termination afterwards won't trigger false positive error
        #         # report.
        #         if self.worker_pids_set:
        #             _remove_worker_pids(id(self))
        #             self.worker_pids_set = False
        #
        #         self.done_event.set()
        #
        #         # Exit `pin_memory_thread` first because exiting workers may
        #         # leave corrupted data in `worker_result_queue` which
        #         # `pin_memory_thread` reads from.
        #         if hasattr(self, 'pin_memory_thread'):
        #             # Use hasattr in case error happens before we set the
        #             # attribute. First time do `worker_result_queue.put` in
        #             # this process.
        #
        #             # `cancel_join_thread` in case that `pin_memory_thread`
        #             # exited.
        #             self.worker_result_queue.cancel_join_thread()
        #             self.worker_result_queue.put(None)
        #             self.pin_memory_thread.join()
        #
        #             # Indicate that no more data will be put on this queue by
        #             # the current process. This **must** be called after
        #             # `pin_memory_thread` is joined because that thread
        #             # shares the same pipe handles with this loader thread.
        #             # If the handle is closed, Py3 will error in this case,
        #             # but Py2 will just time out even if there is data in the
        #             # queue.
        #             self.worker_result_queue.close()
        #
        #         # Exit workers now.
        #         for q in self.index_queues:
        #             q.put(None)
        #             # Indicate that no more data will be put on this queue by
        #             # the current process.
        #             q.close()
        #         for w in self.workers:
        #             w.join()

        def __del__(self):
            # Nethin addition: self.num_workers is always 0, so the below code
            # was removed.
            # if self.num_workers > 0:
            #     self._shutdown_workers()
            pass

    def default_collate(batch):
        r"""This function is an altered copy of
        torch.utils.data.dataloader.default_collate. PyTorch pydoc follows:

        Puts each data field into a tensor with outer dimension batch size.
        """
        # Nethin addition: No tensors
        error_msg = "batch must contain ndarrays, numbers, dicts or " + \
                    "lists; found {}"
        # error_msg = "batch must contain tensors, numbers, dicts or " + \
        #             "lists; found {}"
        elem_type = type(batch[0])
        # Nethin addition: No batches will be torch tensors, so removed.
        # if isinstance(batch[0], torch.Tensor):
        #     out = None
        #     if _use_shared_memory:
        #         # If we're in a background process, concatenate directly into
        #         # a shared memory tensor to avoid an extra copy
        #         numel = sum([x.numel() for x in batch])
        #         storage = batch[0].storage()._new_shared(numel)
        #         out = batch[0].new(storage)
        #     return torch.stack(batch, 0, out=out)
        if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                # Nethin addition: Stack numpy arrays instead of Pytorch
                # tensors
                return np.concatenate([b for b in batch], axis=0)
                # return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                # Nethin addition: Make numpy arrays instead of Pytorch tensors
                return np.array(list(map(py_type, batch)))
                # return numpy_type_map[elem.dtype.name](list(map(py_type,
                #                                                 batch)))
        # Nethin addition: Use six instead of torch._six:
        elif isinstance(batch[0], float):
            return np.array(batch)
        elif isinstance(batch[0], six.integer_types):
            return np.array(batch)
        # elif isinstance(batch[0], int_classes):
        #     return torch.LongTensor(batch)
        # elif isinstance(batch[0], float):
        #     return torch.DoubleTensor(batch)
        elif isinstance(batch[0], six.string_types):
            # elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], collections.abc.Mapping):
            # print(batch)
            # elif isinstance(batch[0], container_abcs.Mapping):
            return {key: default_collate([d[key] for d in batch])
                    for key in batch[0]}
        elif isinstance(batch[0], collections.abc.Sequence):
            # elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    class Sampler(object):
        r"""This class is a copy of torch.utils.data.sampler.Sampler. PyTorch
        pydoc follows:

        Base class for all Samplers.

        Every Sampler subclass has to provide an __iter__ method, providing a
        way to iterate over indices of dataset elements, and a __len__ method
        that returns the length of the returned iterators.
        """
        def __init__(self, data_source):
            pass

        def __iter__(self):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError

    class SequentialSampler(Sampler):
        r"""This class is a copy of torch.utils.data.sampler.SequentialSampler.
        PyTorch pydoc follows:

        Samples elements sequentially, always in the same order.

        Arguments:
            data_source (Dataset): dataset to sample from
        """
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            return iter(range(len(self.data_source)))

        def __len__(self):
            return len(self.data_source)

    class RandomSampler(Sampler):
        r"""This class is an altered copy of
        torch.utils.data.sampler.RandomSampler. PyTorch pydoc follows:

        Samples elements randomly. If without replacement, then sample from
        a shuffled dataset. If with replacement, then user can specify
        ``num_samples`` to draw.

        Parameters
        ----------
        data_source : Dataset
            The dataset to sample from.

        num_samples : int, optional
            The number of samples to draw, default is ``len(dataset)``.

        replacement : bool, optional
            Samples are drawn with replacement if ``True``, default is
            ``False``.
        """
        def __init__(self, data_source, replacement=False, num_samples=None):
            self.data_source = data_source
            self.num_samples = num_samples
            self.replacement = replacement

            if self.num_samples is not None and replacement is False:
                raise ValueError("With replacement=False, num_samples should "
                                 "not be specified, since a random permute "
                                 "will be performed.")

            if self.num_samples is None:
                self.num_samples = len(self.data_source)

            if not isinstance(self.num_samples, int) or self.num_samples <= 0:
                raise ValueError("num_samples should be a positive integeral "
                                 "value, but got num_samples={}".format(
                                         self.num_samples))
            if not isinstance(self.replacement, bool):
                raise ValueError("replacement should be a boolean value, but "
                                 "got replacement={}".format(self.replacement))

        def __iter__(self):
            n = len(self.data_source)
            if self.replacement:
                # Nethin addition: Use numpy instead of Pytorch
                return iter(np.random.randint(low=0,
                                              high=n,
                                              size=(self.num_samples,),
                                              dtype=np.int64).tolist())
                # return iter(torch.randint(high=n, size=(self.num_samples,),
                #                           dtype=torch.int64).tolist())

            # Nethin addition: Use numpy instead of Pytorch
            return iter(np.random.permutation(n).tolist())
            # return iter(torch.randperm(n).tolist())

        def __len__(self):
            return len(self.data_source)

    class BatchSampler(Sampler):
        r"""This class is an altered copy of
        torch.utils.data.sampler.BatchSampler. PyTorch pydoc follows:

        Wraps another sampler to yield a mini-batch of indices.

        Parameters
        ----------
        sampler : Sampler
            Base sampler.

        batch_size : int
            Size of mini-batch.

        drop_last : bool
            If ``True``, the sampler will drop the last batch if its size would
            be less than ``batch_size``.

        Examples
        --------
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3,
        ...      drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3,
        ...      drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
        def __init__(self, sampler, batch_size, drop_last):
            if not isinstance(sampler, Sampler):
                raise ValueError("sampler should be an instance of "
                                 "data.Sampler, but got sampler={}"
                                 .format(sampler))
            # Nethin addition: Use six instead of torch._six
            # if not isinstance(batch_size, _int_classes) \
            #     or isinstance(batch_size, bool) or batch_size <= 0
            if not isinstance(batch_size, six.integer_types) \
                    or isinstance(batch_size, bool) or batch_size <= 0:
                raise ValueError("batch_size should be a positive integeral "
                                 "value, but got batch_size={}".format(
                                         batch_size))
            if not isinstance(drop_last, bool):
                raise ValueError("drop_last should be a boolean value, but "
                                 "got drop_last={}".format(drop_last))
            self.sampler = sampler
            self.batch_size = batch_size
            self.drop_last = drop_last

        def __iter__(self):
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

        def __len__(self):
            if self.drop_last:
                return len(self.sampler) // self.batch_size
            else:
                return (len(self.sampler) + self.batch_size - 1) \
                    // self.batch_size

    class ExceptionWrapper(object):
        r"""This class is a copy of
        torch.utils.data.dataloader.ExceptionWrapper. PyTorch pydoc follows:

        Wraps an exception plus traceback to communicate across threads.
        """
        def __init__(self, exc_info):
            # It is important that we don't store exc_info, see
            # NOTE [ Python Traceback Reference Cycle Problem ]
            self.exc_type = exc_info[0]
            self.exc_msg = "".join(traceback.format_exception(*exc_info))

    class DataLoader(object):
        r"""This class is an altered copy of
        torch.utils.data.dataloader.DataLoader. PyTorch pydoc follows:

        Data loader. Combines a dataset and a sampler, and provides
        single- or multi-process iterators over the dataset.

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: ``False``).
            sampler (Sampler, optional): defines the strategy to draw samples
                from the dataset. If specified, ``shuffle`` must be False.
            batch_sampler (Sampler, optional): like sampler, but returns a
                batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and
                :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main
                process. (default: ``0``)
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch.
            pin_memory (bool, optional): If ``True``, the data loader will copy
                tensors into CUDA pinned memory before returning them.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch, if the dataset size is not divisible by the
                batch size. If ``False`` and the size of dataset is not
                divisible by the batch size, then the last batch will be
                smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch from workers. Should always be non-negative.
                (default: ``0``)
            worker_init_fn (callable, optional): If not ``None``, this will be
                called on each worker subprocess with the worker id (an int in
                ``[0, num_workers - 1]``) as input, after seeding and before
                data loading. (default: ``None``)

        .. note:: By default, each worker will have its PyTorch seed set to
                  ``base_seed + worker_id``, where ``base_seed`` is a long
                  generated by main process using its RNG. However, seeds for
                  other libraies may be duplicated upon initializing workers
                  (w.g., NumPy), causing each worker to return identical random
                  numbers. (See :ref:`dataloader-workers-random-seed` section
                  in FAQ.) You may use :func:`torch.initial_seed()` to access
                  the PyTorch seed for each worker in :attr:`worker_init_fn`,
                  and use it to set other seeds before data loading.

        .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn`
                     cannot be an unpicklable object, e.g., a lambda function.
        """
        __initialized = False

        def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0,
                     collate_fn=default_collate, pin_memory=False,
                     drop_last=False, timeout=0, worker_init_fn=None):
            self.dataset = dataset
            self.batch_size = batch_size

            # Nethin addition: We don't allow multiprocess loading of images
            if int(num_workers) > 0:
                warnings.warn("PyTorch is not available. Multithreaded "
                              "loading is not available. num_workers will be "
                              "0.")
            self.num_workers = 0
            # self.num_workers = num_workers

            self.collate_fn = collate_fn

            # Nethin addition: We don't allow to pin on GPU
            if bool(pin_memory):
                warnings.warm("PyTorch is not available. Tensors can not be "
                              "pinned on the GPU.")
            self.pin_memory = False
            # self.pin_memory = pin_memory

            self.drop_last = drop_last
            self.timeout = timeout
            self.worker_init_fn = worker_init_fn

            if timeout < 0:
                raise ValueError('timeout option should be non-negative')

            if batch_sampler is not None:
                if batch_size > 1 or shuffle or sampler is not None \
                        or drop_last:
                    raise ValueError('batch_sampler option is mutually '
                                     'exclusive with batch_size, shuffle, '
                                     'sampler, and drop_last')
                self.batch_size = None
                self.drop_last = None

            if sampler is not None and shuffle:
                raise ValueError('sampler option is mutually exclusive with '
                                 'shuffle')

            if self.num_workers < 0:
                raise ValueError('num_workers option cannot be negative; '
                                 'use num_workers=0 to disable '
                                 'multiprocessing.')

            if batch_sampler is None:
                if sampler is None:
                    if shuffle:
                        sampler = RandomSampler(dataset)
                    else:
                        sampler = SequentialSampler(dataset)
                batch_sampler = BatchSampler(sampler, batch_size, drop_last)

            self.sampler = sampler
            self.batch_sampler = batch_sampler
            self.__initialized = True

        def __setattr__(self, attr, val):
            if self.__initialized and attr in ('batch_size', 'sampler',
                                               'drop_last'):
                raise ValueError('{} attribute should not be set after {} is '
                                 'initialized'.format(attr,
                                                      self.__class__.__name__))

            super(DataLoader, self).__setattr__(attr, val)

        def __iter__(self):
            return _DataLoaderIter(self)

        def __len__(self):
            return len(self.batch_sampler)


class SQLiteDataset(Dataset):
    """A dataset defined in an SQLite database. The table is assumed to be
    called "Data", and have the columns:

        ID INTEGER  -- Primary key of row
        SeriesID TEXT  -- ID of series, where all modalities have the same id
        Name TEXT  -- Name of tensor
        Description TEXT  -- Textual description
        Width INTEGER  -- Tensor width
        Height INTEGER  -- Tensor height
        Z INTEGER  -- Slice index
        T INTEGER  -- Time index
        C INTEGER  -- Channel index
        DataType TEXT  -- E.g. "float32"
        Type TEXT  -- "input" or "output"
        Class TEXT  -- "scalar" or "tensor"
        Data BLOB  -- The actual data in little endian byte order
    """
    def __init__(self,
                 sqlite_file,
                 form="2d",
                 transform=None,
                 transform_3d=None):
        """
        Parameters
        ----------
        sqlite_file : str
            Path to the SQLite database to read from.

        form : str, optional
            Whether to read one slice at the time (2d) or one volume image at
            the time (3d). Default is 2d, one slice at the time is read.

        transform : callable, optional
            Transform to apply to each slice. Will always be applied to each
            slice, regardless of the value of ``form``. See ``transform_3d``
            for more information.

        transform_3d : callable, optional
            Transform to apply to each volume image. Will only be applied if
            ``form="3d"``, and if so, it will be applied after ``transform``
            has been applied to each slice first.
        """
        if not _HAS_PANDAS:
            raise RuntimeError("SQLiteDataset requires the Pandas package to "
                               "work.")

        self.sqlite_file = str(sqlite_file)
        if not os.path.exists(self.sqlite_file):
            raise ValueError('Database file "%s" not found.'
                             % (self.sqlite_file,))

        form = str(form)
        if form not in ["2d", "3d"]:
            raise ValueError('Argument "form" must be one of "2d" or "3d".')
        self.form = form

        self.transform = transform
        self.transform_3d = transform_3d

        self._metadata = None
        self._indices = None

    def _get_column_names(self, remove=None):

        cols = ["ID", "SeriesID", "Name", "Description", "Width", "Height",
                "Z", "T", "C", "DataType", "Type", "Class", "Data"]

        if remove is not None:
            if not isinstance(remove, (list, tuple)):
                remove = [str(remove)]

            remove = [str(r).lower() for r in remove]

            cols_ = []
            for col in cols:
                if col.lower() not in remove:
                    cols_.append(col)
            cols = cols_

        return cols

    def _select_query(self, query, params=None):

        db_connection = None
        db_cursor = None
        try:
            db_connection = sqlite3.connect(self.sqlite_file,
                                            check_same_thread=False)
            db_cursor = db_connection.cursor()
            if params is None:
                db_cursor.execute(query)
            else:
                db_cursor.execute(query, params)
            data = db_cursor.fetchall()

            db_cursor.close()
            db_connection.close()

            return data

        except Exception as e:
            if db_cursor is not None:
                db_cursor.close()

            if db_connection is not None:
                db_connection.close()

            raise e

    def _read_metadata(self):

        db_connection = None
        try:
            db_connection = sqlite3.connect(self.sqlite_file,
                                            check_same_thread=False)
            cols = self._get_column_names(remove="Data")
            query = "SELECT " + ", ".join(cols) + " FROM Data;"
            dt = pd.read_sql_query(query, db_connection)
            db_connection.close()

            self._metadata = dt

        except Exception as e:
            if db_connection is not None:
                db_connection.close()

            raise e

        if self.form == "2d":

            self._read_metadata_2d()

        else:  # form == "3d"

            self._read_metadata_3d()

    def _read_metadata_2d(self):

        sids = self._metadata["SeriesID"].unique().tolist()

        sid = sids[0]
        metadata_sid = self._metadata[self._metadata["SeriesID"] == sid]
        names = list(set(metadata_sid["Name"].tolist()))

        slices = dict()
        slices_nonsingleton = dict()
        # name = names[0]
        for name in names:
            metadata_sid_name = metadata_sid[metadata_sid["Name"] == name]
            sizes = metadata_sid_name[["Width", "Height", "Z", "T", "C"]]

            # Check time dimension
            if not all([sz == 1 for sz in sizes["T"]]):
                raise RuntimeError("Time series data currently not "
                                   "supported.")

            num_slices = len(list(set(sizes["Z"].tolist())))
            slices[name] = num_slices
            if num_slices > 1:
                slices_nonsingleton[name] = num_slices

        first_value = list(slices_nonsingleton.values())[0]
        if not all([s == first_value
                    for s in slices_nonsingleton.values()]):
            raise RuntimeError("The numbers of slices do not match!")

        self._indices = list()
        if len(slices_nonsingleton) > 0:

            # Read from first found, assume they have the same number of slices
            first_name = list(slices_nonsingleton.keys())[0]

            # sid_i = 0
            for sid_i in range(len(sids)):
                sid = sids[sid_i]

                metadata_sid = self._metadata[
                        self._metadata["SeriesID"] == sid]
                metadata_sid_name = metadata_sid[
                        metadata_sid["Name"] == first_name]

                # For each slice
                slices = metadata_sid_name["Z"].tolist()
                # slice_idx = 0
                for slice_idx in slices:

                    # List of all channels for this slice
                    ids = (metadata_sid_name[
                            metadata_sid_name["Z"] == slice_idx])["ID"]

                    # Add the first channel, if there are several
                    self._indices.append(ids.min())
        else:
            # sid_i = 0
            for sid_i in range(len(sids)):
                sid = sids[sid_i]

                ids = (self._metadata[self._metadata["SeriesID"] == sid])["ID"]
                self._indices.append(ids.min())

    def _read_metadata_3d(self):

        sids = self._metadata["SeriesID"].unique().tolist()

        self._indices = [None] * len(sids)
        # sid_i = 0
        for sid_i in range(len(sids)):
            sid = sids[sid_i]
            ids = (self._metadata[self._metadata["SeriesID"] == sid])["ID"]
            self._indices[sid_i] = ids.min()

    def _getitem_2d(self, idx):

        # Get the "idx"th patient
        id_ = self._indices[idx]

        row = self._metadata[self._metadata["ID"] == id_]

        # Get SeriesID
        sid = str(row["SeriesID"].tolist()[0])

        # Get slice index
        Z = int(row["Z"].tolist()[0])

        # All rows belonging to a particular SeriesID
        metadata_sid = self._metadata[self._metadata["SeriesID"] == sid]
        names = list(set(metadata_sid["Name"].tolist()))

        # Column names
        cols = self._get_column_names()

        # The return data
        item = dict()

        # name = names[0]
        for name in names:
            # Each channel (name)
            metadata_sid_name = metadata_sid[metadata_sid["Name"] == name]
            # sizes = metadata_sid_name[["Width", "Height", "Z", "T", "C"]]

            # Get a particular slice (list of all channels for a slice)
            if len(metadata_sid_name) == 1:  # Only one slice, or e.g. a scalar
                metadata_sid_name_Z = metadata_sid_name
            else:
                metadata_sid_name_Z = metadata_sid_name[
                        metadata_sid_name["Z"] == Z]
            channel_ids = [int(v)
                           for v in metadata_sid_name_Z["ID"].tolist()]

            tensor = None

            # channel_id_i = 0
            for channel_id_i in range(len(channel_ids)):
                channel_id = channel_ids[channel_id_i]

                data = self._select_query(
                        "SELECT " + ", ".join(cols) + " FROM Data "
                        "WHERE ID = ?;", (channel_id,))
                if len(data) != 1:
                    raise RuntimeError("The database has the wrong "
                                       "format!")

                data = data[0]

                data_type = data[cols.index("DataType")]
                class_ = data[cols.index("Class")]
                if class_ is None:
                    class_ = "tensor"

                # The actual data
                raw_data = data[cols.index("Data")]

                if class_ == "scalar":

                    fmt = "<"
                    if data_type == "float32":
                        fmt += "f"
                    elif data_type == "float64":
                        fmt += "d"
                    else:
                        raise NotImplementedError("Data type currently "
                                                  "not supported!")

                    value = struct.unpack(fmt, raw_data)[0]
                    # print("scalar: " + str(value))

                    if tensor is None:

                        if len(channel_ids) != 1:
                            raise RuntimeError(
                                    "The database has the wrong "
                                    "format. Scalars should not have "
                                    "channels.")

                        tensor = value

                else:  # "tensor"
                    if data_type != "float32":
                        raise ValueError("We can only process float32 "
                                         "tensors at this time.")

                    width = data[cols.index("Width")]
                    height = data[cols.index("Height")]

                    if data_type == "float32":
                        dt = np.dtype("float32")
                    else:
                        raise NotImplementedError("Data type currently "
                                                  "not supported!")

                    # Little endian always
                    # TODO: Put in settings table?
                    dt = dt.newbyteorder("<")

                    value = np.frombuffer(raw_data, dtype=dt)
                    value = value.reshape((height, width))
                    value = value.astype(np.float32)

                    if self.transform is not None:
                        value = self.transform(value)

                    if tensor is None:
                        shape = (width,
                                 height,
                                 1,
                                 len(channel_ids))
                        tensor = np.zeros(shape, dtype=np.float32)

                    # Check width and height
                    if (value.shape[0] != tensor.shape[0]) \
                            or (value.shape[1] != tensor.shape[1]):
                        raise RuntimeError("The data in the database are "
                                           "of different sizes. Use "
                                           "``transform`` to resize the "
                                           "slices.")

                    tensor[:, :, 0, channel_id_i] = value

            if self.transform_3d is not None:
                tensor = self.transform_3d(tensor)

            item[name] = tensor

        return item

    def _getitem_3d(self, idx):

        # Get the "idx"th patient
        id_ = self._indices[idx]

        row = self._metadata[self._metadata["ID"] == id_]

        # Get SeriesID
        series_id = str(row["SeriesID"].iloc[0])

        # Get names
        metadata_sid = self._metadata[
                self._metadata["SeriesID"] == series_id]
        names = list(sorted(set(metadata_sid["Name"].tolist())))

        # Column names
        cols = self._get_column_names()

        # The return data
        item = dict()

        # Fetch sizes for each "name"
        # name = names[0]
        for name in names:
            metadata_sid_name = metadata_sid[metadata_sid["Name"] == name]
            sizes = metadata_sid_name[["Width", "Height", "Z", "T", "C"]]

            # Get slice ids, and assume they are of the same length (crash
            # later)
            slice_idcs = list(set(sizes["Z"].tolist()))

            # Check time dimension
            if not all([sz == 1 for sz in sizes["T"]]):
                raise RuntimeError("Time series data currently not "
                                   "supported.")

            # Get channel ids
            channels = list(set(sizes["C"].tolist()))

            tensor = None

            # channel_i = 0
            for channel_i in range(len(channels)):
                channel = channels[channel_i]

                # slice_idx_i = 0
                for slice_idx_i in range(len(slice_idcs)):
                    slice_idx = slice_idcs[slice_idx_i]

                    metadata_sid_name_c \
                        = metadata_sid_name[
                                metadata_sid_name["C"] == channel]
                    metadata_sid_name_c_z \
                        = metadata_sid_name_c[
                                metadata_sid_name_c["Z"] == slice_idx]
                    if len(metadata_sid_name_c_z) != 1:
                        raise RuntimeError("The database has the wrong "
                                           "format!")

                    data = self._select_query(
                            "SELECT " + ", ".join(cols) + " FROM Data "
                            "WHERE ID = ?;",
                            (metadata_sid_name_c_z["ID"].tolist()[0],))
                    if len(data) != 1:
                        raise RuntimeError("The database has the wrong "
                                           "format!")
                    data = data[0]
                    data_type = data[cols.index("DataType")]
                    class_ = data[cols.index("Class")]
                    if class_ is None:
                        class_ = "tensor"

                    # The actual data
                    raw_data = data[cols.index("Data")]

                    if class_ == "scalar":

                        # Little endian always
                        # TODO: Put in settings table?
                        fmt = "<"
                        if data_type == "float32":
                            fmt += "f"
                        elif data_type == "float64":
                            fmt += "d"
                        else:
                            raise NotImplementedError("Data type currently "
                                                      "not supported!")

                        value = struct.unpack(fmt, raw_data)[0]

                        if tensor is None:

                            if len(channels) != 1:
                                raise RuntimeError(
                                        "The database has the wrong format. "
                                        "Scalars should not have channels.")

                            if len(slice_idcs) != 1:
                                raise RuntimeError(
                                        "The database has the wrong format. "
                                        "Scalars should not have slices.")

                            tensor = value

                    else:  # "tensor"
                        if data_type != "float32":
                            raise ValueError("We can only process float32 "
                                             "tensors at this time.")

                        width = data[cols.index("Width")]
                        height = data[cols.index("Height")]

                        if data_type == "float32":
                            dt = np.dtype("float32")
                        else:
                            raise NotImplementedError("Data type currently "
                                                      "not supported!")

                        # Little endian always
                        # TODO: Put in settings table?
                        dt = dt.newbyteorder("<")

                        value = np.frombuffer(raw_data, dtype=dt)
                        value = value.reshape((height, width))
                        value = value.astype(np.float32)

                        if self.transform is not None:
                            value = self.transform(value)

                        if tensor is None:
                            shape = (width,
                                     height,
                                     len(slice_idcs),
                                     len(channels))
                            tensor = np.zeros(shape, dtype=np.float32)

                        # Check width and height
                        if (value.shape[0] != tensor.shape[0]) \
                                or (value.shape[1] != tensor.shape[1]):
                            raise RuntimeError("The data in the database are "
                                               "of different sizes. Use "
                                               "``transform`` to resize the "
                                               "slices.")

                        tensor[:, :, slice_idx_i, channel_i] = value

            if self.transform_3d is not None:
                tensor = self.transform_3d(tensor)

            item[name] = tensor

        return item

    def __len__(self):

        if (self._metadata is None) or (self._indices is None):
            self._read_metadata()

        return len(self._indices)

    def __getitem__(self, idx):

        if (self._metadata is None) or (self._indices is None):
            self._read_metadata()

        idx = int(idx)

        if self.form == "2d":
            tensor = self._getitem_2d(idx)

        else:  # self.form == "3d"
            tensor = self._getitem_3d(idx)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor


class DicomDataset(with_metaclass(abc.ABCMeta, object)):
    r"""Base class for Dicom datasets.
    """
    def __init__(self,
                 dir_path,
                 image_names=None,
                 exact_image_names=True,
                 channel_names=None,
                 channel_output_names=None,
                 transform=None,
                 cache_size=None,
                 data_format=None):
        r"""
        Parameters
        ----------
        dir_path : str
            Path to the directory containing the images. The subdirectories of
            this directory represents (and contains) the 3-dimensional images.

        image_names : list of str, optional
            The subdirectories to extract files from below the ``dir_path``
            directory. Every element of this list corresponds to an image that
            will be read. If a subdirectory is not in this list, it will not
            be read. If ``exact_image_name`` is ``True``, the elements may be
            regular expressions. Default is ``None``, which means to read all
            subdirectories.

        exact_image_names : bool, optional
            Whether or not to interpret the elements of ``image_names`` as
            regular expressions or not. If ``True``, the names will not be
            interpreted as regular expressions, but will be interpreted as
            constant exact strings; and if ``False``, the names will be
            interpreted as regular expressions. Default is ``True``, do not
            interpret as regular expressions.

        channel_names : list of str or list of str, optional
            The inner strings or lists corresponds to directory names or
            regular expressions defining the names of the subdirectories under
            ``image_names`` that corresponds to channels of this image. Every
            outer element of this list corresponds to a channel of the images
            defined by ``image_names``. The elements of the inner lists are
            alternative names for the subdirectories. If more than one
            subdirectory name matches, only the first one found will be used.
            Default is ``None``, which means to read all channels (note that
            this may mean that the images have different channels, if their
            subdirectories mismatch.)

        channel_output_names : list of str, optional
            Custom names for the output images. The output is a ``dict``, and
            the keys will be the elements of ``channel_output_names``. If
            ``None``, the names will instead be the corresponding channel names
            found using ``channel_names``. If ``channel_names`` is also None,
            then a ``ValueError`` exception is raised.

        transform : callable, optional
            Custom transform to apply to each volume. Default is ``None``,
            which means to not apply any transform.

        cache_size : float or int, optional
            The cache size in gigabytes (GiB, 2**30 bytes). If a value is
            given, it must correspond to at least one byte (``1 / 2**30``). The
            default value is ``None``, which means to not use a cache (nothing
            is stored). Elements are dropped from the cache whenever the stored
            data reaches ``cache_size``, the policy is first in first out (old
            elements are dropped first).

        data_format : str, optional
            One of `channels_last` (default) or `channels_first`. The ordering
            of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while
            `channels_first` corresponds to inputs with shape `(batch,
            channels, height, width)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`. If
            you never set it, then it will be "channels_last".
        """
        if not _HAS_PYDICOM:
            raise RuntimeError('The "pydicom" package is not available.')

        self.dir_path = str(dir_path)
        if not os.path.exists(self.dir_path):
            raise ValueError("The given path does not exist: %s" % (dir_path,))

        if image_names is None:
            self.image_names = None
        elif isinstance(image_names, (list, tuple)):
            self.image_names = []
            for name in image_names:
                if isinstance(name, str):
                    self.image_names.append(str(name))
                else:
                    raise ValueError('``image_names`` must be a list of '
                                     'strings.')
        else:
            raise ValueError('``image_names`` must be a list of strings.')

        self.exact_image_names = bool(exact_image_names)

        if channel_names is None:
            self.channel_names = None
        elif isinstance(channel_names, (list, tuple)):
            self.channel_names = []
            for channel in channel_names:
                if isinstance(channel, str):
                    self.channel_names.append([str(channel)])
                elif isinstance(channel, (list, tuple)):
                    self.channel_names.append([str(name) for name in channel])
                else:
                    raise ValueError('``channel_names`` must be a list of '
                                     'either strings or lists of strings.')
        else:
            raise ValueError('``channel_names`` must be a list of either '
                             'strings or lists of strings.')

        if channel_output_names is None:
            if channel_names is None:
                raise ValueError("Both ``channel_output_names`` and "
                                 "``channel_names`` can't be ``None`` "
                                 "simultaneously.")
            else:
                self.channel_output_names = None
        elif isinstance(channel_output_names, (list, tuple)):
            self.channel_output_names = \
                    [str(name) for name in channel_output_names]
            if len(self.channel_output_names) != len(self.channel_names):
                raise ValueError("The ``channel_output_names`` and "
                                 "``channel_names`` must have the same "
                                 "length.")
            if len(self.channel_output_names) \
                    != len(set(self.channel_output_names)):
                raise ValueError("The elements in ``channel_output_names`` "
                                 " must be unique.")
        else:
            raise ValueError("The ``channel_output_names`` must be a list of "
                             "strings.")

        if transform is None:
            self.transform = None
        else:
            if callable(transform):
                self.transform = transform
            else:
                raise ValueError('``transform`` must be callable.')

        if cache_size is None:
            self.cache_size = None
        else:
            self.cache_size = max(1.0 / 2**30, float(cache_size))
            self._cache = dict()
            self._cache_order = list()
            self._cache_cur_size = 0

        self.data_format = normalize_data_format(data_format)

        self._filtered_image_names = self._get_image_names()

    def _listdir(self, channel_path):
        """Lists all DICOM images in a given folder.
        """
        files = os.listdir(channel_path)

        dicom_files = [None] * (len(files) + 1)
        num_dicom_files = len(dicom_files)
        found_zero = False  # Inconsistent use of indices starting with 0 or 1
        num_dicom_found = 0
        for file in files:
            file_path = os.path.join(channel_path, file)

            try:
                data = pydicom.dcmread(file_path,
                                       stop_before_pixels=True,
                                       force=False,
                                       specific_tags=["InstanceNumber"])

                if hasattr(data, "InstanceNumber"):
                    slice_index = data.InstanceNumber
                    dicom_files[slice_index] = file

                    if slice_index == 0:
                        found_zero = True

                    num_dicom_found += 1

            except pydicom.errors.InvalidDicomError:
                pass  # Skip file if not a DICOM file

        # Remove first or last, depending on indexing starting with 0 or 1
        if found_zero:
            if dicom_files[-1] is not None:
                raise RuntimeError("The order of the slices is not "
                                   "consistent.")
            del dicom_files[-1]
            num_dicom_files -= 1
        else:
            if dicom_files[0] is not None:
                raise RuntimeError("The order of the slices is not "
                                   "consistent.")
            del dicom_files[0]
            num_dicom_files -= 1

        # Remove any extra (non DICOM) files
        for i in range(num_dicom_files - 1, num_dicom_found - 1, -1):
            del dicom_files[i]  # Remove unused spots

        # Check that all slices were found
        for file in dicom_files:
            if file is None:
                raise RuntimeError("All slices could not be found among the "
                                   "files in the directory.")

        return dicom_files


class Dicom3DDataset(DicomDataset):
    r"""A dataset abstraction over 3D Dicom images in a given directory.

    The images are organised in a directory for each image, a subdirectory
    for each channel, and the third-dimension slices for each channel are
    in those subdirectories. E.g., the directory tree

        Im1/A/im1.dcm
          ...
        Im1/A/imN.dcm
        Im1/B/im1.dcm
          ...
        Im1/B/imN.dcm
        Im2/A/im1.dcm
          ...
        Im2/A/imN.dcm
        Im2/B/im1.dcm
          ...
        Im2/B/imN.dcm

    Thus contains two 3-dimensional images each with two channels (A and B),
    and are N slices deep (the slices are ordered according to the
    InstanceNumber tag, and not by their file names).

    It will be assumed that the subdirectories of a given image directory
    contains different "channels" (different image modes, for instance), and
    they will be returned as such. The channel subdirectories and their order
    is determined by the list ``channel_names``.

    It will be assumed that the Dicom files have some particular tags. It will
    be assumed that they have: "InstanceNumber", "RescaleSlope",
    "RescaleIntercept", "Rows", and "Columns". If these tags are missing, the
    files cannot be read and an exception will be raised.

    This dataset requires that the ``pydicom`` package be installed.
    """
    def __init__(self,
                 dir_path,
                 image_names=None,
                 exact_image_names=True,
                 channel_names=None,
                 channel_output_names=None,
                 transform=None,
                 cache_size=None,
                 data_format=None):
        """
        Parameters
        ----------
        dir_path : str
            Path to the directory containing the images. The subdirectories of
            this directory represents (and contains) the 3-dimensional images.

        image_names : list of str, optional
            The subdirectories to extract files from below the ``dir_path``
            directory. Every element of this list corresponds to an image that
            will be read. If a subdirectory is not in this list, it will not
            be read. If ``exact_image_name`` is ``True``, the elements may be
            regular expressions. Default is ``None``, which means to read all
            subdirectories.

        exact_image_names : bool, optional
            Whether or not to interpret the elements of ``image_names`` as
            regular expressions or not. If ``True``, the names will not be
            interpreted as regular expressions, but will be interpreted as
            constant exact strings; and if ``False``, the names will be
            interpreted as regular expressions. Default is ``True``, do not
            interpret as regular expressions.

        channel_names : list of str or list of str, optional
            The inner strings or lists corresponds to directory names or
            regular expressions defining the names of the subdirectories under
            ``image_names`` that corresponds to channels of this image. Every
            outer element of this list corresponds to a channel of the images
            defined by ``image_names``. The elements of the inner lists are
            alternative names for the subdirectories. If more than one
            subdirectory name matches, only the first one found will be used.
            Default is ``None``, which means to read all channels (note that
            this may mean that the images have different channels, if their
            subdirectories mismatch.)

        channel_output_names : list of str, optional
            Custom names for the output images. The output is a ``dict``, and
            the keys will be the elements of ``channel_output_names``. If
            ``None``, the names will instead be the corresponding channel names
            found using ``channel_names``. If ``channel_names`` is also None,
            then a ``ValueError`` exception is raised.

        transform : callable, optional
            Custom transform to apply to each volume. Default is ``None``,
            which means to not apply any transform.

        cache_size : float or int, optional
            The cache size in gigabytes (GiB, 2**30 bytes). If a value is
            given, it must correspond to at least one byte (``1 / 2**30``). The
            default value is ``None``, which means to not use a cache (nothing
            is stored). Elements are dropped from the cache whenever the stored
            data reaches ``cache_size``, the policy is first in first out (old
            elements are dropped first).

        data_format : str, optional
            One of `channels_last` (default) or `channels_first`. The ordering
            of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while
            `channels_first` corresponds to inputs with shape `(batch,
            channels, height, width)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`. If
            you never set it, then it will be "channels_last".
        """
        super().__init__(dir_path,
                         image_names=image_names,
                         exact_image_names=exact_image_names,
                         channel_names=channel_names,
                         channel_output_names=channel_output_names,
                         transform=transform,
                         cache_size=cache_size,
                         data_format=data_format)

    def _get_image_names(self):
        # TODO: May be slow in case there are many files and many qualifiers

        # Get all subdirectories (images)
        images_ = os.listdir(self.dir_path)
        images = []
        for im in images_:
            if self.image_names is None:
                images.append(im)  # In this case we add all subdirectories
            else:
                # Here we only add them if they match the given image names
                if self.exact_image_names:  # Exact match
                    if im in self.image_names:
                        images.append(im)
                else:  # Regular expression match
                    for name in self.image_names:
                        if re.match(name, im):
                            images.append(im)

        return images

    def _get_channel_dirs(self, index):

        dir_path = self.dir_path  # "~/data"
        image_names = self._filtered_image_names  # ["Patient 1", "Patient 2"]
        channel_names = self.channel_names  # [["CT.*", "[CT].*"], ["MR.*"]]

        image_name = image_names[index]  # "Patient 1"
        image_path = os.path.join(dir_path, image_name)  # "~/data/Patient 1"
        channel_dirs_ = os.listdir(image_path)  # ["CT", "MR"]
        channel_dirs = []
        if channel_names is None:
            for channel in channel_dirs_:  # channel = "CT"
                channel_dirs.append(channel)  # channel_dirs = ["CT"]
        else:
            for channel_name in channel_names:  # channel_name = ["CT.*", "[CT].*"]
                found = False
                for channel_re in channel_name:  # channel_re = "CT.*"
                    for channel in channel_dirs_:  # channel = "CT"
                        if re.match(channel_re, channel):
                            channel_dirs.append(channel)  # channel_dirs = ["CT"]
                            found = True
                            break
                    if found:
                        break
                else:
                    raise RuntimeError("Channel %s was not found for image %s"
                                       % (channel_re, image_name))
        # channel_dirs = ["CT", "MR"]
        # return channel_dirs

        all_channel_files = list()
        all_channel_names = list()
        channel_length = None
        for channel_dir_i in range(len(channel_dirs)):  # 0
            channel_dir = channel_dirs[channel_dir_i]  # channel_dir = "CT"
            channel_path = os.path.join(image_path,
                                        channel_dir)  # "~/data/Patient 1/CT"
            dicom_files = self._listdir(channel_path)  # ["im1.dcm", "im2.dcm"]

            # Check that channels have the same length
            if channel_length is None:
                channel_length = len(dicom_files)
            else:
                if channel_length != len(dicom_files):
                    raise RuntimeError("The numbers of slices for channel %s "
                                       "and channel %s do not agree."
                                       % (channel_dir, channel_dirs[0]))

            # TODO: Also check image sizes within a channel.

            # Create full relative or absolute path for all slices
            full_file_names = []
            for file in dicom_files:
                dicom_file = os.path.join(channel_path,
                                          file)  # "~/data/.../CT/im1.dcm"

                full_file_names.append(dicom_file)  # ["~/data/.../im1.dcm"]

            # {"CT": ["~/data/Patient 1/CT/im1.dcm"]}
            if self.channel_output_names is not None:
                channel_dir = self.channel_output_names[channel_dir_i]
            all_channel_files.append(full_file_names)
            all_channel_names.append(channel_dir)

        # [[...], [...]], ["CT", "MR"]
        return all_channel_files, all_channel_names

    def _read_image(self, files):

        slices = [None] * (len(files) + 1)  # Allocate for zero as well
        found_zero = False
        for file in files:
            data = pydicom.dcmread(file)
            image = data.pixel_array.astype(float)

            slice_index = int(float(data.InstanceNumber) + 0.5)
            slope = float(data.RescaleSlope)
            intercept = float(data.RescaleIntercept)

            if slice_index == 0:
                found_zero = True

            # Convert to original units
            image = image * slope + intercept

            if slices[slice_index] is None:
                slices[slice_index] = image
            else:
                raise RuntimeError("The same slice number (InstanceNumber) "
                                   "appeared twice.")

        if found_zero:
            if slices[-1] is not None:
                raise RuntimeError("The order of the slices is not "
                                   "consistent.")
            del slices[-1]
        else:
            if slices[0] is not None:
                raise RuntimeError("The order of the slices is not "
                                   "consistent.")
            del slices[0]

        for slice_ in slices:
            if slice_ is None:
                raise RuntimeError("All slices could not be found among the "
                                   "files in the directory.")

        slices = np.array(slices)

        return slices

    def __getitem__(self, index):

        # Use cache and in cache?
        if (self.cache_size is not None) and (index in self._cache):
            image = dict(self._cache[index][0])  # Use stored data
        else:
            channel_files, channel_names = self._get_channel_dirs(index)

            image = dict()
            for channel_i in range(len(channel_names)):
                channel_image = self._read_image(channel_files[channel_i])

                channel_image = np.transpose(channel_image, axes=[1, 2, 0])

                if self.data_format == "channels_last":
                    channel_image = channel_image[..., np.newaxis]
                else:
                    channel_image = channel_image[np.newaxis, ...]

                image[channel_names[channel_i]] = channel_image

            if self.cache_size is not None:  # Use cache?
                this_size = utils.sizeof(image)
                while self._cache_cur_size + this_size > self.cache_size * 2**30:
                    index_drop = self._cache_order[0]
                    index_size = self._cache[index_drop][1]
                    del self._cache[index_drop]
                    self._cache_cur_size -= index_size
                    self._cache_order = self._cache_order[1:]

                self._cache[index] = [dict(image), this_size]
                self._cache_cur_size += this_size
                self._cache_order.append(index)

        # Perform transform last, then augmentation will work as well
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self._filtered_image_names)


class Dicom2DDataset(DicomDataset):
    r"""A dataset abstraction over Dicom images (or image slices).

    The images are organised in a directory for each image, a subdirectory
    for each channel, and the third-dimension slices (or independent images)
    for each channel are in those subdirectories. E.g., the directory tree

        Im1/A/im1.dcm
          ...
        Im1/A/imN.dcm
        Im1/B/im1.dcm
          ...
        Im1/B/imN.dcm
        Im2/A/im1.dcm
          ...
        Im2/A/imN.dcm
        Im2/B/im1.dcm
          ...
        Im2/B/imN.dcm

    Thus contains two (possibly 3-dimensional) images each with two channels (A
    and B), and each is made up of N images (possibly slice, and if so: the
    slices are ordered according to the InstanceNumber tag, and not by their
    file names).

    It will be assumed that the subdirectories of a given image directory
    contains different "channels" (different image modes, for instance), and
    they will be returned as such. The channel subdirectories and their order
    is determined by the list ``channel_names``.

    It will be assumed that the Dicom files have some particular tags. It will
    be assumed that they have: "InstanceNumber", "RescaleSlope",
    "RescaleIntercept", "Rows", and "Columns". If these tags are missing, the
    files cannot be read and an exception will be raised.

    This dataset requires that the ``pydicom`` package be installed.
    """
    def __init__(self,
                 dir_path,
                 image_names=None,
                 exact_image_names=True,
                 channel_names=None,
                 channel_output_names=None,
                 transform=None,
                 cache_size=None,
                 data_format=None):
        """
        Parameters
        ----------
        dir_path : str
            Path to the directory containing the images. The subdirectories of
            this directory represents (and contains) the 2-dimenaional images
            (e.g., slices).

        image_names : list of str, optional
            The subdirectories to extract files from below the ``dir_path``
            directory. Every element of this list corresponds to an image that
            will be read. If a subdirectory is not in this list, it will not
            be read. If ``exact_image_name`` is ``True``, the elements may be
            regular expressions. Default is ``None``, which means to read all
            subdirectories.

        exact_image_names : bool, optional
            Whether or not to interpret the elements of ``image_names`` as
            regular expressions or not. If ``True``, the names will not be
            interpreted as regular expressions, but will be interpreted as
            constant exact strings; and if ``False``, the names will be
            interpreted as regular expressions. Default is ``True``, do not
            interpret as regular expressions.

        channel_names : list of str or list of str, optional
            The inner strings or lists corresponds to directory names or
            regular expressions defining the names of the subdirectories under
            ``image_names`` that corresponds to channels of this image. Every
            outer element of this list corresponds to a channel of the images
            defined by ``image_names``. The elements of the inner lists are
            alternative names for the subdirectories. If more than one
            subdirectory name matches, only the first one found will be used.
            Default is ``None``, which means to read all channels (note that
            this may mean that the images end up with different channels, if
            their subdirectories mismatch).

        channel_output_names : list of str, optional
            Custom names for the output images. The output is a ``dict``, and
            the keys will be the elements of ``channel_output_names``. If
            ``None``, the names will instead be the corresponding channel names
            found using ``channel_names``. If ``channel_names`` is also None,
            then a ``ValueError`` exception is raised.

        transform : callable, optional
            Custom transform to apply to each 2-dimensional image. Default is
            ``None``, which means to not apply any transform.

        cache_size : float or int, optional
            The cache size in gigabytes (GiB, 2**30 bytes). If a value is
            given, it must correspond to at least one byte (``1 / 2**30``). The
            default value is ``None``, which means to not use a cache (nothing
            is stored). Elements are dropped from the cache whenever the stored
            data reaches ``cache_size``, the policy is first in first out (old
            elements are dropped first).

        data_format : str, optional
            One of `channels_last` (default) or `channels_first`. The ordering
            of the dimensions in the inputs. `channels_last` corresponds to
            inputs with shape `(batch, height, width, channels)` while
            `channels_first` corresponds to inputs with shape `(batch,
            channels, height, width)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`. If
            you never set it, then it will be "channels_last".
        """
        super().__init__(dir_path,
                         image_names=image_names,
                         exact_image_names=exact_image_names,
                         channel_names=channel_names,
                         channel_output_names=channel_output_names,
                         transform=transform,
                         cache_size=cache_size,
                         data_format=data_format)

        self._all_images, self._image_names = self._get_all_images()
        if self.channel_output_names is not None:
            self._image_names = self.channel_output_names

    def _get_image_names(self):
        # TODO: May be slow in case there are many files and many qualifiers

        # Get all subdirectories (images)
        images_ = os.listdir(self.dir_path)
        images = []
        for im in images_:
            if self.image_names is None:
                images.append(im)  # In this case we add all subdirectories
            else:
                # Here we only add them if they match the given image names
                if self.exact_image_names:  # Exact match
                    if im in self.image_names:
                        images.append(im)
                else:  # Regular expression match
                    for name in self.image_names:
                        if re.match(name, im):
                            images.append(im)

        return images

    def _get_all_images(self):
        """Returns a list of all the files included in this dataset in order.

        TODO: With many files, this function may be overly slow. There is no
        need to list all files immediately, we could do this in a lazy fashion.
        """
        dir_path = self.dir_path  # "~/data"
        image_names = self._filtered_image_names  # ["Patient 1", "Patient 2"]
        channel_names = self.channel_names  # [["CT.*", "[CT].*"], ["MR.*"]]

        channel_dirs = {}
        all_channel_names = None
        for image_name in image_names:  # "Patient 1"
            channel_dirs[image_name] = []
            # "~/data/Patient 1"
            image_path = os.path.join(dir_path, image_name)
            channel_dirs_ = os.listdir(image_path)  # ["CT", "MR"]
            if channel_names is None:
                for channel in channel_dirs_:  # channel = "CT"
                    # channel_dirs = ["CT"]
                    channel_dirs[image_name].append(channel)
            else:
                # channel_name = ["CT.*", "[CT].*"]
                for channel_name in channel_names:
                    found = False
                    for channel_re in channel_name:  # channel_re = "CT.*"
                        for channel in channel_dirs_:  # channel = "CT"
                            if re.match(channel_re, channel):
                                # channel_dirs = ["CT"]
                                channel_dirs[image_name].append(channel)
                                found = True
                                break
                        if found:
                            break
                    else:
                        raise RuntimeError("Channel %s was not found for "
                                           "image %s"
                                           % (channel_re, image_name))
                if all_channel_names is None:
                    all_channel_names = channel_dirs[image_name]
                elif all_channel_names != channel_dirs[image_name]:
                    raise RuntimeError("The channels are inconsistent between "
                                       "images.")

        # channel_dirs = {"Patient 1": ["CT", "MR"], ...}

        all_images = list()
        channel_length = None
        for image_name in image_names:  # "Patient 1"
            all_channel_files = dict()
            channel_length = None
            for channel_dir_i in range(len(channel_dirs[image_name])):  # 0
                # channel_dir = "CT"
                channel_dir = channel_dirs[image_name][channel_dir_i]
                # "~/data/Patient 1/CT"
                channel_path = os.path.join(image_path, channel_dir)
                # ["im1.dcm", "im2.dcm"]
                dicom_files = self._listdir(channel_path)

                # Check that channels have the same length
                if channel_length is None:
                    channel_length = len(dicom_files)
                else:
                    if channel_length != len(dicom_files):
                        raise RuntimeError("The numbers of slices for channel "
                                           "%s and channel %s do not agree."
                                           % (channel_dir, channel_dirs[0]))

                # TODO: Also check image sizes within a channel.

                # Create full relative or absolute path for all slices
                full_file_names = []
                for file in dicom_files:
                    dicom_file = os.path.join(channel_path,
                                              file)  # "~/data/.../CT/im1.dcm"

                    # ["~/data/.../im1.dcm"]
                    full_file_names.append(dicom_file)

                # {"CT": ["~/data/Patient 1/CT/im1.dcm"]}
                all_channel_files[channel_dir] = full_file_names

            # Make a list of (CT, MR) tuples (file names)
            all_channel_names_ = []
            for channel_name in all_channel_names:
                all_channel_names_.append(all_channel_files[channel_name])
            all_channel_names_ = list(zip(*all_channel_names_))

            all_images.extend(all_channel_names_)

        # Return the pairs of filenames and the name of the resp. modalities
        return all_images, all_channel_names

    def _read_image_slice(self, file):

        data = pydicom.dcmread(file)
        image = data.pixel_array.astype(float)

        slope = float(data.RescaleSlope)
        intercept = float(data.RescaleIntercept)

        # Convert to original units
        image = image * slope + intercept

        return image

    def __getitem__(self, index):

        # Use cache and in cache?
        if (self.cache_size is not None) and (index in self._cache):
            image = dict(self._cache[index][0])  # Use stored data
        else:
            images = self._all_images[index]
            image = {}
            for channel_i in range(len(self._image_names)):
                channel = self._image_names[channel_i]
                channel_image = self._read_image_slice(images[channel_i])
                if self.data_format == "channels_last":
                    channel_image = channel_image[..., np.newaxis]
                else:
                    channel_image = channel_image[np.newaxis, ...]
                image[channel] = channel_image

            if self.cache_size is not None:  # Use cache?
                this_size = utils.sizeof(image)
                while self._cache_cur_size + this_size > self.cache_size * 2**30:
                    index_drop = self._cache_order[0]
                    index_size = self._cache[index_drop][1]
                    del self._cache[index_drop]
                    self._cache_cur_size -= index_size
                    self._cache_order = self._cache_order[1:]

                self._cache[index] = [dict(image), this_size]
                self._cache_cur_size += this_size
                self._cache_order.append(index)

        # Perform transform last, then augmentation will work as well
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self._all_images)


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
                top = self.random_state.randint(0,
                                                max(1, image.shape[0] - crop0))
                left = self.random_state.randint(0,
                                                 max(1,
                                                     image.shape[1] - crop1))
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
    def __init__(self,
                 X,
                 batch_size=32,
                 wrap_around=False,
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

        if not _HAS_PYDICOM:
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

        self.data_format = normalize_data_format(data_format)

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
            channel_path = os.path.join(image_path,
                                        channel_dir)  # "~/data/Pat 1/CT"
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
                dicom_file = os.path.join(channel_path,
                                          file)  # "~/data/Pat 1/CT/im1.dcm"

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
                        indices = self.random_state.choice(
                                len(files),
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
        try:
            data = pydicom.dcmread(file_name)
        except (AttributeError):  # dicom, will be deprecated!
            data = pydicom.read_file(file_name)

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
                queue_lim = max(
                        1,
                        (self.random_pool_size - 1) * self._average_num_slices)
            if self._file_queue_len() < queue_lim:
                update_done = self._file_queue_update()

            file_names = None
            try:
                file_names = self._file_queue_pop()
            except (IndexError):
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
                        images = np.transpose(images,
                                              (1, 2, 0))  # Channels last
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

        if not _HAS_PYDICOM:
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

        self.data_format = normalize_data_format(data_format)

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
            data = pydicom.read_file(file_name)
        except (pydicom.filereader.InvalidDicomError, FileNotFoundError):
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
            except (IndexError):
                if not update_done:  # Images not added, no images in queue
                    self.throw(StopIteration)

            if file_name is not None:
                image = self._read_image(file_name)
                if image is not None:
                    image = self._process_image(image)

                    return_images.append(image)

        return return_images


class Numpy2DGenerator(BaseGenerator):
    """A generator over 2D images in a given directory, saved as npz files.

    It will be assume that all npz files in the directory are to be read. The
    returned data will have shape ``(batch_size, height, width, channels)`` or
    ``(batch_size, channels, height, width)``, depending on the value of
    ``data_format``.

    Parameters
    ----------
    dir_path : str, or list of str
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

    pool_size : int, optional
        A pool of buffered images to be kept in memory at all times. It may not
        be possible to load all of them in memory at once, why a pool is used
        instead. If ``randomize_order=True``, the pool is randomised as well.
        The value of ``pool_size`` determines how many files will be read and
        kept in the pool at the same time. When the number of slices in the
        pool falls below ``pool_size``, a new image will automatically be
        loaded into the pool, and if ``randomize_order=True`` the pool will be
        reshuffled, to improve the mixing. If the ``pool_size`` is small, only
        a few image will be kept in the pool at any given time, and
        mini-batches may not be independent, depending on if the images are
        order-dependent. If possible, the value of ``pool_size`` should be set
        to the number of files in the given directory. Default is 100, which
        means to use a pool of 100 images.

    randomize_order : bool, optional
        Whether or not to randomise the order of the images as they are read.
        The order will be completely random if ``pool_size`` is the same size
        as the number of npz files in the given directory. Otherwise, only
        ``pool_size`` npz files will be loaded at any given time and only
        randomized within the loaded set of files. Use ``pool_size`` in order
        to control the amount of randomness in the outputs. Default is False,
        do not randomise the order of the images.

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
                 pool_size=100,
                 randomize_order=False,
                 data_format=None,
                 random_state=None):

        if isinstance(dir_path, six.string_types):
            dir_path = [dir_path]
        dir_path = list(dir_path)
        self.dir_path = [str(d) for d in dir_path]
        self.image_key = str(image_key)

        if file_names is None:
            self.file_names = None
        elif isinstance(file_names, six.string_types):
            self.file_names = [str(file_names)]
        else:
            self.file_names = [str(file_name) for file_name in file_names]

        if batch_size is None:
            self.batch_size = batch_size
        else:
            self.batch_size = max(1, int(batch_size))

        self.restart_generation = bool(restart_generation)
        self.pool_size = max(1, int(pool_size))
        self.randomize_order = bool(randomize_order)
        self.data_format = normalize_data_format(data_format)
        self.random_state = utils.normalize_random_state(
                random_state, rand_functions=["rand", "randint", "choice"])

        self._all_files = self._list_dir()  # ["~usr/im1.npz", "~usr/im2.npz"]
        self._file_i = 0
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
        all_images = []
        for dir_ in self.dir_path:
            all_files = os.listdir(dir_)
            for file in all_files:
                if file.endswith(".npz"):
                    full_file = os.path.join(dir_, file)
                    if os.path.isfile(full_file):
                        if self.file_names is None:
                            all_images.append(full_file)
                        elif file in self.file_names:
                            all_images.append(full_file)

        return all_images

    def _read_file(self, file):

        image = np.load(file)

        return image[self.image_key]

    def _slice_queue_update(self, file):

        try:
            # full_file = os.path.join(self.dir_path, file)
            # image = self._read_file(full_file)
            image = self._read_file(file)
        except (KeyError):
            raise ValueError('Key "%s" does not exist in image file "%s".'
                             % (self.image_key, file))
        except (Exception):  # TODO: Specify other possible exceptions?
            return False

        self._slice_queue_push(image)

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
        super(Numpy2DGenerator, self).throw(typ, **kwargs)

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

        return_images = []
        while (self.batch_size is None) \
                or (len(return_images) < self.batch_size):

            while not self._throw_stop_iteration:
                # If we have enough slices, don't update
                if len(self._slice_queue) >= self.pool_size:
                    break

                self._file_i += 1
                if (self.batch_size is None) and self.restart_generation:

                    if self._file_i >= len(self._all_files):
                        self._file_i = 0  # Restart next time
                        break  # Do not read any more files now

                elif (self.batch_size is None) \
                        and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                elif (self.batch_size is not None) and self.restart_generation:

                    if self._file_i >= len(self._all_files):
                        self._file_i = 0  # Restart from first file

                else:  # (self.batch_size is not None) \
                    #        and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                self._slice_queue_update(self._all_files[self._file_i])

            try:
                image = self._slice_queue_pop()
                return_images.append(image)
            except (IndexError):  # Empty queue
                if (self.batch_size is None):
                    break  # Done, return images
                elif (not self.restart_generation):
                    # We did not end on a full batch
                    if len(return_images) != self.batch_size:
                        self._throw_stop_iteration = True
                        self.throw(StopIteration)

        return return_images


class Numpy3DGenerator(BaseGenerator):
    """A generator over 3D images in a given directory, saved as npz files.

    It will be assume that all npz files in the directory are to be read. The
    returned data will have shape ``(batch_size, height, width, channels)`` or
    ``(batch_size, channels, height, width)``, depending on the value of
    ``data_format``.

    Parameters
    ----------
    dir_path : str, or list of str
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

        if isinstance(dir_path, six.string_types):
            dir_path = [dir_path]
        dir_path = list(dir_path)
        self.dir_path = [str(d) for d in dir_path]
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

        self.data_format = normalize_data_format(data_format)
        self.random_state = utils.normalize_random_state(
                random_state, rand_functions=["rand", "randint", "choice"])

        self._all_files = self._list_dir()  # ["~usr/im1.npz", "~usr/im2.npz"]
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
        all_images = []
        for dir_ in self.dir_path:
            all_files = os.listdir(dir_)
            for file in all_files:
                if file.endswith(".npz"):
                    full_file = os.path.join(dir_, file)
                    if os.path.isfile(full_file):
                        if self.file_names is None:
                            all_images.append(full_file)
                        elif file in self.file_names:
                            all_images.append(full_file)

        return all_images

    def _read_file(self, file):

        image = np.load(file)

        return image[self.image_key]

    def _slice_queue_update(self, file):

        try:
            # full_file = os.path.join(self.dir_path, file)
            # image = self._read_file(full_file)
            image = self._read_file(file)
        except (KeyError):
            raise ValueError('Key "%s" does not exist in image file "%s".'
                             % (self.image_key, file))
        except (Exception):  # TODO: Specify other possible exceptions?
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

                elif (self.batch_size is None) \
                        and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                elif (self.batch_size is not None) \
                        and self.restart_generation:

                    if self._file_i >= len(self._all_files):
                        self._file_i = 0  # Restart from first file

                else:  # (self.batch_size is not None) \
                    #        and (not self.restart_generation):

                    if self._file_i >= len(self._all_files):
                        # Don't restart next time
                        self._throw_stop_iteration = True
                        break  # Do not read any more files now

                self._slice_queue_update(self._all_files[self._file_i])

            try:
                image = self._slice_queue_pop()
                return_images.append(image)
            except (IndexError):  # Empty queue
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
            if not isinstance(tag, pydicom.tag.Tag):
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

                file_meta = pydicom.dataset.Dataset()
                file_meta.MediaStorageSOPClassUID = "CT Image Storage"
                file_meta.MediaStorageSOPInstanceUID = "1.%d.%d.%d" \
                                                       % (channel_id,
                                                          slice_i,
                                                          uid)
                uid += 1
                file_meta.ImplementationClassUID = "2.3.7.6.1.2.%s" \
                                                   % (nethin.__version__,)

                data = pydicom.dataset.FileDataset(filename, {},
                                                   file_meta=file_meta,
                                                   preamble=b"\0" * 128)
                data.PatientName = patient_name
                data.PatientID = image_name

                data.is_little_endian = True
                data.is_implicit_VR = True

                dt = datetime.datetime.now()
                data.ContentDate = dt.strftime('%Y%m%d')
                # Long format with micro seconds:
                data.ContentTime = dt.strftime('%H%M%S.%f')

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
