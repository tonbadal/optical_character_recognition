# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import string

import numpy as np
from PIL import Image

DEFAULT_SEED = 123456


class DataProvider(object):
    """Data provider."""

    def __init__(self, batch_size, max_num_batches=-1, which_set='train', prop=0.8,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            which_set (string): Whether the created object will be used to
                train or test the model.
            prop (float): THe proportion of data used for training, in the range
                (0, 1], set by default to 0.8.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        assert which_set in ['train', 'test'], (
            'Expected which_set to be either train or test. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set

        assert 0 < prop <= 1, (
            'Expected prop to be 0 < prop <= 1. '
            'Got {0}'.format(prop)
        )
        self.prop = prop

        self.inputs = []
        self.targets = []

        data_path = 'English/Fnt/'
        folders = os.listdir(data_path)
        self.num_classes = len(folders)
        characters = string.digits + string.ascii_uppercase + string.ascii_lowercase

        # Create ID - Character dictionary and One-Hot labels dictionary
        id2char = {}
        labels_dict = {}
        for i in range(len(characters)):
            # ID - Character
            id2char[i + 1] = characters[i]

            # One Hot encoding
            zeros = np.zeros(self.num_classes)
            zeros[i] += 1
            labels_dict[characters[i]] = zeros

            i += 1

        # Read images
        for folder in folders:
            char_id = int(folder[-2:])
            one_hot_label = labels_dict[id2char[char_id]]
            images = os.listdir(data_path + folder)

            # Load only a proportion of the DB
            len_images = len(images)
            if self.which_set == 'train':
                images = images[:int(len_images * self.prop)]
            elif self.which_set == 'test':
                images = images[int(len_images * self.prop):]

            for image_name in images:
                image_name = data_path + folder + '/' + image_name
                image = Image.open(image_name)
                image = image.resize((28, 28))
                pixels = 1 - np.array(image, dtype=np.float32) / 255
                pixels = pixels.reshape((784,))

                self.inputs.append(pixels)
                self.targets.append(one_hot_label)

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(self.inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()
