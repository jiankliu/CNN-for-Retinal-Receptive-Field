"""
Preprocessing utility functions for loading and formatting experimental data
"""

from __future__ import absolute_import, division, print_function
import os
from functools import partial
from itertools import repeat
from collections import namedtuple
import numpy as np
import h5py
from scipy.stats import zscore
from .metrics import notify, allmetrics
Exptdata = namedtuple('Exptdata', ['X', 'y'])
dt = 1e-2
__all__ = ['Experiment', 'loadexpt']


class DataSet(object):
    """Class to keep track of loaded experiment data"""

    def __init__(self, rootpath, h5path, history, batchsize, holdout=0.0, cfg=None):
        """Keeps track of experimental data
           Whether stimulus should be zscored (default: True)
        """

        assert holdout >= 0 and holdout < 1, "holdout must be between 0 and 1"
        self.rootpath = rootpath
        self.h5path = h5path
        self.batchsize = batchsize
        self.dt = dt
        self.holdout = holdout

        # load training data, and generate the train/validation split, for each filename
        self._train_data = {}
        self._train_batches = list()
        self._validation_batches = list()

        noise = False
        if not cfg == None:
            if cfg['noise']:
                noise = True

        # load the training experiment as an Exptdata tuple
        filename = self.h5path + 'train.hdf5'
        self._train_data = loadexpt(filename, history, cfg=cfg)

        # generate the train/validation split
        length = self._train_data.X.shape[0]
        train, val = _train_val_split(length, self.batchsize, holdout)

        # append these train/validation batches to the master list
        self._train_batches.extend(train)
        self._validation_batches.extend(val)

        # load the data for each experiment, store as a list of Exptdata tuple
        filename = self.h5path + 'test.hdf5'
        self._test_data = loadexpt(filename, history)
        self._validate_data = self._test_data

        # save batches_per_epoch for calculating # epochs later
        self.batches_per_epoch = len(self._train_batches)

    def train(self, shuffle=True):
        """Returns a generator that yields batches of *training* data

        Parameters
        ----------
        shuffle : boolean
            Whether or not to shuffle the time points before making batches
        """
        # generate an order in which to go through the batches
        indices = np.arange(len(self._train_batches))
        if shuffle:
            np.random.shuffle(indices)

        # yield training data, one batch at a time
        for ix in indices:
            inds = self._train_batches[ix]
            yield self._train_data.X[inds], self._train_data.y[inds]

    def validate(self):
        """Evaluates the model on the validation set

        Parameters
        ----------
        modelrate : function
            A function that takes a spatiotemporal stimulus and predicts a firing rate
        """
        # choose a random validation batch

        #expt, inds = self._train_batches[np.random.randint(len(self._validation_batches))]

        # load the stimulus and response on this batch
        #X = self._train_data[expt].X[inds]
        #y = self._train_data[expt].y[inds]

        val_len = 300
        X = self._test_data.X[:val_len]
        y = self._test_data.y[:val_len]

        return X, y

    def test(self, test_type):
        """Tests model predictions on the repeat stimuli

        Parameters
        ----------
        modelrate : function
            A function that takes a spatiotemporal stimulus and predicts a firing rate
        """
        X = self._test_data[test_type].X
        y = self._test_data[test_type].y
        #print ('X.shape')
        #print (X.shape)
        
        return X, y

def loadexpt(filename, history, cfg=None):
    """
    Loads an experiment from an h5 file on disk
    """

    assert history > 0 and type(history) is int, "Temporal history must be a positive integer"

    with notify('Loading data:{}'.format(filename)):
        # load the hdf5 file
        percent_ratio = 1
        if not cfg == None:
            if cfg['percent']:
                percent_ratio = cfg['percent_ratio']

        print('\npercent_ratio: ' + str(percent_ratio))

        with h5py.File(filename, mode='r') as f:
            r = np.array(f['r'])
            ex_len = int(percent_ratio * len(r))
            X = np.array(f['X'])[:ex_len]
            y = np.array(f['y'])[:ex_len]
            spk_num = np.sum(y>0)
            print('\nSpike number: ' + str(spk_num))
            r = r[:ex_len]
            if history > 1:
                X_roll = rolling_window(X, history, time_axis=0)
                r = r[history:]
            else:
                X_roll = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

            if not cfg == None:
                if cfg['noise']:
                    X_roll = stim_noise(X_roll, cfg['noise_ratio'])

    return Exptdata(X_roll, r)


def rolling_window(array, window, time_axis=0):
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr


def _train_val_split(length, batchsize, holdout):

    # compute the number of available batches, given the fixed batch size
    num_batches = int(np.floor(length / batchsize))

    # the total number of samples for training
    total = int(num_batches * batchsize)

    # generate batch indices, and shuffle the deck of batches
    batch_indices = np.arange(total).reshape(num_batches, batchsize)
    np.random.shuffle(batch_indices)

    # compute the held out (validation) batches
    num_holdout = int(np.round(holdout * num_batches))

    return batch_indices[num_holdout:].copy(), batch_indices[:num_holdout].copy()


def stim_noise(X, ratio):

    X_flat = X.reshape(X.shape[0],-1)
    for i in range(len(X_flat)):
        pixels = list(np.arange(len(X_flat[i])))
        import random
        pixels_select = random.sample(pixels, int(ratio*len(X_flat[i])))
        import copy
        pixels_shuffle = copy.deepcopy(pixels_select)
        random.shuffle(pixels_shuffle)
        X_flat[i][pixels_select] = X_flat[i][pixels_shuffle]
    return X_flat.reshape(X.shape)
