"""
Metrics comparing predicted and recorded firing rates
"""

from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import sklearn
from scipy.stats import pearsonr,zscore
from functools import wraps
from tqdm import tqdm
from contextlib import contextmanager
from itertools import combinations, repeat
from numbers import Number

def multicell(metric):
    """Decorator for turning a function that takes two 1-D numpy arrays, and
    makes it work for when you have a list of 1-D arrays or a 2-D array, where
    the metric is applied to each item in the list or each matrix row.
    """
    @wraps(metric)
    def multicell_wrapper(r, rhat, **kwargs):

        # ensure that the arguments have the right shape / dimensions
        for arg in (r, rhat):
            assert isinstance(arg, (np.ndarray, list, tuple)), \
                "Arguments must be a numpy array or list of numpy arrays"

        # convert arguments to matrices
        true_rates = np.atleast_2d(r)
        model_rates = np.atleast_2d(rhat)

        assert true_rates.ndim == 2, "Arguments have too many dimensions"
        assert true_rates.shape == model_rates.shape, "Shapes must be equal"

        # compute scores for each pair
        scores = [metric(true_rate, model_rate)
                  for true_rate, model_rate in zip(true_rates, model_rates)]

        # return the mean across cells and the full list
        return np.nanmean(scores), scores

    return multicell_wrapper


@multicell
def cc(x, y):
    """Pearson's correlation coefficient

    If r, rhat are matrices, cc() computes the average pearsonr correlation
    of each column vector
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    epsilon = 0.01
    def ss(a):
        return np.sum(a*a)
    r_den = np.sqrt(ss(xm+epsilon) * ss(ym+epsilon))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    return r


@multicell
def lli(r, rhat):
    """Log-likelihood (arbitrary units)"""
    epsilon = 1e-9
    return np.mean(r * np.log(rhat + epsilon) - rhat)


@multicell
def rmse(r, rhat):
    """Root mean squared error"""
    return np.sqrt(np.mean((rhat - r) ** 2))


@multicell
def fev(r, rhat):
    """Fraction of explained variance

    https://wikipedia.org/en/Fraction_of_variance_unexplained
    """
    return 1.0 - rmse(r, rhat)[0]**2 / r.var()

@contextmanager
def notify(title):
    """Context manager for printing messages of the form 'Loading... Done.'

    Parameters
    ----------
    title : string
        A message / title to print

    Usage
    -----
    >>> with notify('Loading'):
    >>>    # do long running task
    >>>    time.sleep(0.5)
    >>> Loading... Done.

    """

    print(title + '... ', end='')
    sys.stdout.flush()
    try:
        yield
    finally:
        print('Done.')

def allmetrics(r, rhat, functions):
    """Evaluates the given responses on all of the given metrics

    Parameters
    ----------
    r : array_like
        True response, with shape (# of samples, # of cells)

    rhat : array_like
        Model response, with shape (# of samples, # of cells)

    functions : list of strings
        Which functions from the metrics module to evaluate on
    """
    avg_scores = {}
    all_scores = {}
    for function in functions:
        avg, cells = eval(function + "(r.T, rhat.T)")
        avg_scores[function] = avg
        all_scores[function] = cells

    return avg_scores, all_scores

