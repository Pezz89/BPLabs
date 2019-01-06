#! /usr/bin/env python

import numpy as np
from scipy.fftpack import fft, ifft

import pdb

def _acorr_last_axis(x, nfft, maxlag):
    a = np.real(ifft(np.abs(fft(x, n=nfft) ** 2)))
    return a[..., :maxlag+1] / x.shape[-1]


def acorr(blk, max_lag=None):
    """
    This code was adapted from the audio_lazy project:
        https://pythonhosted.org/audiolazy/_modules/audiolazy/lazy_analysis.html#acorr
    """
    if max_lag is None:
        max_lag = blk.size


    return [sum(blk[n] * blk[n + tau] for n in range(len(blk) - tau)) for tau in range(max_lag + 1)]

def nextpow2(n):
    """Return the next power of 2 such as 2^p >= n.
    Notes
    -----
    Infinite and nan are left untouched, negative values are not allowed."""
    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res


def _acorr2(x):
    '''
    '''

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    maxlag = x.size
    nfft = 2 ** nextpow2(2 * maxlag - 1)
    y = _acorr_last_axis(x, nfft, maxlag)
    pdb.set_trace()
