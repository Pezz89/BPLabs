import sys
sys.path.insert(0, "matrix_test/helper_modules")
import numpy as np
from signalops import rolling_window_lastaxis, calc_rms


def detect_silences(x, fs, threshold=-30.):
    print("Detecting silence in wav files...")
    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    x = x.sum(axis=1)/2.
    env = calc_rms(x, window=int(fs*0.1))
    threshold = (10**(threshold/20.))*np.max(env)
    silence = env < threshold
    # Get segment start end indexes for all silences in envelope
    sil_start = np.where(np.sign(np.diff(silence.astype(float))) == 1)[0]
    sil_end = np.where(np.sign(np.diff(silence.astype(float))) == -1)[0]
    if silence[0]:
        sil_start = np.concatenate([[0], sil_start])
    if silence[-1]:
        sil_end = np.concatenate([sil_end, [env.size]])
    segs = np.vstack([sil_start, sil_end]).T
    validSegs = np.diff(segs) > 0.02*fs
    segs = segs[np.repeat(validSegs, 2, axis=1)].reshape(-1, 2)
    return segs


def slices_to_mask(slices, mask_length):
    out = np.zeros(mask_length, dtype=bool)
    for s in slices:
        out[s[0]:s[1]] = True
    return out


def rms_no_silences(x, fs, threshold):
    silences = detect_silences(x, fs, threshold)
    sil_mask = slices_to_mask(silences, x.size)
    rms = np.sqrt(np.mean(x[~sil_mask]**2))
    return rms
