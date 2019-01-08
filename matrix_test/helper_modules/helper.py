import multiprocessing
import numpy as np

def multiprocess_map(func, args, jobs=-1):
    if jobs == -1:
        jobs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(func, args)
    return results


def slice_to_bool(slices, size):
    """
    Convert 2D array of numpy slices indexes to a boolean array of the size
    specified
    """
    if (slices > size).any():
        raise ValueError("Slice index exists that is larger than size")
    sil = np.zeros(size, dtype=bool)
    for sl in slices:
        sil[sl[0]:sl[1]] = True
    return sil
