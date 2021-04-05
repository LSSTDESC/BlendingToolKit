"""Tools for multiprocessing in BTK."""
import multiprocessing as mp
from itertools import starmap


def multiprocess(func, input_args, cpus, verbose=False):
    """Sole Function that implements multiprocessing across mini-batches for BTK."""
    if cpus > 1:
        if verbose:
            print(
                f"Running mini-batch of size {len(input_args)} with multiprocessing with "
                f"pool {cpus}"
            )
        with mp.Pool(processes=cpus) as pool:
            results = pool.starmap(func, input_args)
    else:
        if verbose:
            print(f"Running mini-batch of size {len(input_args)} serial {cpus} times")
        results = list(starmap(func, input_args))
    return results
