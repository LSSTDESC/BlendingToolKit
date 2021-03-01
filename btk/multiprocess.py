"""Tools for multiprocessing in BTK."""
import multiprocessing as mp
from itertools import starmap


def multiprocess(func, input_args, cpus, multiprocessing=False, verbose=False):
    if multiprocessing:
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
