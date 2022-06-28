"""Tools for multiprocessing in BTK."""
import multiprocessing as mp
from itertools import repeat, starmap


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def _pool_starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _starmap_with_kwargs(fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return starmap(_apply_args_and_kwargs, args_for_starmap)


def get_current_process():
    """Return ID of current process or 'main' if no multiprocessing."""
    if mp.current_process().name == "MainProcess":
        return "main"
    return mp.current_process().ident


def multiprocess(fn, args_iter, kwargs_iter=None, cpus=1, verbose=False):
    """Sole function that implements multiprocessing across mini-batches/batches for BTK.

    Args:
        fn (function): Function to run in parallel on each positional arguments returned by
            `args_iter` and each keyword arguments returned by `kwargs_iter`.
        args_iter (iter): Iterator returning positional arguments to be passed in to function for
            multiprocessing. This iterator must have a `__len__` method implemented. Each
            argument returned by the iterator must be unpackable like: `*args`.
        kwargs_iter (iter): Iterator returning keyword arguments to be passed in to
            function for multiprocessing. Default value `None` means that no keyword arguments
            are passed in. Each element returned by the iterator must be a `dict`.
        cpus (int): # of cpus to use for multiprocessing.
        verbose (bool): Whether to print information related to multiprocessing
    """
    kwargs_iter = repeat({}) if kwargs_iter is None else kwargs_iter
    if cpus > 1:
        if verbose:
            print(
                f"Running mini-batch of size {len(args_iter)} with multiprocessing with "
                f"pool {cpus}"
            )
        with mp.Pool(processes=cpus) as pool:
            results = _pool_starmap_with_kwargs(pool, fn, args_iter, kwargs_iter)
    else:
        if verbose:
            print(f"Running mini-batch of size {len(args_iter)} serial {cpus} times")
        results = list(_starmap_with_kwargs(fn, args_iter, kwargs_iter))
    return results
