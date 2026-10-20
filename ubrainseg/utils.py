import json
from functools import wraps
from time import time
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger
from skimage.transform import resize

ANISO_THRESHOLD = 3


def timer(func):

    def tictoc(*args, **kwargs):
        t = time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} cost : {time() - t:.3f} s")
        return result

    return tictoc


def timerComments(
    comments: Optional[str] = None,
):

    def decorator(func):

        @wraps(func)
        def tictoc(*args, **kwargs):

            echo = func.__name__ if comments is None else comments
            print_ = logger.debug if comments is None else logger.warning
            t = time()
            result = func(*args, **kwargs)
            print_(f"{echo} cost : {time() - t:.3f} s")

            return result

        return tictoc

    return decorator


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def compute_new_shape(old_shape: Union[tuple, list, np.ndarray],
                      old_spacing: Union[tuple, list, np.ndarray],
                      new_spacing: Union[tuple, list, np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def resize_segmentation(segmentation, new_shape, order=3):
    """Resize a segmentation map via one-hot encoding to avoid interpolation artifacts."""
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


@timer
def argmax0(arr: np.ndarray) -> np.ndarray:
    return np.argmax(arr, axis=0)


@timer
def softmax0(x: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(x), 0).numpy()


class Verbose:

    def __init__(
        self,
        verbose: bool = True
    ):
        self.verbose = verbose

    def __enter__(self):

        if not self.verbose:
            logger.disable('ubrainseg')
        else:
            logger.enable('ubrainseg')

        return self.verbose

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.verbose:
            logger.disable('ubrainseg')


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
