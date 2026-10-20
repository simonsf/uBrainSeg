import itertools
from typing import List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from ..utils import dummy_context, timer, timerComments


@timer
def pad_nd_image(image: Union[torch.Tensor, np.ndarray], new_shape: Tuple[int, ...] = None,
                 mode: str = "constant", kwargs: dict = None, return_slicer: bool = False) -> \
        Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]]:

    old_shape = np.array(image.shape)

    if len(new_shape) < len(image.shape):  # keep the leading (channel, z) axes of the input
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        logger.info(f"Padding: {tuple(image.shape)} -> {tuple(res.shape)}")
        return res, slicer


@timer
def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16) \
        -> torch.tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).half()
    mn = gaussian_importance_map.min()
    if mn == 0:
        gaussian_importance_map.clip_(min=mn)

    return gaussian_importance_map


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)

        logger.info(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, im_shape: {tuple(image_size)}, tile_size: {tuple(tile_size)}')
        logger.info(f"tile_step_size {tile_step_size}, steps:{steps}")

        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)

        logger.info(f'n_steps {np.prod([len(i) for i in steps])}, image shape: {tuple(image_size)}, tile_size: {tuple(tile_size)}')
        logger.info(f"tile_step_size {tile_step_size}, steps:{steps}")

        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer


@timer
def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, text: dict, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:

    prediction = network(x, text)
    if mirror_axes is not None:

        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        axes_combinations = [
            c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
        ]
        for axes in axes_combinations:
            prediction += torch.flip(network(torch.flip(x, (*axes,)), text), (*axes,))
        prediction /= num_predictons

    return prediction.to('cpu') if isinstance(prediction, torch.Tensor) else prediction


@timerComments("INFERENCE")
def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         input_text: dict,
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         device: torch.device = torch.device('cuda')) -> Union[np.ndarray, torch.Tensor]:

    with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():

        assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

        logger.info(f"step_size: {tile_step_size}")
        logger.info(f"mirror_axes: {mirror_axes}")

        # pad if the image is smaller than the tile size
        data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True)

        if use_gaussian:
            gaussian = compute_gaussian(tile_size, sigma_scale=1. / 8)

        slicers = list(get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size))

        predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half)

        for slicer in slicers:
            workon = data[slicer][None]
            workon = workon.to(device, non_blocking=False)

            prediction = maybe_mirror_and_predict(network, workon, input_text, mirror_axes)[0]  # 0: batch=1

            predicted_logits[slicer] += (prediction * gaussian if use_gaussian else prediction)
            n_predictions[slicer[1:]] += (gaussian if use_gaussian else 1)

        predicted_logits /= n_predictions

    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
