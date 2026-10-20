from typing import List, Tuple, Union

import numpy as np
import torch


def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def create_nonzero_mask(data):
    """Return a mask that is True where the data is nonzero."""
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def create_nonzero_mask_with_foreground_arr(
    data: np.ndarray,
    foreground_arr: np.ndarray
) -> np.ndarray:  # bool

    nonzero_mask = torch.zeros(data.shape[1:], dtype=torch.bool)

    for c in range(data.shape[0]):
        if c == 0:
            this_mask = torch.from_numpy(foreground_arr).to(torch.bool)
        else:
            this_mask = torch.from_numpy(data[c]).to(torch.bool)
        nonzero_mask = nonzero_mask | this_mask

    return nonzero_mask.numpy()


def get_bbox_from_mask_faster(nonzero_mask: np.ndarray) -> List[List[int]]:

    nonzero_tensor = torch.from_numpy(nonzero_mask)
    nonzero = torch.nonzero(nonzero_tensor, as_tuple=True)

    min_x, min_y, min_z = (torch.min(nonzero[i]).item() for i in (0, 1, 2))
    max_x, max_y, max_z = (torch.max(nonzero[i]).item() + 1 for i in (0, 1, 2))

    bbox = [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

    return bbox


def crop_to_nonzero(data, nonzero_label=-1, foreground_arr=None):
    if foreground_arr is None:
        nonzero_mask = create_nonzero_mask(data)
        bbox = get_bbox_from_mask(nonzero_mask)
    else:
        nonzero_mask = create_nonzero_mask_with_foreground_arr(data, foreground_arr)
        bbox = get_bbox_from_mask_faster(nonzero_mask)

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    nonzero_mask = nonzero_mask[slicer][None]
    nonzero_mask = nonzero_mask.astype(np.int8)
    nonzero_mask[nonzero_mask == 0] = nonzero_label
    nonzero_mask[nonzero_mask > 0] = 0
    nonzero_mask = nonzero_mask[0]

    return data, nonzero_mask, bbox


def revert_cropping(predicted_probabilities: np.ndarray,
                    bbox: List[List[int]],
                    original_shape: Union[List[int], Tuple[int, ...]]):
    # revert cropping
    probs_reverted_cropping = np.zeros((predicted_probabilities.shape[0], *original_shape),
                                        dtype=predicted_probabilities.dtype)
    slicer = bounding_box_to_slice(bbox)
    probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_probabilities
    return probs_reverted_cropping
