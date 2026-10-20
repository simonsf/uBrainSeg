from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from loguru import logger


def save_seg(seg: np.ndarray, properties: dict) -> sitk.Image:
    assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
    output_dimension = len(properties['sitk_stuff']['spacing'])
    assert 1 < output_dimension < 4
    if output_dimension == 2:
        seg = seg[0]
    itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
    itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
    itk_image.SetOrigin(properties['sitk_stuff']['origin'])
    itk_image.SetDirection(properties['sitk_stuff']['direction'])
    return itk_image


def save_prob(prob: np.ndarray, properties: dict) -> List[sitk.Image]:
    images = []
    channels = prob.shape[0]
    prob = prob.squeeze()
    for c in range(1, channels):
        logger.info(f"Saving probmap channel {c}...")
        c_prob = prob[c]
        im = sitk.GetImageFromArray(c_prob)
        im.SetSpacing(properties['sitk_stuff']['spacing'])
        im.SetOrigin(properties['sitk_stuff']['origin'])
        im.SetDirection(properties['sitk_stuff']['direction'])
        images.append(im)
    return images


def get_data_and_properites(ims: List[sitk.Image]) -> Tuple[np.ndarray, dict]:

    images = []
    spacings = []
    origins = []
    directions = []

    spacings_for_nnunet = []
    for f in ims:
        spacings.append(f.GetSpacing())
        origins.append(f.GetOrigin())
        directions.append(f.GetDirection())
        npy_image = sitk.GetArrayFromImage(f)

        npy_image = npy_image[None]
        # sitk reads (x,y,z) which corresponds to array axes (z,y,x), hence reversed
        spacings_for_nnunet.append(list(spacings[-1])[::-1])

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

    stacked_images = np.vstack(images)
    dict = {
        'sitk_stuff': {
            'spacing': spacings[0],
            'origin': origins[0],
            'direction': directions[0]
        },
        'spacing': spacings_for_nnunet[0]
    }

    return stacked_images.astype(np.float32), dict


def get_data_and_properites_faster(ims: List[sitk.Image]) -> Tuple[np.ndarray, dict]:

    images = []
    spacings = []
    origins = []
    directions = []

    spacings_for_nnunet = []

    for f in ims:

        spacings.append(f.GetSpacing())
        origins.append(f.GetOrigin())
        directions.append(f.GetDirection())
        npy_image = torch.from_numpy(sitk.GetArrayViewFromImage(f).astype(float))

        npy_image = npy_image[None]

        spacings_for_nnunet.append(list(spacings[-1])[::-1])

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

    stacked_images = torch.vstack(images)
    dict = {
        'sitk_stuff': {
            'spacing': spacings[0],
            'origin': origins[0],
            'direction': directions[0]
        },
        'spacing': spacings_for_nnunet[0]
    }

    return stacked_images.to(torch.float32).numpy(), dict
