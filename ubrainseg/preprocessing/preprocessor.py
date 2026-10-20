from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from loguru import logger

from ..imageio import get_data_and_properites, get_data_and_properites_faster
from ..utils import compute_new_shape, timer, timerComments
from .cropping import crop_to_nonzero
from .resampling import resample_data_or_seg_to_shape

#: fixed Hounsfield-unit brain window for CT inputs (SynthSeg convention)
CT_WINDOW = (0.0, 80.0)


class Preprocessor:
    def __init__(
        self,
        faster: bool = True
    ):
        self.faster = faster

    @timerComments(comments="PREPROCESSING")
    def exec(
        self,
        image_files: List[sitk.Image],
        plans: dict,
        foreground_arr: np.ndarray = None,
        ct: bool = False
    ) -> Tuple[np.ndarray, dict]:
        # order: transpose -> CT window -> crop -> normalize -> resample (reverted on export)
        data, data_properites = get_data_and_properites(image_files) if not self.faster \
                                    else get_data_and_properites_faster(image_files)

        data = data.transpose([0, *[i + 1 for i in plans.transpose_forward]])
        original_spacing = [data_properites['spacing'][i] for i in plans.transpose_forward]

        data_properites['shape_before_cropping'] = data.shape[1:]

        if ct:
            # air (-1000 HU) maps to 0 so the nonzero mask can exclude it;
            # bone/calification saturates at the upper bound
            data = np.clip(data, *CT_WINDOW)

        data, nonzero_mask, bbox = crop_to_nonzero(data, foreground_arr=foreground_arr)

        data_properites['bbox_used_for_cropping'] = bbox
        data_properites['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        data = normalize(data, nonzero_mask, plans, ct=ct)

        logger.info(f"shape_before_cropping: {data_properites['shape_before_cropping']}")
        logger.info(f"shape_after_cropping:  {data_properites['shape_after_cropping_and_before_resampling']}")
        logger.info(f"Cropped bbox: {data_properites['bbox_used_for_cropping']}")

        # resample
        target_spacing = plans.config.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing

        old_shape = data.shape[1:]
        new_shape = compute_new_shape(old_shape, original_spacing, target_spacing)

        logger.info(f'old_spacing: {[round(i, 3) for i in original_spacing]}')
        logger.info(f'new_spacing: {[round(i, 3) for i in target_spacing]}')
        logger.info(f'old_shape: {tuple(old_shape)} -> calc_new_shape: {tuple(new_shape)}')
        logger.info(f"resampled params: {plans.config.resampling_fn_data_kwargs}")

        data = resample_data_or_seg_to_shape(data, new_shape, original_spacing, target_spacing, faster=self.faster, **plans.config.resampling_fn_data_kwargs)

        logger.info(f'resampled finished. new shape: {data.shape}')

        return data, data_properites


@timer
def normalize(data: np.ndarray, nonzero_mask: np.ndarray, plans: dict, ct: bool = False) -> np.ndarray:

    def _z_score(image, *args, **kwargs):

        logger.info("Using ZScore Normalization...")

        if kwargs.get('use_nonzero_mask', False):
            logger.info("Using NonZero Mask...")
            mask = kwargs['nonzero_mask'] >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
            logger.info(f"mean={mean:.3f}, std={std:.3f}")

        return image

    def _ct_z_score(image, *args, **kwargs):
        logger.info("Using CT (self percentile) Normalization...")

        mask = kwargs['nonzero_mask'] >= 0
        lower_bound, upper_bound = np.percentile(image[mask], [0.5, 99.5])
        range_mask = mask & (image >= lower_bound) & (image <= upper_bound)
        mean = image[range_mask].mean()
        std = image[range_mask].std()
        image = (image - mean) / (max(std, 1e-8))
        logger.info(f"p0.5={lower_bound:.3f}, p99.5={upper_bound:.3f}")
        logger.info(f"mean={mean:.3f}, std={std:.3f}")

        return image

    def _no_norm(image, *args, **kwargs):
        logger.info("No Normalization...")
        return image

    _norm = {
        "ZScoreNormalization": _z_score,
    }

    for c in range(data.shape[0]):
        norm = _ct_z_score if ct else _norm.get(plans.config.normalization_schemes[c], _no_norm)
        data[c] = norm(
            data[c],
            intensity_properties=plans.foreground_intensity_properties_per_channel[str(c)],
            nonzero_mask=nonzero_mask,
            use_nonzero_mask=plans.config.use_mask_for_norm[c],
        )

    return data
