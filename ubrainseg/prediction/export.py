from functools import partial
from typing import List, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch

from ..imageio import save_prob, save_seg
from ..utils import argmax0, softmax0, timerComments
from ..preprocessing.cropping import bounding_box_to_slice, revert_cropping
from ..preprocessing.resampling import resample_data_or_seg_to_shape


@timerComments("POSTPROCESSING")
def export_prediction_from_softmax(
    predicted_array_or_file: np.ndarray,
    properties_dict: dict,
    plans: dict,
    save_probabilities: bool = False,
    faster: bool = False,
    device: torch.device = torch.device('cpu'),
) -> Union[
        sitk.Image,  # seg
        Tuple[sitk.Image, List[sitk.Image]]  # seg + [probs]
]:

    predicted_array_or_file = torch.from_numpy(predicted_array_or_file).to(torch.float32).numpy()

    # resample to original shape
    current_spacing = plans.config.spacing if \
        len(plans.config.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *plans.config.spacing]

    resample_seg = partial(resample_data_or_seg_to_shape, **plans.config['resampling_fn_probabilities_kwargs'])
    predicted_array_or_file = resample_seg(predicted_array_or_file,
                                           properties_dict['shape_after_cropping_and_before_resampling'],
                                           current_spacing,
                                           properties_dict['spacing'],
                                           faster=faster,
                                           device=device)

    probs = softmax0(predicted_array_or_file)
    segmentation = argmax0(probs)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans.transpose_backward)
    seg = save_seg(segmentation_reverted_cropping, properties_dict)

    # save
    if save_probabilities:

        # revert cropping
        probs_reverted_cropping = revert_cropping(probs,
                                                  properties_dict['bbox_used_for_cropping'],
                                                  properties_dict['shape_before_cropping'])
        # revert transpose
        probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                           plans.transpose_backward])
        probs = save_prob(probs_reverted_cropping, properties_dict)

    return seg if not save_probabilities else (seg, probs)
