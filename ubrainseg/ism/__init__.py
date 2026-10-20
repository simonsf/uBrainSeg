"""uBrainSeg Image Synthesis Module (ISM): label-driven synthetic image generation."""

from .augmentation import (SpatialDeformation, RandomCrop, RandomFlip,
                           BiasField, IntensityAugmentation, GaussianBlur,
                           SimulateAcquisition)
from .generation import (SynthesisConfig, ImageSynthesisModule, ISM,
                         synthesize, build_gmm_params, sample_gmm_image,
                         sample_random_resolution, draw_value_from_distribution,
                         GENERATION_LABELS, LABEL_NAMES)
from .imageio import load_mask, load_image, save_volume, mask_voxel_spacing

__all__ = [
    'ImageSynthesisModule', 'ISM', 'SynthesisConfig',
    'synthesize', 'build_gmm_params', 'sample_gmm_image',
    'sample_random_resolution', 'draw_value_from_distribution',
    'GENERATION_LABELS', 'LABEL_NAMES',
    'SpatialDeformation', 'RandomCrop', 'RandomFlip', 'BiasField',
    'IntensityAugmentation', 'GaussianBlur', 'SimulateAcquisition',
    'load_mask', 'load_image', 'save_volume', 'mask_voxel_spacing',
]
