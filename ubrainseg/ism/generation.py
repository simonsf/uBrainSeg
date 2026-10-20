"""Label-driven synthetic image generation: the Image Synthesis Module (ISM).

Pipeline: deform labels (Φgeo) -> GMM intensity sampling (G) -> image-level
augmentation (Φimg).
"""

import numpy as np

from .augmentation import (SpatialDeformation, RandomCrop, RandomFlip,
                           BiasField, IntensityAugmentation, GaussianBlur,
                           SimulateAcquisition)

#: Label convention of uBrainSeg: 0=background, 1=CSF, 2=GM, 3=WM.
GENERATION_LABELS = (0, 1, 2, 3)
LABEL_NAMES = ('background', 'CSF', 'GM', 'WM')


class SynthesisConfig:
    """Hyper-parameters of the ISM (defaults match the paper)."""

    def __init__(self,
                 # labels / resolution
                 generation_labels=GENERATION_LABELS,
                 atlas_res=1.0,
                 # Φgeo: geometric transformation
                 flipping=False,
                 output_shape=None,
                 scaling_bounds=0.1,
                 rotation_bounds=15.0,
                 shearing_bounds=0.02,
                 translation_bounds=False,
                 nonlin_std=2.0,
                 nonlin_scale=0.04,
                 # G: GMM priors
                 prior_means=None,           # None -> U(0, 255)
                 prior_stds=None,            # None -> U(0, 35)
                 prior_distributions='uniform',
                 # Φimg: appearance augmentation
                 bias_field_std=0.2,
                 bias_scale=0.025,
                 clip=False,
                 normalize=True,
                 gamma_std=0.5,
                 noise_std=0.0,
                 # Φimg: resolution variability
                 randomise_res=True,
                 max_res_iso=8.0,
                 max_res_aniso=8.0,
                 data_res=None):
        self.generation_labels = np.array(generation_labels, dtype=int)
        self.atlas_res = (np.ones(3) * float(atlas_res)
                          if np.ndim(atlas_res) == 0 else np.array(atlas_res, dtype=float))
        self.flipping = flipping
        self.output_shape = output_shape
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.prior_means = prior_means
        self.prior_stds = prior_stds
        self.prior_distributions = prior_distributions
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.clip = clip
        self.normalize = normalize
        self.gamma_std = gamma_std
        self.noise_std = noise_std
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.data_res = data_res


def draw_value_from_distribution(prior, distribution='uniform'):
    """Draw one value per column from a (2, K) prior: rows are [lo, hi]."""
    prior = np.asarray(prior, dtype=float)
    if prior.ndim == 1:
        prior = prior[np.newaxis, :]
    lo, hi = prior[0], prior[1]
    if distribution == 'uniform':
        return np.random.uniform(lo, hi, size=len(lo))
    return np.random.normal(lo, hi, size=len(lo))


def sample_gmm_image(label_map, generation_labels, means, stds):
    """Sample a synthetic image: voxels of each label are draws from N(mean_i, std_i).

    Background (label 0) is excluded from sampling and stays at intensity 0.
    """
    image = np.zeros(label_map.shape, dtype=np.float32)
    for i, lbl in enumerate(generation_labels):
        if lbl == 0:
            continue
        mask = label_map == lbl
        if mask.any():
            draws = np.random.normal(means[i], max(stds[i], 1e-6), mask.sum())
            image[mask] = np.maximum(draws, 0).astype(np.float32)
    return image


def build_gmm_params(generation_labels, prior_means=None, prior_stds=None,
                     prior_distributions='uniform'):
    """Sample per-label GMM means/stds from the (2, K) [lo, hi] priors."""
    generation_labels = np.array(generation_labels, dtype=int)
    K = len(generation_labels)

    pm = prior_means if prior_means is not None else np.array(
        [[0.0] * K, [255.0] * K], dtype=np.float32)
    ps = prior_stds if prior_stds is not None else np.array(
        [[0.0] * K, [35.0] * K], dtype=np.float32)
    pm = np.asarray(pm, dtype=float)
    ps = np.asarray(ps, dtype=float)
    if pm.shape[1] != K:
        raise ValueError('prior_means has %d columns, expected %d labels'
                         % (pm.shape[1], K))
    if ps.shape != pm.shape:
        raise ValueError('prior_stds shape %s does not match prior_means %s'
                         % (ps.shape, pm.shape))

    means = draw_value_from_distribution(pm, prior_distributions)
    stds = np.maximum(draw_value_from_distribution(ps, prior_distributions), 1e-6)
    return generation_labels, means, stds


def sample_random_resolution(atlas_res, max_res_iso=8.0, max_res_aniso=8.0):
    """Sample a random low acquisition resolution (isotropic or anisotropic)."""
    n_dims = len(atlas_res)
    res = np.array(atlas_res, dtype=float)

    modes = []
    if max_res_iso is not None:
        modes.append('iso')
    if max_res_aniso is not None:
        modes.append('aniso')
    if not modes:
        return res

    mode = np.random.choice(modes)
    if mode == 'iso':
        r = np.random.uniform(atlas_res.max(), max_res_iso)
        res = np.ones(n_dims) * r
    else:
        axis = np.random.randint(n_dims)
        mr = max_res_aniso if np.ndim(max_res_aniso) == 0 else max_res_aniso[axis]
        res[axis] = np.random.uniform(atlas_res[axis], mr)
    return res


def _compute_crop_shape(input_shape, cfg):
    if cfg.output_shape is None:
        return None
    out = np.array(cfg.output_shape)
    if out.ndim == 0:
        out = np.ones(3, dtype=int) * int(out)
    return list(out.astype(int))


def synthesize(label_map, means, stds, cfg=None):
    """Run the full deform -> sample -> augment pipeline; returns (image, labels)."""
    cfg = cfg if cfg is not None else SynthesisConfig()
    label_map = np.asarray(label_map)

    # ---- Φgeo: geometric transformation (deform first) ----
    deformer = SpatialDeformation(
        scaling_bounds=cfg.scaling_bounds,
        rotation_bounds=cfg.rotation_bounds,
        shearing_bounds=cfg.shearing_bounds,
        translation_bounds=cfg.translation_bounds,
        nonlin_std=cfg.nonlin_std,
        nonlin_scale=cfg.nonlin_scale,
        inter_method='nearest',   # labels: nearest-neighbour interpolation
    )
    label_map = deformer(label_map.astype(np.float32)).astype(np.int32)

    crop_shape = _compute_crop_shape(label_map.shape, cfg)
    if crop_shape is not None:
        label_map = RandomCrop(crop_shape)(label_map)

    if cfg.flipping:
        label_map = RandomFlip(prob=0.5)(label_map)

    # ---- G: GMM intensity sampling (sample second) ----
    image = sample_gmm_image(label_map, cfg.generation_labels, means, stds)

    # ---- Φimg: image-level augmentation (augment last) ----
    if cfg.bias_field_std > 0:
        image = BiasField(std=cfg.bias_field_std, scale=cfg.bias_scale)(image)

    image = IntensityAugmentation(clip=cfg.clip, normalize=cfg.normalize,
                                  gamma_std=cfg.gamma_std,
                                  noise_std=cfg.noise_std)(image)

    if cfg.data_res is not None:
        data_res = np.array(cfg.data_res, dtype=float)
        if data_res.ndim == 0:
            data_res = np.ones(3) * float(data_res)
        image = SimulateAcquisition(cfg.atlas_res, data_res)(image)
    elif cfg.randomise_res:
        acq_res = sample_random_resolution(cfg.atlas_res, cfg.max_res_iso,
                                           cfg.max_res_aniso)
        image = SimulateAcquisition(cfg.atlas_res, acq_res)(image)
    else:
        image = GaussianBlur([0.5] * 3)(image)

    return image.astype(np.float32), label_map.astype(np.int32)


class ImageSynthesisModule:
    """Public API of the ISM: label map in -> (image, label map) out.

    A fresh GMM is sampled for every call, so repeated calls on the same
    mask yield diverse contrasts.
    """

    def __init__(self, **kwargs):
        self.cfg = SynthesisConfig(**kwargs)

    def sample_gmm(self):
        """Draw per-label GMM means/stds from the configured priors."""
        _, means, stds = build_gmm_params(
            self.cfg.generation_labels,
            prior_means=self.cfg.prior_means,
            prior_stds=self.cfg.prior_stds,
            prior_distributions=self.cfg.prior_distributions)
        return means, stds

    def synthesize(self, label_map):
        means, stds = self.sample_gmm()
        return synthesize(label_map, means, stds, self.cfg)

    __call__ = synthesize


ISM = ImageSynthesisModule
