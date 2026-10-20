"""Command line interface for the uBrainSeg Image Synthesis Module (ISM)."""
import argparse
import os

import numpy as np

from ubrainseg.ism import ImageSynthesisModule, load_mask, save_volume


def parse_args():
    parser = argparse.ArgumentParser(
        description='uBrainSeg ISM: synthesize (image, label) pairs from tissue masks')
    parser.add_argument('--mask', required=True,
                        help='input tissue segmentation mask (.nii.gz / .nii), '
                             'labels: background=0, CSF=1, GM=2, WM=3')
    parser.add_argument('--output_dir', required=True,
                        help='directory for the synthetic outputs')
    parser.add_argument('--count', type=int, default=1,
                        help='number of synthetic volumes to generate (default 1)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # Φgeo
    parser.add_argument('--no_deformation', action='store_true',
                        help='disable the random geometric deformation')
    parser.add_argument('--scaling_bounds', type=float, default=0.1,
                        help='max scaling deviation (paper: 0.1 -> [0.9, 1.1])')
    parser.add_argument('--rotation_bounds', type=float, default=15.0,
                        help='max rotation in degrees (paper: 15)')
    parser.add_argument('--shearing_bounds', type=float, default=0.02,
                        help='max shearing coefficient (paper: 0.02)')
    parser.add_argument('--nonlin_std', type=float, default=2.0,
                        help='std of the nonlinear deformation field (paper: 2)')
    parser.add_argument('--flip', action='store_true',
                        help='enable random left-right flipping')

    # G
    parser.add_argument('--prior_means', default=None,
                        help='.npy file with (2, K) uniform bounds for GMM means '
                             '(default U(0, 255) per the paper)')
    parser.add_argument('--prior_stds', default=None,
                        help='.npy file with (2, K) uniform bounds for GMM stds '
                             '(default U(0, 35) per the paper)')

    # Φimg
    parser.add_argument('--bias_field_std', type=float, default=0.2,
                        help='std of the simulated bias field (paper: 0.2)')
    parser.add_argument('--gamma_std', type=float, default=0.5,
                        help='std of the Gamma transformation exponent')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std of additive Gaussian noise')
    parser.add_argument('--max_res', type=float, default=8.0,
                        help='maximum simulated acquisition resolution in mm (paper: 8)')
    parser.add_argument('--data_res', type=float, nargs='+', default=None,
                        help='fixed low resolution in mm instead of a random one, '
                             'e.g. --data_res 3 3 3')
    parser.add_argument('--no_res_sim', action='store_true',
                        help='disable resolution variability simulation')

    # misc
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                        help='random crop to this spatial shape, e.g. --output_shape 160 160 160')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    mask, ref = load_mask(args.mask)
    print(f'mask shape: {mask.shape}, labels: {np.unique(mask).tolist()}')

    ism = ImageSynthesisModule(
        scaling_bounds=0.0 if args.no_deformation else args.scaling_bounds,
        rotation_bounds=0.0 if args.no_deformation else args.rotation_bounds,
        shearing_bounds=0.0 if args.no_deformation else args.shearing_bounds,
        nonlin_std=0.0 if args.no_deformation else args.nonlin_std,
        flipping=args.flip,
        prior_means=np.load(args.prior_means) if args.prior_means else None,
        prior_stds=np.load(args.prior_stds) if args.prior_stds else None,
        bias_field_std=args.bias_field_std,
        gamma_std=args.gamma_std,
        noise_std=args.noise_std,
        max_res_iso=None if args.no_res_sim else args.max_res,
        max_res_aniso=None if args.no_res_sim else args.max_res,
        data_res=args.data_res,
        output_shape=args.output_shape,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.count):
        image, labels = ism(mask)
        img_path = os.path.join(args.output_dir, f'synth_{i:04d}_image.nii.gz')
        lbl_path = os.path.join(args.output_dir, f'synth_{i:04d}_label.nii.gz')
        save_volume(image, ref, img_path)
        save_volume(labels, ref, lbl_path)
        print(f'[{i + 1}/{args.count}] image -> {img_path} '
              f'(range [{image.min():.3f}, {image.max():.3f}]), '
              f'labels -> {lbl_path} (labels: {np.unique(labels).tolist()})')

    print(f'done. {args.count} synthetic pair(s) written to {args.output_dir}')


if __name__ == '__main__':
    main()
