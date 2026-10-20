"""Command line interface for UBrainSeg inference."""
import argparse
import os

import SimpleITK as sitk

from ubrainseg import UBrainSegModel


def parse_args():
    parser = argparse.ArgumentParser(description='UBrainSeg brain tissue segmentation')
    parser.add_argument('--image', required=True, help='input MR image (.nii.gz / .nii)')
    parser.add_argument('--output', required=True, help='output segmentation file path')
    parser.add_argument('--weights_dir', default='weights',
                        help='directory containing ubrainseg_state_dict.pth and deploy_plans.json')
    parser.add_argument('--clip', default='openai/clip-vit-base-patch32',
                        help="CLIP text checkpoint: local HuggingFace checkpoint directory, "
                             "or a model id that will be downloaded from the HuggingFace hub")
    parser.add_argument('--age', default=None, help='patient age, e.g. "54 years"')
    parser.add_argument('--gender', default=None, help='patient gender, e.g. male / female')
    parser.add_argument('--modality', default=None,
                        help='image modality: T1, T2, FLAIR, DWI, SWI or Phase')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU')
    parser.add_argument('--ct', action='store_true',
                        help='treat the input as a CT scan: clip Hounsfield units to the brain '
                             'window [0, 80] and use per-image percentile normalization')
    parser.add_argument('--no_tta', action='store_true', help='disable test-time augmentation (mirroring)')
    parser.add_argument('--save_probabilities', action='store_true', help='also save class probability maps')
    parser.add_argument('--fast', action='store_true', help='faster GPU preprocessing / export')
    args = parser.parse_args()
    return args


def build_text(args):
    if args.age is None and args.gender is None and args.modality is None:
        return None
    return {'age': args.age, 'gender': args.gender, 'modality': args.modality}


def main():
    args = parse_args()

    model = UBrainSegModel()
    model.load(
        model_pth=os.path.join(args.weights_dir, 'ubrainseg_state_dict.pth'),
        plans_pth=os.path.join(args.weights_dir, 'deploy_plans.json'),
        gpu_id=None if args.gpu < 0 else args.gpu,
        clip_ckpt=args.clip,
    )

    im = sitk.ReadImage(args.image)
    text = build_text(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    result = model(im, text, tta=not args.no_tta, faster=args.fast,
                   save_probabilities=args.save_probabilities, ct=args.ct)

    if isinstance(result, tuple):
        seg, probs = result
        # probs is one image per foreground class (CSF, GM, WM)
        prob_list = probs if isinstance(probs, (list, tuple)) else [probs]
        base = args.output[:-7] if args.output.endswith('.nii.gz') else os.path.splitext(args.output)[0]
        for i, p in enumerate(prob_list):
            prob_path = f'{base}_prob{i + 1}.nii.gz'
            sitk.WriteImage(p, prob_path)
            print(f'probability map {i + 1} saved to {prob_path}')
    else:
        seg = result

    sitk.WriteImage(seg, args.output)
    print(f'segmentation saved to {args.output} (labels: background=0, CSF=1, GM=2, WM=3)')


if __name__ == '__main__':
    main()
