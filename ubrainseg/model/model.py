from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from easydict import EasyDict as edict
from loguru import logger

from ..prediction.export import export_prediction_from_softmax
from ..prediction.predict import predict_sliding_window_return_logits
from ..preprocessing.preprocessor import Preprocessor
from ..utils import Verbose, load_json, timerComments
from .network import build_network

nnUNetImage = Union[sitk.Image, List[sitk.Image]]

nnUNetImageResults = Union[
    sitk.Image,              # segmentation
    Tuple[                   # segmentation with probabilities
        sitk.Image,
        Union[
            sitk.Image,          # 1 channel
            Tuple[sitk.Image]    # multi channel
        ]
    ]
]


class UBrainSegModel:
    """UBrainSeg inference model."""

    def __call__(
        self,
        im: nnUNetImage,
        text: Optional[dict] = None,
        verbose: bool = True,
        tta: bool = True,
        use_gaussian: bool = True,
        tile_step_size: float = 0.5,
        save_probabilities: bool = False,
        faster: bool = False,
        foreground_arr: Optional[np.ndarray] = None,
        grid_sample_on_gpu: bool = False,
        ct: bool = False,
    ):
        seg = self.infer(
            im=im,
            text=text,
            verbose=verbose,
            tta=tta,
            use_gaussian=use_gaussian,
            tile_step_size=tile_step_size,
            save_probabilities=save_probabilities,
            faster=faster,
            foreground_arr=foreground_arr,
            grid_sample_on_gpu=grid_sample_on_gpu,
            ct=ct,
        )
        return seg

    @timerComments("LOADMODEL")
    def load(
        self,
        model_pth: str,
        plans_pth: str,
        gpu_id: int = 0,
        verbose: bool = False,
        clip_ckpt: str = 'openai/clip-vit-base-patch32',
    ):

        with Verbose(verbose):

            for k, v in locals().items():
                if k != "self":
                    logger.info(f"{k}: {v}")

            if gpu_id is None or not torch.cuda.is_available():
                self.device = torch.device('cpu')
                if gpu_id is not None:
                    logger.warning("CUDA not available, running on CPU.")
            else:
                self.device = torch.device('cuda', index=gpu_id)

            self.plans = edict(load_json(plans_pth))

            self.network = build_network(self.plans, clip_ckpt)
            state_dict = torch.load(model_pth, map_location='cpu')
            missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
            # the frozen CLIP backbone is loaded from the public checkpoint
            missing = [k for k in missing if not k.startswith('text_encoder.clip_text_model.')]
            assert not missing, f"missing weights: {missing}"
            assert not unexpected, f"unexpected keys: {unexpected}"
            self.network = self.network.to(self.device)
            self.network.eval()
            logger.info("Load model OK")

        return self

    @timerComments("INFERFLOW")
    def infer(
        self,
        im: nnUNetImage,
        text: Optional[dict] = None,
        verbose: bool = True,
        tta: bool = True,
        use_gaussian: bool = True,
        tile_step_size: float = 0.5,
        save_probabilities: bool = False,
        faster: bool = False,
        foreground_arr: Optional[np.ndarray] = None,
        grid_sample_on_gpu: bool = False,
        ct: bool = False,
    ) -> nnUNetImageResults:

        with Verbose(verbose):

            for k, v in locals().items():
                if k != "im" and k != "self" and k != "foreground_arr":
                    logger.info(f"{k}: {v}")
                if k == "foreground_arr":
                    logger.info(f"{k}: " + (f"None" if v is None else f"{v.shape}"))

            if not isinstance(im, list):
                im = [im]
            for im_ in im:
                assert isinstance(im_, sitk.Image), \
                    'im must be a SimpleITK.Image or a list of SimpleITK.Image'

            data, properties = Preprocessor(faster=faster).exec(im, self.plans, foreground_arr=foreground_arr, ct=ct)

            with torch.no_grad():

                data = torch.Tensor(data)
                prediction = predict_sliding_window_return_logits(
                    self.network, data, text, self.plans.output_channels,
                    self.plans.config.patch_size,
                    mirror_axes=self.plans.mirror_axes if tta else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    device=self.device)

                prediction = prediction.to('cpu').numpy()
                seg = export_prediction_from_softmax(
                    prediction, properties, self.plans, save_probabilities,
                    faster=faster,
                    device=self.device if grid_sample_on_gpu else torch.device("cpu")
                )

            if not isinstance(seg, tuple):
                return seg
            else:
                return (
                    seg[0],
                    seg[1][0] if len(seg[1]) == 1 else tuple(seg[1])
                )
