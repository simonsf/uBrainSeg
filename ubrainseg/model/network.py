import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import (convert_dim_to_conv_op,
                                                                  get_matching_instancenorm)


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder plus a trained projection head that
    outputs the FiLM parameters (gamma, beta) for the affine modulation."""

    def __init__(self, clip_ckpt, bottleneck_channels=320) -> None:
        super().__init__()
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, bottleneck_channels * 2),
        )
        self.bottleneck_channels = bottleneck_channels

        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def text2tokens(self, text_class):
        age_list = text_class.get('age')
        gender_list = text_class.get('gender')
        modality_list = text_class.get('modality')

        text_list = []

        if not isinstance(age_list, list):
            age_list = [age_list]
            gender_list = [gender_list]
            modality_list = [modality_list]

        for i in range(len(age_list)):
            modality = modality_list[i]
            gender = gender_list[i]
            age = age_list[i]

            if modality is None or gender is None or age is None:
                text_list.append(None)
                continue
            text = 'This is a brain {} image from a {} patient at {} old.'.format(modality, gender, age)
            text_list.append(text)

        if all(t is None for t in text_list):
            return None
        valid_texts = [t for t in text_list if t is not None]
        valid_idx = [i for i, t in enumerate(text_list) if t is not None]
        tokens = self.tokenizer(valid_texts, padding=True, return_tensors="pt")

        device = next(self.clip_text_model.parameters()).device
        tokens = {key: tokens[key].to(device) for key in tokens.keys()}
        return tokens, valid_idx

    def forward(self, text):
        if text is None:
            return None
        out = self.text2tokens(text)
        if out is None:
            return None
        tokens, valid_idx = out
        with torch.no_grad():
            clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output  # (B_valid, 512)
        params = self.projection(text_embedding)  # (B_valid, 2*C)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return gamma, beta, valid_idx


class UBrain(nn.Module):

    def __init__(self,
                 image_encoder,
                 clip_ckpt,
                 mask_decoder,
                 ) -> None:
        super().__init__()
        self.encoder = image_encoder
        self.text_encoder = TextEncoder(clip_ckpt)
        self.decoder = mask_decoder

    def forward(self, image, text, **kwargs):

        image_embeddings = self.encoder(image)

        fusion_embedding = image_embeddings[-1]

        # channel-wise affine modulation (FiLM); bypassed cases keep identity
        gamma = torch.ones(1, device=image.device, dtype=fusion_embedding.dtype)
        beta = torch.zeros(1, device=image.device, dtype=fusion_embedding.dtype)
        if text is not None:
            text_out = self.text_encoder(text)  # (gamma, beta, valid_idx) or None
            if text_out is not None:
                gamma_t, beta_t, valid_idx = text_out
                B = fusion_embedding.shape[0]
                if len(valid_idx) == B:
                    gamma = gamma_t.to(fusion_embedding.dtype).view(B, -1, 1, 1, 1)
                    beta = beta_t.to(fusion_embedding.dtype).view(B, -1, 1, 1, 1)
                else:
                    gamma_full = torch.ones(B, fusion_embedding.shape[1],
                                            1, 1, 1, device=fusion_embedding.device,
                                            dtype=fusion_embedding.dtype)
                    beta_full = torch.zeros_like(gamma_full)
                    gamma_full[valid_idx] = gamma_t.to(fusion_embedding.dtype).view(len(valid_idx), -1, 1, 1, 1)
                    beta_full[valid_idx] = beta_t.to(fusion_embedding.dtype).view(len(valid_idx), -1, 1, 1, 1)
                    gamma, beta = gamma_full, beta_full

        fusion_embedding = gamma * fusion_embedding + beta
        image_embeddings[-1] = fusion_embedding

        logits = self.decoder(image_embeddings)

        return logits


def build_network(plans, clip_ckpt):
    """Build the network described by the deploy plans."""
    cfg = plans.config
    dim = len(cfg.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    num_stages = len(cfg.conv_kernel_sizes)
    features_per_stage = [min(cfg.UNet_base_num_features * (2 ** i), cfg.unet_max_num_features)
                          for i in range(num_stages)]

    unet = PlainConvUNet(
        input_channels=len(plans.channel_names),
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=cfg.conv_kernel_sizes,
        strides=cfg.pool_op_kernel_sizes,
        n_conv_per_stage=cfg.n_conv_per_stage_encoder,
        num_classes=plans.output_channels,
        n_conv_per_stage_decoder=cfg.n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(conv_op),
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
    )
    return UBrain(unet.encoder, clip_ckpt, unet.decoder)
