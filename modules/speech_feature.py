"""
Wrapper Module for Speech Initial Features through 1D-Convolutions
Modified from HuggingFace Transformers
https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
"""

import torch.nn as nn
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeatureEncoder,
    Wav2Vec2ConformerFeatureProjection,
)


class Wav2Vec2ConfigStyle:
    def __init__(self, config):
        config = config["speech_feature"]
        self.num_feat_extract_layers = config["num_feat_extract_layers"]
        self.conv_bias = config["conv_bias"]
        self.conv_dim = config["conv_dim"]
        self.conv_kernel = config["conv_kernel"]
        self.conv_stride = config["conv_stride"]
        self.feat_extract_activation = config["feat_extract_activation"]
        self.feat_extract_norm = config["feat_extract_norm"]
        self.position_embeddings_type = config["position_embeddings_type"]
        self.hidden_size = config["output_dim"]
        self.layer_norm_eps = 1e-05
        self.feat_proj_dropout = 0.0


class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super(SpeechEncoder, self).__init__()

        self.config = Wav2Vec2ConfigStyle(config)
        self.speech_encoder = Wav2Vec2ConformerFeatureEncoder(self.config)
        self.projection = Wav2Vec2ConformerFeatureProjection(self.config)

    def forward(self, input):
        output = self.speech_encoder(input)
        output, _ = self.projection(output.transpose(1, 2))
        return output.transpose(1, 2)
