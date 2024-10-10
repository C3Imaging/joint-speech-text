"""
Wrapper Module for Shared Encoder
"""

import torch.nn as nn
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class SharedEncoder(nn.Module):
    def __init__(self, config):
        super(SharedEncoder, self).__init__()
        self.config = config["shared_encoder"]
        self.shared_encoder = ConformerEncoder(
            idim=self.config["input_dim"],
            attention_dim=self.config["att_dim"],
            attention_heads=self.config["att_head"],
            linear_units=self.config["linear_units"],
            num_blocks=self.config["num_blocks"],
            input_layer=None,
            dropout_rate=self.config["dropout_rate"],
            positional_dropout_rate=self.config["positional_dropout_rate"],
            attention_dropout_rate=self.config["attention_dropout_rate"],
            normalize_before=self.config["normalize_before"],
            concat_after=self.config["concat_after"],
            positionwise_layer_type=self.config["positionwise_layer_type"],
            positionwise_conv_kernel_size=self.config["positionwise_conv_kernel_size"],
            macaron_style=self.config["macaron_style"],
            pos_enc_layer_type=self.config["pos_enc_layer_type"],
            selfattention_layer_type=self.config["selfattention_layer_type"],
            activation_type=self.config["activation_type"],
            use_cnn_module=self.config["use_cnn_module"],
            cnn_module_kernel=self.config["cnn_module_kernel"],
            zero_triu=self.config["zero_triu"],
        )

    def forward(self, feats, feats_len):
        """Encode input sequence.

        Args:
            feats (torch.Tensor): Input tensor (#batch, time, idim).
            feats_len (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        feats_mask = make_pad_mask(feats_len).unsqueeze(1)
        out, _ = self.shared_encoder(feats, feats_mask)
        return out
