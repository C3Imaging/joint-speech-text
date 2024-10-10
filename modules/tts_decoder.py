"""
Wrapper Module for Mel Spectrogram Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet


class TTSDecoder(nn.Module):
    def __init__(self, config):
        super(TTSDecoder, self).__init__()
        self.config = config["tts_decoder"]
        self.tts_decoder = ConformerEncoder(
            idim=self.config["input_dim"],
            attention_dim=self.config["att_dim"],
            attention_heads=self.config["att_head"],
            linear_units=self.config["linear_units"],
            num_blocks=self.config["num_blocks"],
            input_layer=self.config["input_layer"],
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
        self.out_proj = torch.nn.Linear(
            self.config["att_dim"], self.config["melspec_dim"]
        )
        self.postnet = Postnet(
            idim=self.config["melspec_dim"],
            odim=self.config["melspec_dim"],
            n_layers=self.config["postnet_layers"],
            n_chans=self.config["postnet_chans"],
            n_filts=self.config["postnet_filts"],
            use_batch_norm=True,
            dropout_rate=self.config["postnet_dropout_rate"],
        )
        self.spk_embed_proj = torch.nn.Linear(
            config["spkrecog"]["spk_embed_dim"], self.config["input_dim"]
        )

    def forward(self, feats, feats_len, spk_embeds):
        """Encode input sequence.

        Args:
            feats (torch.Tensor): Input tensor (#batch, time, idim).
            feats_len (torch.Tensor): Mask tensor (#batch, time).
            spk_embeds (torch.Tensor): (#batch, att_dim)

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        spk_embeds = self.spk_embed_proj(F.normalize(spk_embeds))
        feats_mask = make_pad_mask(feats_len).unsqueeze(1)
        embedded_feats = feats + spk_embeds.unsqueeze(1)
        out, _ = self.tts_decoder(embedded_feats, feats_mask)
        before_postnet = self.out_proj(out)
        after_postnet = before_postnet + self.postnet(
            before_postnet.transpose(1, 2)
        ).transpose(1, 2)
        return after_postnet, before_postnet
