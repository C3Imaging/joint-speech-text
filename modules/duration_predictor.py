"""
Wrapper module for Unsupervised Duration Alignment
Reference: ESPNet
"""

import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictor,
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DurationModel(torch.nn.Module):
    def __init__(self, config):
        super(DurationModel, self).__init__()
        self.config = config["duration"]

        self.spk_embed_proj = torch.nn.Linear(
            config["spkrecog"]["spk_embed_dim"], self.config["input_dim"]
        )
        self.duration_predictor = DurationPredictor(
            idim=self.config["input_dim"],
            n_layers=self.config["layers"],
            n_chans=self.config["filter_size"],
            kernel_size=self.config["kernel_size"],
            dropout_rate=self.config["dropout"],
        )
        self.duration_predictor_loss = DurationPredictorLoss()

    def forward(self, txt, txt_len, dur_tgt, spk_embed):
        spk_embed = self.spk_embed_proj(F.normalize(spk_embed))
        txt_masks = make_pad_mask(txt_len)
        embedded_txt = txt + spk_embed.unsqueeze(1)
        predicted_durations = self.duration_predictor(embedded_txt, txt_masks)
        duration_loss = self.duration_predictor_loss(predicted_durations, dur_tgt)

        return predicted_durations, duration_loss

    def inference(self, txt, txt_len, spk_embed):
        spk_embed = self.spk_embed_proj(F.normalize(spk_embed))
        txt_masks = make_pad_mask(txt_len)
        embedded_txt = txt + spk_embed.unsqueeze(1)
        predicted_durations = self.duration_predictor.inference(embedded_txt, txt_masks)
        return predicted_durations
