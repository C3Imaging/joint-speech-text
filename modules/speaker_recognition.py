""" Wrapper Module for MFA Conformer Speaker Recognition"""

import torch

from modules.mfa_conformer.amsoftmax_loss import AMSoftmax
from modules.mfa_conformer.conformer_cat import Conformer


class SpeakerRecognition(torch.nn.Module):
    def __init__(self, spkrecog_config):
        super(SpeakerRecognition, self).__init__()
        self.spkrecog_config = spkrecog_config["spkrecog"]
        self.spk_embedder = Conformer(
            n_mels=self.spkrecog_config["input_dim"],
            num_blocks=self.spkrecog_config["num_blocks"],
            output_size=self.spkrecog_config["spk_embed_dim"],
            embedding_dim=self.spkrecog_config["spk_embed_dim"],
            input_layer=self.spkrecog_config["input_layer"],
            pos_enc_layer_type=self.spkrecog_config["pos_enc_layer_type"],
        )
        self.loss_function = AMSoftmax(
            in_feats=self.spkrecog_config["spk_embed_dim"],
            n_classes=self.spkrecog_config["num_speakers"],
            m=self.spkrecog_config["margin"],
            s=self.spkrecog_config["scale"],
        )

    def forward(self, feats, lens, targets):
        """
        Feats: [B, DIM, T]
        Lens: [B]
        Targets: [B]
        """

        spk_embeds = self.spk_embedder(feats.transpose(1, 2), lens)
        loss = self.loss_function(spk_embeds, targets)
        return spk_embeds, loss

    def inference(self, feats, lens, targets):
        spk_embeds = self.spk_embedder(feats.transpose(1, 2), lens)
        acc = self.loss_function.accuracy(spk_embeds, targets)
        return spk_embeds, acc
