"""
Wrapper module for Unsupervised Duration Alignment
Reference: ESPNet
"""

import torch
from espnet2.gan_tts.jets.alignments import AlignmentModule, viterbi_decode
from espnet2.gan_tts.jets.loss import ForwardSumLoss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class AlignerModel(torch.nn.Module):
    def __init__(self, config):
        super(AlignerModel, self).__init__()
        self.config = config["aligner"]

        self.alignment_encoder = AlignmentModule(
            self.config["txt_embed_dim"], self.config["asr_embed_dim"]
        )
        self.forwardsum_loss = ForwardSumLoss()

    def forward(self, speech, speech_len, txt, txt_len):
        """Args:
            text (Tensor): Batched text embedding (B, T_text, adim).
            speech (Tensor): Batched acoustic feature (B, T_feats, odim).
            txt_len (Tensor): Text length tensor (B,).
            speech_len (Tensor): Feature length tensor (B,).

        Returns:
            Tensor: Log probability of attention matrix (B, T_feats, T_text).
        """

        txt_masks = make_pad_mask(txt_len)
        log_p_attn = self.alignment_encoder(txt, speech, txt_len, speech_len, txt_masks)
        forwardsum_loss = self.forwardsum_loss(log_p_attn, txt_len, speech_len)
        ds, bin_loss = viterbi_decode(log_p_attn, txt_len, speech_len)

        return ds, bin_loss, forwardsum_loss, log_p_attn
