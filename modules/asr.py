""" Wrapper Module for Automatic Speech Recognition"""

import torch
import torch.nn.functional as F


class ASR(torch.nn.Module):
    def __init__(self, config):
        super(ASR, self).__init__()
        self.config = config["asr"]

        # Classification Head
        self.asr_head = torch.nn.Linear(
            self.config["input_dim"], self.config["num_tokens"]
        )

    def forward(self, feats, feats_lens, asr_tgt_tokens, asr_tgt_tokens_len):
        logits = F.relu(self.asr_head(feats))

        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        asr_tgt_tokens_mask = asr_tgt_tokens > 0
        flattened_targets = asr_tgt_tokens.masked_select(asr_tgt_tokens_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs,
                flattened_targets,
                feats_lens,
                asr_tgt_tokens_len,
                blank=0,
                reduction="mean",
                zero_infinity=False,
            )

        # predicted_ids = torch.argmax(logits, dim=-1)
        return logits, loss
