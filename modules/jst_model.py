"""
Wrapper Module for Joint-Speech-Text Model
"""

import os

import torch
from espnet2.gan_tts.jets.length_regulator import GaussianUpsampling
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

from modules.aligner import AlignerModel
from modules.asr import ASR
from modules.duration_predictor import DurationModel
from modules.shared_encoder import SharedEncoder
from modules.speaker_recognition import SpeakerRecognition
from modules.speech_feature import SpeechEncoder
from modules.text_feature import TextEncoder
from modules.tts_decoder import TTSDecoder


class JSTModel(torch.nn.Module):
    def __init__(self, config):
        super(JSTModel, self).__init__()
        self.config = config

        self.speech_encoder = SpeechEncoder(config)
        self.asr = ASR(config)
        self.spkrecog = SpeakerRecognition(config)
        self.aligner = AlignerModel(config)
        self.duration_predictor = DurationModel(config)
        self.shared_encoder = SharedEncoder(config)
        self.tts_decoder = TTSDecoder(config)
        self.text_encoder = TextEncoder(config)

        self.length_regulator = GaussianUpsampling()
        self.asr_pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.shared_encoder_loss_fn = torch.nn.L1Loss(reduction="none")
        self.tts_decoder_loss_fn = torch.nn.L1Loss(reduction="none")

    def forward(
        self,
        audio,
        audio_len,
        text,
        text_len,
        asr_tgt,
        asr_tgt_len,
        spk_tgt,
        melspec_tgt,
        melspec_tgt_len,
    ):
        """
        audio: [B, T]
        audio_len: [B]
        text: [B, T]
        text_len: [B]
        asr_tgt: [B, T]
        asr_tgt_len: [B]
        melspec_tgt: [B, DIM, T]
        melspec_tgt_len: [B, T]
        """

        # Speech Encoder
        speech_embeds = self.speech_encoder(audio)
        speech_embeds_len = (((audio_len / 320)).round() - 1).to(
            torch.int
        )  # hopsize is 320, 16000 -> 49 for the minus 1

        if speech_embeds.shape[-1] > melspec_tgt_len.max():
            speech_embeds = speech_embeds[:, :, : melspec_tgt_len.max()]
        elif speech_embeds.shape[-1] < melspec_tgt_len.max():
            melspec_tgt = melspec_tgt[:, :, : speech_embeds.shape[-1]]

        # Speaker Recognition
        spk_embeds, spkrecog_loss = self.spkrecog(
            speech_embeds, speech_embeds_len, spk_tgt
        )

        # Shared Encoder and ASR
        asr_embeds = self.shared_encoder(
            speech_embeds.transpose(1, 2), speech_embeds_len
        )
        _, asr_loss = self.asr(asr_embeds, speech_embeds_len, asr_tgt, asr_tgt_len)

        # Text Encoder
        text_embeds = self.text_encoder(text, text_len)

        # Aligner (STOP GRADIENT)
        duration_target, bin_loss, forwardsum_loss, _ = self.aligner(
            speech_embeds.transpose(1, 2).detach(),
            speech_embeds_len,
            text_embeds.detach(),
            text_len,
        )
        aligner_loss = forwardsum_loss + bin_loss

        # Duration Predictor (STOP GRADIENT)
        predicted_duration, duration_loss = self.duration_predictor(
            text_embeds.detach(),
            text_len,
            duration_target.detach(),
            spk_embeds.detach(),
        )

        # Upsampling
        h_masks = make_non_pad_mask(speech_embeds_len)
        d_masks = make_non_pad_mask(text_len)
        upsampled_text_embeds = self.length_regulator(
            text_embeds, duration_target, h_masks, d_masks
        )

        # Shared Encoder for Text and Loss
        shared_text_embeds = self.shared_encoder(
            upsampled_text_embeds, speech_embeds_len
        )
        shared_encoder_loss = self.shared_encoder_loss_fn(
            shared_text_embeds, asr_embeds
        )

        # TTS Decoder
        after_melspec, before_melspec = self.tts_decoder(
            shared_text_embeds, speech_embeds_len, spk_embeds.detach()
        )

        # L1 Loss Weighting
        weights_melspec = (
            melspec_tgt.abs()
            .sum(-1, keepdim=True)
            .ne(0)
            .float()
            .repeat(1, 1, melspec_tgt.shape[-1])
        )
        weights_shared = (
            melspec_tgt.abs()
            .sum(-1, keepdim=True)
            .ne(0)
            .float()
            .repeat(1, 1, speech_embeds.shape[-2])
        )

        before_tts_loss = self.tts_decoder_loss_fn(before_melspec, melspec_tgt)
        before_tts_loss = (
            before_tts_loss * weights_melspec
        ).sum() / weights_melspec.sum()

        after_tts_loss = self.tts_decoder_loss_fn(after_melspec, melspec_tgt)
        after_tts_loss = (
            after_tts_loss * weights_melspec
        ).sum() / weights_melspec.sum()

        tts_loss = before_tts_loss + after_tts_loss

        shared_encoder_loss = (
            shared_encoder_loss * weights_shared
        ).sum() / weights_shared.sum()

        return (
            asr_loss,
            spkrecog_loss,
            aligner_loss,
            duration_loss,
            shared_encoder_loss,
            tts_loss,
            forwardsum_loss,
            bin_loss,
            before_tts_loss,
            after_tts_loss,
        )

    def evaluate(
        self,
        audio,
        audio_len,
        text,
        text_len,
        asr_tgt,
        asr_tgt_len,
        spk_tgt,
        melspec_tgt,
        melspec_tgt_len,
    ):
        with torch.no_grad():
            # Speech Encoder
            speech_embeds = self.speech_encoder(audio)
            speech_embeds_len = (((audio_len / 320)).round() - 1).to(
                torch.int
            )  # hopsize is 320, 16000 -> 49 for the minus 1

            if speech_embeds.shape[-1] > max(speech_embeds_len):
                speech_embeds_len += speech_embeds.shape[-1] - max(speech_embeds_len)
            elif speech_embeds.shape[-1] < max(speech_embeds_len):
                speech_embeds_len -= max(speech_embeds_len) - speech_embeds.shape[-1]

            # Speaker Recognition
            spk_embeds, spkrecog_acc = self.spkrecog.inference(
                speech_embeds, speech_embeds_len, spk_tgt
            )

            # Shared Encoder and ASR
            asr_embeds = self.shared_encoder(
                speech_embeds.transpose(1, 2), speech_embeds_len
            )
            asr_logits, _ = self.asr(
                asr_embeds, speech_embeds_len, asr_tgt, asr_tgt_len
            )

            # Text Encoder
            text_embeds = self.text_encoder(text, text_len)

            # Aligner
            duration_target, _, _, log_p_attn = self.aligner(
                speech_embeds.transpose(1, 2).detach(),
                speech_embeds_len,
                text_embeds.detach(),
                text_len,
            )
            # Duration Predictor (STOP GRADIENT)
            predicted_duration = self.duration_predictor.inference(
                text_embeds.detach(),
                text_len,
                spk_embeds.detach(),
            )
            preddur_text_embeds_len = torch.sum(predicted_duration, dim=1).int()

            # Upsampling
            h_masks = make_non_pad_mask(preddur_text_embeds_len)
            d_masks = make_non_pad_mask(text_len)
            preddur_text_embeds = self.length_regulator(
                text_embeds, predicted_duration, h_masks, d_masks
            )
            h_masks = make_non_pad_mask(speech_embeds_len)
            guided_text_embeds = self.length_regulator(
                text_embeds, duration_target, h_masks, d_masks
            )

            # Shared Encoder
            shared_preddur_text_embeds = self.shared_encoder(
                preddur_text_embeds, preddur_text_embeds_len
            )
            shared_guided_text_embeds = self.shared_encoder(
                guided_text_embeds, speech_embeds_len
            )

            # TTS Decoder
            preddur_melspec, before_postnet = self.tts_decoder(
                shared_preddur_text_embeds, preddur_text_embeds_len, spk_embeds
            )
            guided_melspec, _ = self.tts_decoder(
                shared_guided_text_embeds, speech_embeds_len, spk_embeds
            )

        return (
            spk_embeds,
            spkrecog_acc,
            asr_logits,
            preddur_melspec,
            preddur_text_embeds_len,
            guided_melspec,
            before_postnet,
            log_p_attn,
        )

    def asr_inference(self, audio):
        audio = audio.unsqueeze(0)
        asr_tgt = torch.randint(1, 10, (1, 10))  # Dummy target
        asr_tgt_len = torch.tensor([10])  # Dummy target
        with torch.no_grad():
            speech_embeds = self.speech_encoder(audio)
            speech_embeds_len = torch.tensor(
                [(round(((len(audio[0]) / 320))) - 1)]
            )  # hopsize is 320, 16000 -> 49 for the minus 1
            if speech_embeds.shape[-1] > speech_embeds_len:
                speech_embeds_len += speech_embeds.shape[-1] - speech_embeds_len
            elif speech_embeds.shape[-1] < speech_embeds_len:
                speech_embeds_len -= speech_embeds_len - speech_embeds.shape[-1]

            asr_embeds = self.shared_encoder(
                speech_embeds.transpose(1, 2), speech_embeds_len
            )
            asr_logits, _ = self.asr(
                asr_embeds, speech_embeds_len, asr_tgt, asr_tgt_len
            )
            return asr_logits

    def tts_inference(self, text, audio):
        audio = audio.unsqueeze(0)
        spk_tgt = torch.tensor([3])  # Dummy target
        text = text.unsqueeze(0)
        with torch.no_grad():
            speech_embeds = self.speech_encoder(audio)
            speech_embeds_len = torch.tensor(
                [(round(((len(audio[0]) / 320))) - 1)]
            )  # hopsize is 320, 16000 -> 49 for the minus 1
            if speech_embeds.shape[-1] > speech_embeds_len:
                speech_embeds_len += speech_embeds.shape[-1] - speech_embeds_len
            elif speech_embeds.shape[-1] < speech_embeds_len:
                speech_embeds_len -= speech_embeds_len - speech_embeds.shape[-1]
            spk_embeds, _ = self.spkrecog.inference(
                speech_embeds, speech_embeds_len, spk_tgt
            )

            text_len = torch.tensor([len(text[0])])
            text_embeds = self.text_encoder(text, text_len)

            predicted_duration = self.duration_predictor.inference(
                text_embeds.detach(),
                text_len,
                spk_embeds.detach(),
            )
            preddur_text_embeds_len = torch.sum(predicted_duration, dim=1).int()
            h_masks = make_non_pad_mask(preddur_text_embeds_len)
            d_masks = make_non_pad_mask(text_len)
            preddur_text_embeds = self.length_regulator(
                text_embeds, predicted_duration, h_masks, d_masks
            )

            shared_preddur_text_embeds = self.shared_encoder(
                preddur_text_embeds, preddur_text_embeds_len
            )
            preddur_melspec, _ = self.tts_decoder(
                shared_preddur_text_embeds, preddur_text_embeds_len, spk_embeds
            )
            return preddur_melspec

    def save_model(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "checkpoint.pth"))

    def load_model(self, load_path):
        self.load_state_dict(torch.load(os.path.join(load_path, "checkpoint.pth")))
