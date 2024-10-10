""" Module for Data Loading and Feature Extraction"""

import json
import string

import librosa
import torch
from g2p_en import G2p
from transformers import AutoProcessor

from utils.mel_spectrogram import TacotronSTFT


class FeatExtract(torch.utils.data.Dataset):
    def __init__(self, wav_file, preprocess_config):
        super(FeatExtract, self).__init__()

        self.preprocess_config = preprocess_config["preprocessing"]
        with open(wav_file, "r") as infile:
            self.wav_list = infile.read().splitlines()

        self.g2p = G2p()
        phoneme_file = f"{wav_file.rsplit('/', 1)[0]}/phoneme_tokens.txt"
        with open(phoneme_file, "r") as infile:
            self.phoneme_map = infile.read().splitlines()

        spk_file = f"{wav_file.rsplit('/', 1)[0]}/spk_list.txt"
        with open(spk_file, "r") as infile:
            self.spk_list = infile.read()
            self.spk_list = json.loads(self.spk_list)

        self.asr_tokenizer = AutoProcessor.from_pretrained(
            "facebook/data2vec-audio-base-960h"
        )

        # Mel Spectrogram Extractor
        self.STFT = TacotronSTFT(
            self.preprocess_config["stft"]["filter_length"],
            self.preprocess_config["stft"]["hop_length"],
            self.preprocess_config["stft"]["win_length"],
            self.preprocess_config["mel"]["n_mel_channels"],
            self.preprocess_config["audio"]["sampling_rate"],
            self.preprocess_config["mel"]["mel_fmin"],
            self.preprocess_config["mel"]["mel_fmax"],
        )

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path, spk, text = self.wav_list[idx].split("|")

        audio, _ = librosa.load(
            wav_path, sr=self.preprocess_config["audio"]["sampling_rate"]
        )
        audio = torch.from_numpy(audio)
        audio = audio / max(abs(audio))

        phonemes = self.g2p(text)
        for idx in range(len(phonemes)):
            if phonemes[idx] not in self.phoneme_map:
                phonemes[idx] = 78
            else:
                phonemes[idx] = self.phoneme_map.index(phonemes[idx])
        phonemes = torch.as_tensor(phonemes)

        punc_list = list(string.punctuation)
        punc_list.remove("'")
        transcript = text
        for punc in punc_list:
            transcript = transcript.replace(punc, "")
        transcript = transcript.upper()
        asr_tgt_tokens = self.asr_tokenizer.tokenizer(
            transcript, return_tensors="pt"
        ).input_ids.squeeze(0)

        spk_tgt = self.spk_list[spk]

        melspec_tgt = melspec_tgt = self.STFT.mel_spectrogram(audio.unsqueeze(0))
        melspec_tgt = melspec_tgt.squeeze(0).transpose(0, 1)
        speech_embed_len = int((round((len(audio) / 320)) - 1))
        melspec_tgt = melspec_tgt[
            :speech_embed_len, :
        ]  # Always match for conv1d features

        return audio, phonemes, asr_tgt_tokens, spk_tgt, melspec_tgt, text


class FeatLoad(torch.utils.data.Dataset):
    def __init__(self, wav_file):
        with open(wav_file, "r") as infile:
            self.wav_list = infile.read().splitlines()

        # Specific for VCTK folder setup
        for idx in range(len(self.wav_list)):
            self.wav_list[idx] = (
                self.wav_list[idx].replace("wav16", "feats").replace(".wav", ".pt")
            )

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        feat_path = self.wav_list[idx].split("|")[0]
        data = torch.load(feat_path)
        return (
            data["audio"],
            data["phonemes"],
            data["asr_tgt_tokens"],
            data["spk_tgt"],
            data["melspec_tgt"],
            data["text"],
        )


def custom_collate_fn(batch):
    audio = []
    audio_len = []
    phonemes = []
    phonemes_len = []
    asr_tgt = []
    asr_tgt_len = []
    spk_tgt = []
    melspec_tgt = []
    melspec_tgt_len = []
    text = []

    for item in batch:
        audio.append(item[0])
        audio_len.append(item[0].shape[0])
        phonemes.append(item[1])
        phonemes_len.append(item[1].shape[0])
        asr_tgt.append(item[2])
        asr_tgt_len.append(item[2].shape[0])
        spk_tgt.append(item[3])
        melspec_tgt.append(item[4])
        melspec_tgt_len.append(item[4].shape[0])
        text.append(item[5])

    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
    audio_len = torch.IntTensor(audio_len)
    phonemes = torch.nn.utils.rnn.pad_sequence(
        phonemes, batch_first=True, padding_value=0
    )
    phonemes_len = torch.IntTensor(phonemes_len)
    asr_tgt = torch.nn.utils.rnn.pad_sequence(
        asr_tgt, batch_first=True, padding_value=0
    )
    asr_tgt_len = torch.IntTensor(asr_tgt_len)
    spk_tgt = torch.LongTensor(spk_tgt)
    melspec_tgt = torch.nn.utils.rnn.pad_sequence(
        melspec_tgt, batch_first=True, padding_value=0
    )
    melspec_tgt_len = torch.IntTensor(melspec_tgt_len)

    return (
        audio,
        audio_len,
        phonemes,
        phonemes_len,
        asr_tgt,
        asr_tgt_len,
        spk_tgt,
        melspec_tgt,
        melspec_tgt_len,
        text,
    )
