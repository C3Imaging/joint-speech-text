""" Evaluation script for all modules NOT including unseen set """

import fnmatch
import json
import multiprocessing
import os
import random
import shutil
import string
import sys

try:
    sys.path.append(os.path.abspath(os.getcwd()))
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import yaml
from parallel_wavegan.utils.utils import load_model as vocoder_load
from pymcd.mcd import Calculate_MCD
from scipy.io.wavfile import write as audio_write
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from modules.dataset import FeatLoad, custom_collate_fn
from modules.jst_model import JSTModel


def chunks(data, num):
    return [data[i : i + num] for i in range(0, len(data), num)]


def compute_mcd(job_id, data_slice):
    global preddur_min, preddur_max, guided_min, guided_max
    print(f"JOB {job_id} started")
    for wavf in tqdm(data_slice):
        preddur = os.path.join(wav_dir, wavf)
        target = os.path.join(wav_dir, wavf.replace("_preddur.wav", "_target.wav"))
        guided = os.path.join(wav_dir, wavf.replace("_preddur.wav", "_guided.wav"))

        preddur_dtw = mcd_dtw.calculate_mcd(target, preddur)
        with open(os.path.join(wav_dir, "preddur_values.txt"), "a") as outfile:
            outfile.write(f"{preddur_dtw}\n")
        guided_dtw = mcd_dtw.calculate_mcd(target, guided)
        with open(os.path.join(wav_dir, "guided_values.txt"), "a") as outfile:
            outfile.write(f"{guided_dtw}\n")

    print(f"JOB {job_id} finished")


def dispatch_jobs(data, job_number):
    total = len(data)
    chunk_size = total // job_number
    slice = chunks(data, chunk_size)
    jobs = []

    for i, s in enumerate(slice):
        j = multiprocessing.Process(target=compute_mcd, args=(i, s))
        jobs.append(j)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()


config = sys.argv[1]
config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)

checkpoint_dir = os.path.join(
    config["training"]["exp_dir"], config["evaluation"]["checkpoint"]
)

# Model Loading
device = config["evaluation"]["device"]
jst_model = JSTModel(config).to(device)
jst_model.load_model(checkpoint_dir)
lm_processor = AutoProcessor.from_pretrained("./language_model")
asr_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
vocoder = vocoder_load("./vocoder/checkpoint-1100000steps.pkl").to(device)

wer_fn = torchmetrics.WordErrorRate()
cer_fn = torchmetrics.CharErrorRate()

punc_list = string.punctuation
punc_list = punc_list.replace("'", "")


# SEEN
eval_dir = os.path.join(checkpoint_dir, "seen")
spk_file = config["evaluation"]["spk_file"]
with open(spk_file, "r") as infile:
    spk_list = infile.read()
    spk_list = json.loads(spk_list)
    spk_list = list(spk_list.keys())
    spk_list = spk_list[:-9]  # Particular for VCTK, remove unseen speakers
tsne_spks = random.choices(spk_list, k=10)

if os.path.exists(eval_dir):
    shutil.rmtree(eval_dir)
os.makedirs(eval_dir, exist_ok=False)
os.makedirs(os.path.join(eval_dir, "wav"), exist_ok=False)

dataset = FeatLoad(config["evaluation"]["seen_list"])
dataloader = DataLoader(
    dataset,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=True,
    batch_size=config["evaluation"]["batch_size"],
)

asr_out = open(os.path.join(eval_dir, "asr_transcript.txt"), "a")

with torch.no_grad():
    jst_model.eval()
    wer_total = 0
    cer_total = 0
    asr_wer_total = 0
    asr_cer_total = 0
    spkrecog_acc_total = 0

    spk_embeds_list = []
    spk_labels = []

    for batch, (
        audio,
        audio_len,
        text,
        text_len,
        asr_tgt,
        asr_tgt_len,
        spk_tgt,
        melspec_tgt,
        melspec_tgt_len,
        transcript,
    ) in enumerate(tqdm(dataloader)):
        # Load features to device
        audio = audio.to(device)
        audio_len = audio_len.to(device)
        text = text.to(device)
        text_len = text_len.to(device)
        asr_tgt = asr_tgt.to(device)
        asr_tgt_len = asr_tgt_len.to(device)
        spk_tgt = spk_tgt.to(device)
        melspec_tgt = melspec_tgt.to(device)
        melspec_tgt_len = melspec_tgt_len.to(device)

        (
            spk_embeds,
            spkrecog_acc,
            asr_logits,
            preddur_melspec,
            preddur_melspec_len,
            guided_melspec,
            before_postnet,
            log_p_attn,
        ) = jst_model.evaluate(
            audio,
            audio_len,
            text,
            text_len,
            asr_tgt,
            asr_tgt_len,
            spk_tgt,
            melspec_tgt,
            melspec_tgt_len,
        )

        # ASR OUTPUTS
        for idx in range(len(transcript)):
            data = transcript[idx]
            for punc in punc_list:
                data = data.replace(punc, "")
            data = data.upper()
            transcript[idx] = data

        lm_predicted_txt = []
        asr_text_list = []
        for idx, logits in enumerate(asr_logits.cpu().numpy()):
            pred_text = lm_processor.decode(logits).text
            lm_predicted_txt.append(pred_text)
            asr_text = asr_processor.decode(torch.argmax(torch.tensor(logits), dim=-1))
            asr_text_list.append(asr_text)
            asr_out.write(
                f"ASR: {asr_text} -- LM: {pred_text} -- TGT: {transcript[idx]}\n"
            )
        wer_total += wer_fn(lm_predicted_txt, transcript).item()
        cer_total += cer_fn(lm_predicted_txt, transcript).item()
        asr_wer_total += wer_fn(asr_text_list, transcript).item()
        asr_cer_total += cer_fn(asr_text_list, transcript).item()

        # TTS OUTPUTS
        target_audio_list = []
        guided_audio_list = []
        preddur_audio_list = []
        for idx in range(len(melspec_tgt)):
            target_audio_list.append(
                vocoder.inference(
                    melspec_tgt[idx][: melspec_tgt_len[idx] - 1], normalize_before=True
                ).transpose(0, 1)
            )
            guided_audio_list.append(
                vocoder.inference(
                    guided_melspec[idx][: melspec_tgt_len[idx] - 1],
                    normalize_before=True,
                ).transpose(0, 1)
            )
            preddur_audio_list.append(
                vocoder.inference(
                    preddur_melspec[idx][: preddur_melspec_len[idx] - 1],
                    normalize_before=True,
                ).transpose(0, 1)
            )

        for idx in range(len(target_audio_list)):
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_target.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                target_audio_list[idx].squeeze(0).cpu().numpy(),
            )
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_guided.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                guided_audio_list[idx].squeeze(0).cpu().numpy(),
            )
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_preddur.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                preddur_audio_list[idx].squeeze(0).cpu().numpy(),
            )

        # Mel Spectrogram Plotting
        fig, axs = plt.subplots(
            melspec_tgt.shape[0],
            4,
            sharex=True,
            sharey=True,
            figsize=(10.80, 7.20),
        )
        fig.suptitle("Before Postnet vs Predicted vs Target vs Attention")
        for idx in range(melspec_tgt.shape[0]):
            preddur_melspec[idx][preddur_melspec_len[idx] :] = 0.0
            axs[idx, 1].imshow(
                preddur_melspec[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            axs[idx, 2].imshow(
                melspec_tgt[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            axs[idx, 3].imshow(
                log_p_attn[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            before_postnet[idx][preddur_melspec_len[idx] :] = 0.0
            axs[idx, 0].imshow(
                before_postnet[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
        fig.savefig(os.path.join(eval_dir, f"{batch}_melspec.png"))
        plt.close(fig)

        # Speaker Recognition Results
        spkrecog_acc_total += spkrecog_acc
        for idx in range(spk_embeds.shape[0]):
            if spk_list[spk_tgt[idx].cpu()] in tsne_spks:
                spk_embeds_list.append(spk_embeds[idx].cpu())
                spk_labels.append(spk_list[spk_tgt[idx].cpu()])


asr_summary = f"\nASR WER: {asr_wer_total/len(dataloader)*100:.4f}% -- CER: {asr_cer_total/len(dataloader)*100:.4f}%\n"
asr_out.write(asr_summary)
asr_summary = f"LM WER: {wer_total/len(dataloader)*100:.4f}% -- CER: {cer_total/len(dataloader)*100:.4f}%\n"
asr_out.write(asr_summary)
asr_out.close()

with open(os.path.join(eval_dir, "spkrecog.txt"), "a") as outfile:
    spkrecog_summary = f"ACC: {spkrecog_acc_total/len(dataloader):.4f}%\n"
    outfile.write(spkrecog_summary)

color_dict = [
    "black",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "blue",
    "teal",
    "purple",
    "lime",
]
color_map = {}
for item, spk in enumerate(tsne_spks):
    color_map[spk] = color_dict[item]

spk_embeds_list = torch.stack(spk_embeds_list, dim=0)
tsne = TSNE(n_components=2, random_state=123)
spk_embeds_2d = tsne.fit_transform(spk_embeds_list)

fig, axs = plt.subplots(1, 1, figsize=(10.80, 10.80))

for idx in range(spk_embeds_list.shape[0]):
    color = color_map[spk_labels[idx]]
    label = (
        spk_labels[idx]
        if spk_labels[idx] not in axs.get_legend_handles_labels()[1]
        else ""
    )
    axs.scatter(spk_embeds_2d[idx][0], spk_embeds_2d[idx][1], c=color, label=label)
axs.legend()
fig.savefig(os.path.join(eval_dir, "seen_tsne.png"))

mcd_dtw = Calculate_MCD(MCD_mode="dtw")
wav_dir = os.path.join(eval_dir, "wav")
wav_list = os.listdir(wav_dir)
wav_list = [
    i
    for i in wav_list
    if not (fnmatch.fnmatch(i, "*guided*") or fnmatch.fnmatch(i, "*target*"))
]

dispatch_jobs(wav_list, 16)

with open(os.path.join(wav_dir, "preddur_values.txt"), "r") as infile:
    preddur_values = infile.read().splitlines()
preddur_values = [float(i) for i in preddur_values]

preddur_mean = np.mean(np.array(preddur_values))
preddur_std = np.std(np.array(preddur_values))
preddur_min = np.min(np.array(preddur_values))
preddur_max = np.max(np.array(preddur_values))

with open(os.path.join(wav_dir, "guided_values.txt"), "r") as infile:
    guided_values = infile.read().splitlines()
guided_values = [float(i) for i in guided_values]

guided_mean = np.mean(np.array(guided_values))
guided_std = np.std(np.array(guided_values))
guided_min = np.min(np.array(guided_values))
guided_max = np.max(np.array(guided_values))


preddur_summary = f"PREDDUR -- MEAN: {preddur_mean:.4f} -- STDEV: {preddur_std:.4f} -- MIN: {preddur_min:.4f} -- MAX: {preddur_max:.4f}"
guided_summary = f"GUIDED -- MEAN: {guided_mean:.4f} -- STDEV: {guided_std:.4f} -- MIN: {guided_min:.4f} -- MAX: {guided_max:.4f}"

with open(os.path.join(eval_dir, "tts.txt"), "a") as outfile:
    outfile.write(f"{preddur_summary}\n{guided_summary}\n")


# UNSEEN
eval_dir = os.path.join(checkpoint_dir, "unseen")
spk_file = config["evaluation"]["spk_file"]
with open(spk_file, "r") as infile:
    spk_list = infile.read()
    spk_list = json.loads(spk_list)
    spk_list = list(spk_list.keys())
    spk_list = spk_list[-9:]  # Particular for VCTK, remove seen speakers
tsne_spks = random.choices(spk_list, k=10)

if os.path.exists(eval_dir):
    shutil.rmtree(eval_dir)
os.makedirs(eval_dir, exist_ok=False)
os.makedirs(os.path.join(eval_dir, "wav"), exist_ok=False)

dataset = FeatLoad(config["evaluation"]["unseen_list"])
dataloader = DataLoader(
    dataset,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=True,
    batch_size=config["evaluation"]["batch_size"],
)

asr_out = open(os.path.join(eval_dir, "asr_transcript.txt"), "a")

with torch.no_grad():
    jst_model.eval()
    wer_total = 0
    cer_total = 0
    asr_wer_total = 0
    asr_cer_total = 0
    spkrecog_acc_total = 0

    spk_embeds_list = []
    spk_labels = []

    for batch, (
        audio,
        audio_len,
        text,
        text_len,
        asr_tgt,
        asr_tgt_len,
        spk_tgt,
        melspec_tgt,
        melspec_tgt_len,
        transcript,
    ) in enumerate(tqdm(dataloader)):
        # Load features to device
        audio = audio.to(device)
        audio_len = audio_len.to(device)
        text = text.to(device)
        text_len = text_len.to(device)
        asr_tgt = asr_tgt.to(device)
        asr_tgt_len = asr_tgt_len.to(device)
        spk_tgt = spk_tgt.to(device)
        melspec_tgt = melspec_tgt.to(device)
        melspec_tgt_len = melspec_tgt_len.to(device)

        (
            spk_embeds,
            spkrecog_acc,
            asr_logits,
            preddur_melspec,
            preddur_melspec_len,
            guided_melspec,
            before_postnet,
            log_p_attn,
        ) = jst_model.evaluate(
            audio,
            audio_len,
            text,
            text_len,
            asr_tgt,
            asr_tgt_len,
            spk_tgt,
            melspec_tgt,
            melspec_tgt_len,
        )

        # ASR OUTPUTS
        for idx in range(len(transcript)):
            data = transcript[idx]
            for punc in punc_list:
                data = data.replace(punc, "")
            data = data.upper()
            transcript[idx] = data

        lm_predicted_txt = []
        asr_text_list = []
        for idx, logits in enumerate(asr_logits.cpu().numpy()):
            pred_text = lm_processor.decode(logits).text
            lm_predicted_txt.append(pred_text)
            asr_text = asr_processor.decode(torch.argmax(torch.tensor(logits), dim=-1))
            asr_text_list.append(asr_text)
            asr_out.write(
                f"ASR: {asr_text} -- LM: {pred_text} -- TGT: {transcript[idx]}\n"
            )
        wer_total += wer_fn(lm_predicted_txt, transcript).item()
        cer_total += cer_fn(lm_predicted_txt, transcript).item()
        asr_wer_total += wer_fn(asr_text_list, transcript).item()
        asr_cer_total += cer_fn(asr_text_list, transcript).item()

        # TTS OUTPUTS
        target_audio_list = []
        guided_audio_list = []
        preddur_audio_list = []
        for idx in range(len(melspec_tgt)):
            target_audio_list.append(
                vocoder.inference(
                    melspec_tgt[idx][: melspec_tgt_len[idx] - 1], normalize_before=True
                ).transpose(0, 1)
            )
            guided_audio_list.append(
                vocoder.inference(
                    guided_melspec[idx][: melspec_tgt_len[idx] - 1],
                    normalize_before=True,
                ).transpose(0, 1)
            )
            preddur_audio_list.append(
                vocoder.inference(
                    preddur_melspec[idx][: preddur_melspec_len[idx] - 1],
                    normalize_before=True,
                ).transpose(0, 1)
            )

        for idx in range(len(target_audio_list)):
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_target.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                target_audio_list[idx].squeeze(0).cpu().numpy(),
            )
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_guided.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                guided_audio_list[idx].squeeze(0).cpu().numpy(),
            )
            outpath = os.path.join(eval_dir, "wav", f"{batch}_{idx}_preddur.wav")
            audio_write(
                outpath,
                config["preprocessing"]["audio"]["sampling_rate"],
                preddur_audio_list[idx].squeeze(0).cpu().numpy(),
            )

        # Mel Spectrogram Plotting
        fig, axs = plt.subplots(
            melspec_tgt.shape[0],
            4,
            sharex=True,
            sharey=True,
            figsize=(10.80, 7.20),
        )
        fig.suptitle("Before Postnet vs Predicted vs Target vs Attention")
        for idx in range(melspec_tgt.shape[0]):
            preddur_melspec[idx][preddur_melspec_len[idx] :] = 0.0
            axs[idx, 1].imshow(
                preddur_melspec[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            axs[idx, 2].imshow(
                melspec_tgt[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            axs[idx, 3].imshow(
                log_p_attn[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
            before_postnet[idx][preddur_melspec_len[idx] :] = 0.0
            axs[idx, 0].imshow(
                before_postnet[idx].transpose(0, 1).cpu().numpy(), origin="lower"
            )
        fig.savefig(os.path.join(eval_dir, f"{batch}_melspec.png"))
        plt.close(fig)

        # Speaker Recognition Results
        spkrecog_acc_total += spkrecog_acc
        for idx in range(spk_embeds.shape[0]):
            if spk_list[spk_tgt[idx].cpu()] in tsne_spks:
                spk_embeds_list.append(spk_embeds[idx].cpu())
                spk_labels.append(spk_list[spk_tgt[idx].cpu()])


asr_summary = f"\nASR WER: {asr_wer_total/len(dataloader)*100:.4f}% -- CER: {asr_cer_total/len(dataloader)*100:.4f}%\n"
asr_out.write(asr_summary)
asr_summary = f"LM WER: {wer_total/len(dataloader)*100:.4f}% -- CER: {cer_total/len(dataloader)*100:.4f}%\n"
asr_out.write(asr_summary)
asr_out.close()

with open(os.path.join(eval_dir, "spkrecog.txt"), "a") as outfile:
    spkrecog_summary = f"ACC: {spkrecog_acc_total/len(dataloader):.4f}%\n"
    outfile.write(spkrecog_summary)

color_dict = [
    "black",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "blue",
    "teal",
    "purple",
    "lime",
]
color_map = {}
for item, spk in enumerate(tsne_spks):
    color_map[spk] = color_dict[item]

spk_embeds_list = torch.stack(spk_embeds_list, dim=0)
tsne = TSNE(n_components=2, random_state=123)
spk_embeds_2d = tsne.fit_transform(spk_embeds_list)

fig, axs = plt.subplots(1, 1, figsize=(10.80, 10.80))

for idx in range(spk_embeds_list.shape[0]):
    color = color_map[spk_labels[idx]]
    label = (
        spk_labels[idx]
        if spk_labels[idx] not in axs.get_legend_handles_labels()[1]
        else ""
    )
    axs.scatter(spk_embeds_2d[idx][0], spk_embeds_2d[idx][1], c=color, label=label)
axs.legend()
fig.savefig(os.path.join(eval_dir, "seen_tsne.png"))


mcd_dtw = Calculate_MCD(MCD_mode="dtw")
wav_dir = os.path.join(eval_dir, "wav")
wav_list = os.listdir(wav_dir)
wav_list = [
    i
    for i in wav_list
    if not (fnmatch.fnmatch(i, "*guided*") or fnmatch.fnmatch(i, "*target*"))
]

dispatch_jobs(wav_list, 16)

with open(os.path.join(wav_dir, "preddur_values.txt"), "r") as infile:
    preddur_values = infile.read().splitlines()
preddur_values = [float(i) for i in preddur_values]

preddur_mean = np.mean(np.array(preddur_values))
preddur_std = np.std(np.array(preddur_values))
preddur_min = np.min(np.array(preddur_values))
preddur_max = np.max(np.array(preddur_values))

with open(os.path.join(wav_dir, "guided_values.txt"), "r") as infile:
    guided_values = infile.read().splitlines()
guided_values = [float(i) for i in guided_values]

guided_mean = np.mean(np.array(guided_values))
guided_std = np.std(np.array(guided_values))
guided_min = np.min(np.array(guided_values))
guided_max = np.max(np.array(guided_values))


preddur_summary = f"PREDDUR -- MEAN: {preddur_mean:.4f} -- STDEV: {preddur_std:.4f} -- MIN: {preddur_min:.4f} -- MAX: {preddur_max:.4f}"
guided_summary = f"GUIDED -- MEAN: {guided_mean:.4f} -- STDEV: {guided_std:.4f} -- MIN: {guided_min:.4f} -- MAX: {guided_max:.4f}"

with open(os.path.join(eval_dir, "tts.txt"), "a") as outfile:
    outfile.write(f"{preddur_summary}\n{guided_summary}\n")
