""" Training Script for all modules using ddp"""

import os
import shutil
import sys

try:
    sys.path.append(os.path.abspath(os.getcwd()))
except Exception:
    pass

import torch
import torch.multiprocessing as mp
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.log_utils as log_utils
from modules.dataset import FeatLoad, custom_collate_fn
from modules.jst_model import JSTModel

if len(sys.argv) != 2:
    print("Usage: python scripts/train_all_ddp.py <config_file>")
    sys.exit(1)


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "13579"
    os.environ["MASTER_PORT"] = "24680"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int, world_size: int, config):
    ddp_setup(rank, world_size)

    # Logging Utilities
    if rank == 0:
        logger = log_utils.logger(
            __name__,
            os.path.join(config["training"]["exp_dir"], f"gpu_{rank}_train.log"),
        )
        tb_logger = SummaryWriter(
            os.path.join(config["training"]["exp_dir"], "tensorboard")
        )

    # Training Proper
    train_dataset = FeatLoad(config["training"]["train_list"])
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=True,
        batch_size=config["training"]["batch_size"],
        sampler=DistributedSampler(train_dataset),
    )

    jst_model = JSTModel(config)
    print(jst_model)
    jst_model = jst_model.to(rank)
    jst_optimizer = torch.optim.Adam(
        jst_model.parameters(),
        lr=float(config["training"]["global_learning_rate"]),
        betas=(0.9, 0.98),
        eps=1e-09,
    )

    continue_epoch = -1  # epoch numbering starts at 0

    if config["training"]["continue"]:
        checkpoint = config["training"]["continue_checkpoint"]
        checkpoint_dir = os.path.join(config["training"]["exp_dir"], checkpoint)
        continue_epoch = int(checkpoint.split("_")[-1])

        jst_model.load_model(checkpoint_dir)
        jst_optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        )

        logger.info(f"Model is loaded from {checkpoint_dir}")

    jst_model = DDP(jst_model, device_ids=[rank])
    jst_model.train()

    for epoch in range(continue_epoch + 1, config["training"]["max_epoch"]):
        train_dataloader.sampler.set_epoch(epoch)

        epoch_total_loss = 0
        epoch_asr_loss = 0
        epoch_spkrecog_loss = 0
        epoch_aligner_loss = 0
        epoch_duration_loss = 0
        epoch_shared_encoder_loss = 0
        epoch_tts_loss = 0
        epoch_forwardsum_loss = 0
        epoch_bin_loss = 0
        epoch_before_tts_loss = 0
        epoch_after_tts_loss = 0

        jst_optimizer.zero_grad()  # Initial grad zero-out

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
            _,
        ) in enumerate(tqdm(train_dataloader)):
            # Load features to device
            audio = audio.to(jst_model.device)
            audio_len = audio_len.to(jst_model.device)
            text = text.to(jst_model.device)
            text_len = text_len.to(jst_model.device)
            asr_tgt = asr_tgt.to(jst_model.device)
            asr_tgt_len = asr_tgt_len.to(jst_model.device)
            spk_tgt = spk_tgt.to(jst_model.device)
            melspec_tgt = melspec_tgt.to(jst_model.device)
            melspec_tgt_len = melspec_tgt_len.to(jst_model.device)

            if ((batch + 1) % config["training"]["grad_accum"]) == 0 or (
                (batch + 1) == len(train_dataloader)
            ):
                # Finished N steps of accumulating gradients, proceed to update all weights
                (
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
                ) = jst_model(
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

                total_loss = (
                    asr_loss
                    + spkrecog_loss
                    + aligner_loss
                    + duration_loss
                    + shared_encoder_loss
                    + tts_loss
                )
                total_loss = total_loss / config["training"]["grad_accum"]

                # Model Update
                total_loss.backward()
                jst_optimizer.step()
                jst_optimizer.zero_grad()
            else:
                # Accumulate gradients and inside the no sync interface
                with jst_model.no_sync():
                    (
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
                    ) = jst_model(
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

                    total_loss = (
                        asr_loss
                        + spkrecog_loss
                        + aligner_loss
                        + duration_loss
                        + shared_encoder_loss
                        + tts_loss
                    )
                    total_loss = total_loss / config["training"]["grad_accum"]

                    # Only backward the loss
                    total_loss.backward()

            # Loss logging
            epoch_total_loss += total_loss.item()
            epoch_asr_loss += asr_loss.item()
            epoch_spkrecog_loss += spkrecog_loss.item()
            epoch_aligner_loss += aligner_loss.item()
            epoch_duration_loss += duration_loss.item()
            epoch_shared_encoder_loss += shared_encoder_loss.item()
            epoch_tts_loss += tts_loss.item()
            epoch_forwardsum_loss += forwardsum_loss.item()
            epoch_bin_loss += bin_loss.item()
            epoch_before_tts_loss += before_tts_loss.item()
            epoch_after_tts_loss += after_tts_loss.item()

        if rank == 0 and (epoch % config["training"]["checkpoint_epoch"]) == 0:
            checkpoint_dir = os.path.join(
                config["training"]["exp_dir"], f"checkpoint_{epoch}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            jst_model.module.save_model(checkpoint_dir)
            torch.save(
                jst_optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt")
            )

        if rank == 0:
            logger.info(
                f"Epoch: {epoch} -- Total: {epoch_total_loss/len(train_dataloader):.4f} -- ASR: {epoch_asr_loss/len(train_dataloader):.4f} -- SpkRecog: {epoch_spkrecog_loss/len(train_dataloader):.4f} -- Aligner: {epoch_aligner_loss/len(train_dataloader):.4f} -- Duration: {epoch_duration_loss/len(train_dataloader):.4f} -- Shared Encoder: {epoch_shared_encoder_loss/len(train_dataloader):.4f} -- TTS: {epoch_tts_loss/len(train_dataloader):.4f}"
            )

            tb_logger.add_scalar(
                "Loss/Total", epoch_total_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/ASR", epoch_asr_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/SpkRecog", epoch_spkrecog_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/Aligner", epoch_aligner_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/Duration", epoch_duration_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/Shared Encoder",
                epoch_shared_encoder_loss / len(train_dataloader),
                epoch,
            )
            tb_logger.add_scalar(
                "Loss/TTS", epoch_tts_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/ForwardSum", epoch_forwardsum_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/Bin", epoch_bin_loss / len(train_dataloader), epoch
            )
            tb_logger.add_scalar(
                "Loss/Before Postnet",
                epoch_before_tts_loss / len(train_dataloader),
                epoch,
            )
            tb_logger.add_scalar(
                "Loss/After Postnet",
                epoch_after_tts_loss / len(train_dataloader),
                epoch,
            )

        torch.distributed.barrier()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"GPU COUNT: {world_size}")

    config = sys.argv[1]
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    if not config["training"]["continue"]:
        if os.path.exists(config["training"]["exp_dir"]):
            shutil.rmtree(config["training"]["exp_dir"])
    os.makedirs(config["training"]["exp_dir"], exist_ok=True)

    mp.spawn(main, args=(world_size, config), nprocs=world_size)
