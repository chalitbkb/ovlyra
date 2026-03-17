import glob
import os
import shutil
import tempfile

import lightning.fabric as lightning_fabric
import torch
import wandb
from absl import logging

from tts.core import constants
from tts.utils import configuration, custom_logging


# TODO: consider loading the tokenizer too to minimize human mistakes.
# TODO: investigate why using this method with DDP reduces VRAM usage a bit.
def load_from_checkpoint(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_file_to_resume_from: str,
    load_full_checkpoint: bool = True,
) -> tuple[torch.nn.Module, custom_logging.Statistics, torch.optim.Optimizer]:
    """Loads the most appropriate checkpoint file and updates model state."""
    checkpoint = {"model": model}
    if load_full_checkpoint:
        checkpoint.update({"optimizer": optimizer, "loss_statistics": {}})

    fabric.load(checkpoint_file_to_resume_from, checkpoint, strict=True)
    statistics = None
    if load_full_checkpoint:
        statistics = custom_logging.Statistics.from_dict(checkpoint["loss_statistics"])

    return model, statistics, optimizer


def _cleanup_old_checkpoints(
    directory: str,
    keep_n: int,
) -> None:
    """Removes old checkpoints, keeping only the most recent `keep_n`.

    This is called BEFORE saving a new checkpoint to free disk space
    for the atomic save mechanism (which temporarily uses 2x space via /tmp).
    We keep `keep_n - 1` so that after the new save, total = keep_n.
    """
    checkpoint_files = [
        f
        for f in os.listdir(directory)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        return

    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Keep at most (keep_n - 1) to leave room for the one we're about to save.
    max_to_keep = max(0, keep_n - 1)
    num_to_remove = len(checkpoint_files) - max_to_keep
    files_to_remove = checkpoint_files[:num_to_remove] if num_to_remove > 0 else []

    for f in files_to_remove:
        filepath = os.path.join(directory, f)
        try:
            file_size_gb = os.path.getsize(filepath) / (1024**3)
            os.remove(filepath)
            logging.info(
                "Pre-save cleanup: removed old checkpoint %s (%.2f GB freed).", f, file_size_gb
            )
        except OSError as e:
            logging.warning("Failed to remove old checkpoint %s: %s", f, e)


def _cleanup_tmp() -> None:
    """Removes stale checkpoint temp files from /tmp to free space."""
    tmp_dir = tempfile.gettempdir()
    for f in glob.glob(os.path.join(tmp_dir, "tmp*")):
        try:
            if os.path.isfile(f):
                size = os.path.getsize(f)
                # Only remove large files (>100MB) that look like stale checkpoints
                if size > 100 * 1024 * 1024:
                    os.remove(f)
                    logging.info(
                        "Cleaned stale tmp file: %s (%.2f GB freed).",
                        os.path.basename(f),
                        size / (1024**3),
                    )
        except OSError:
            pass


def _log_disk_usage(label: str) -> None:
    """Logs current disk usage for debugging space issues."""
    try:
        stat = shutil.disk_usage("/kaggle/working")
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        logging.info(
            "Disk usage [%s]: %.1f GB used / %.1f GB total (%.1f GB free)",
            label,
            used_gb,
            total_gb,
            free_gb,
        )
    except OSError:
        pass  # Not on Kaggle or path doesn't exist


def save_to_checkpoint(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    config: configuration.ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    statistics: custom_logging.Statistics,
    checkpoint_name: str | None = None,
) -> str:
    """Saves the model and training state to a checkpoint.

    IMPORTANT: Old checkpoints are deleted BEFORE saving the new one,
    not after. This is critical because:
    1. Lightning's atomic save writes to /tmp first, then copies to dest
    2. On Kaggle, /tmp and /kaggle/working are on different filesystems
    3. This means the copy phase temporarily uses 2x checkpoint size
    4. If old checkpoints aren't cleaned first, disk runs out during copy
    """
    checkpoint_name = checkpoint_name or f"checkpoint_{statistics.step}.pt"
    checkpoint_file = os.path.join(config.checkpointing.directory, checkpoint_name)

    # === STEP 1: Free disk space BEFORE saving ===
    if fabric.is_global_zero:
        keep_only_last_n_checkpoints = config.checkpointing.keep_only_last_n_checkpoints
        if keep_only_last_n_checkpoints is not None:
            _log_disk_usage("before cleanup")
            _cleanup_old_checkpoints(
                config.checkpointing.directory,
                keep_n=keep_only_last_n_checkpoints,
            )
            _cleanup_tmp()
            _log_disk_usage("after cleanup, before save")

    # Sync all ranks before saving (rank 0 may have deleted files)
    fabric.barrier()

    # === STEP 2: Save the new checkpoint ===
    checkpoint = {
        "model": model,
        "loss_statistics": statistics.as_dict(),
        "optimizer": optimizer,
        "config": config.to_dict(),
    }
    fabric.save(path=checkpoint_file, state=checkpoint)

    if fabric.is_global_zero:
        _log_disk_usage("after save")

    return checkpoint_file


def save_config(
    experiment_config: configuration.ExperimentConfig,
    checkpoint_dir: str,
    use_wandb: bool,
):
    """Saves model config to a file."""
    config_file = os.path.join(checkpoint_dir, constants.CONFIG_FILE_NAME)
    with open(config_file, "w") as f:
        f.write(str(experiment_config))

    # Config might be with new values after first initialized, to ensure
    # consistency, the config here should be updated with wandb.
    #
    # TODO: sweep run set value can be overriden by python's training code
    #       leading to config here and one shown in the W&B UI being different.
    if use_wandb:
        wandb.config.update(experiment_config.to_dict(), allow_val_change=True)
