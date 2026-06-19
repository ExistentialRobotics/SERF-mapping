"""Unified logging for training with wandb and tensorboard support."""

import time
from pathlib import Path
from typing import Dict


class TrainLogger:
    """Unified logging for wandb and tensorboard."""

    def __init__(
        self,
        output_dir: Path,
        dataset_name: str,
        config: dict,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None

        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = output_dir / f"runs/{dataset_name}_{timestamp}"

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"[Train] TensorBoard log: {log_dir}")

        self._wandb = None
        if use_wandb:
            import wandb
            self._wandb = wandb
            wandb_config = config.get('wandb', {})
            self._wandb.init(
                project=wandb_config.get('project', 'neural-point-map'),
                name=f"{dataset_name}_{timestamp}",
                config=config,
                dir=str(log_dir) if log_dir.exists() else str(output_dir),
            )

    def log_step(self, step: int, loss: float, cos_sim: float, num_points: int):
        """Log metrics for a training step."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/step", loss, step)

        if self._wandb is not None:
            self._wandb.log({
                "loss/step": loss,
                "cosine_similarity": cos_sim,
                "num_points": num_points,
                "global_step": step,
            })

    def log_epoch(self, epoch: int, avg_loss: float, total_points: int):
        """Log metrics for an epoch."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/epoch", avg_loss, epoch)

        if self._wandb is not None:
            self._wandb.log({
                "loss/epoch": avg_loss,
                "epoch": epoch,
                "total_map_points": total_points,
            })

    def log_contrastive_losses(self, step: int, loss_dict: Dict[str, float]):
        """Log loss components."""
        if self.use_tensorboard and self.tb_writer is not None:
            for key, val in loss_dict.items():
                self.tb_writer.add_scalar(f"Loss/{key}", val, step)

        if self._wandb is not None:
            wandb_dict = {f"loss/{k}": v for k, v in loss_dict.items()}
            wandb_dict["global_step"] = step
            self._wandb.log(wandb_dict)

    def close(self):
        """Clean up logging resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()
