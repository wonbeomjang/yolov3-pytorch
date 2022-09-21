from argparse import Namespace
from typing import Dict, Any, Optional, Union
import os

import pkg_resources as pkg
import torch
from torch import nn, optim, Tensor


def _load_wandb() -> object:
    try:
        import wandb

        assert hasattr(wandb, '__version__')
        if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2'):
            try:
                wandb_login_success = wandb.login(timeout=30)
            except wandb.errors.UsageError:
                wandb_login_success = False
            if not wandb_login_success:
                wandb = None
    except (ImportError, AssertionError):
        wandb = None

    return wandb


def _get_checkpoint_dir(checkpoint_dir: str, weights: Optional[str] = None, resume: bool = False) -> str:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if weights is not None:
        run_dir = os.path.split(weights)[0]
    else:
        exp_dir = os.listdir(checkpoint_dir)
        run_dir = os.path.join(checkpoint_dir, f"exp{len(exp_dir) if resume else len(exp_dir) + 1}")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return run_dir


class Logger:
    def __init__(self, project: str, config: Namespace = None, weights: Optional[str] = None, resume: bool = False):

        self.metrix = float('inf')
        self.log_dict = {}
        self.checkpoint_dir = _get_checkpoint_dir(config.checkpoint_dir, weights, resume)
        config.checkpoint_dir = self.checkpoint_dir
        self.wandb = _load_wandb()
        self.wandb_run = self._get_wandb_run(project, config, weights, resume)

    def _resume_from_run_id(self, run_id: str):
        run = self.wandb.init()
        artifact = run.use_artifact('jtiger958/kilter-gallery/1i18ukrj_model:v0', type='model')
        artifact_dir = artifact.download()

    def _get_wandb_run(self, project: str, config: Namespace, weights: Optional[str] = None, resume: bool = False):
        if self.wandb:
            if weights is not None:
                wandb_id = torch.load(weights, map_location="cpu").get("wandb_id")
            else:
                wandb_id = torch.load(os.path.join(self.checkpoint_dir, "last.pt"), map_location="cpu").get("wandb_id")\
                    if resume else None
            wandb_run = self.wandb.init(project=project, dir=self.checkpoint_dir, id=wandb_id, resume="allow")
            wandb_run.config.update(config, allow_val_change=True)

            return wandb_run
        return None

    def log(self, log_dict: Dict[str, Any]):
        if self.wandb:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def log_image(self, image: Tensor):
        self.log_dict["image"] = self.wandb.Image(image)

    def end_epoch(self):
        if self.wandb:
            self.wandb.log(self.log_dict)

    def finish(self):
        if self.wandb:
            self.log_artifact()
            self.wandb.finish()

    def log_artifact(self):
        if self.wandb:
            artifact = self.wandb.Artifact(f"{self.wandb_run.id}_model", type="model")
            artifact.add_file(os.path.join(self.checkpoint_dir, "last.pt"), name="last.pt")
            artifact.add_file(os.path.join(self.checkpoint_dir, "best.pt"), name="best.pt")
            self.wandb_run.log_artifact(artifact)

    def save_model(self, net: nn.Module, optimizer: optim.Optimizer, lr_scheduler: Any, epoch: int,
                   metrix: Optional[float] = None):
        save_info = {"state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "metrix": self.metrix}

        if self.wandb:
            save_info["wandb_id"] = self.wandb_run.id

        torch.save(save_info, os.path.join(self.checkpoint_dir, "last.pt"))

        if metrix < self.metrix:
            self.metrix = metrix
            torch.save(save_info, os.path.join(self.checkpoint_dir, "best.pt"))

    def load_state_dict(self, weight: Optional[str] = None, best: bool = False,
                        map_location: Union[torch.device, str] = "cpu"):
        if weight:
            torch.load(weight, map_location=map_location)
        if self.wandb_run:
            return torch.load(os.path.join(self.checkpoint_dir, "best.pt" if best else "last.pt"), map_location=map_location)
        return None

# 임종한
