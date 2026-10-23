"""Base runner interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tmtools import tm_align


class BaseRunner(ABC):
    model: torch.nn.Module
    device: torch.device

    @property
    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def apply_gradients(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        grad_clip_norm: float | None,
    ) -> None:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()

    @staticmethod
    def tm_score(
        pred_ca: np.ndarray,
        true_ca: np.ndarray,
        seq: str,
    ) -> float:
        length = min(len(pred_ca), len(true_ca), len(seq))
        alignment = tm_align(
            pred_ca[:length],
            true_ca[:length],
            seq[:length],
            seq[:length],
        )
        return float(alignment.tm_norm_chain2)

    @staticmethod
    def lddt_score(
        pred_ca: np.ndarray,
        true_ca: np.ndarray,
        *,
        cutoff: float = 15.0,
    ) -> float:
        length = min(len(pred_ca), len(true_ca))
        if length < 2:
            return 0.0
        pred = np.asarray(pred_ca[:length], dtype=np.float64)
        true = np.asarray(true_ca[:length], dtype=np.float64)
        d_true = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
        d_pred = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
        mask = (d_true < cutoff) & ~np.eye(length, dtype=bool)
        pair_counts = mask.sum(axis=1)
        included = pair_counts > 0
        if not np.any(included):
            return 0.0
        error = np.abs(d_true - d_pred)
        thresholds = (0.5, 1.0, 2.0, 4.0)
        per_residue = np.zeros(length, dtype=np.float64)
        for threshold in thresholds:
            per_residue += ((error < threshold) & mask).sum(axis=1) / np.maximum(pair_counts, 1)
        per_residue /= len(thresholds)
        return float(per_residue[included].mean())

    @abstractmethod
    def run_batch(
        self,
        batch: list[SCNProtein],
        loss_fn: Any | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        grad_clip_norm: float | None = 1.0,
        backward: bool = False,
        include_loss: bool = True,
        include_metrics: bool = False,
        **kwargs,
    ) -> tuple[dict[str, float], int]:
        """Run one batch; return (metric totals, sample count)."""
