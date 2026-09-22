from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from torch_geometric.data import Data

from proteintda.config import RUN_CONFIG
from proteintda.lightrosetta.features import protein_to_lightrosetta_data
from proteintda.lightrosetta.loss import LightRoseTTALoss
from proteintda.lightrosetta.path import ensure_lightrosetta_on_path
from proteintda.shared.runner import BaseRunner

ensure_lightrosetta_on_path()

from model.LightRoseTTA import Predict_Network


def default_model_args(device: torch.device | str, overrides: dict | None = None) -> SimpleNamespace:
    args = SimpleNamespace(
        device=str(device),
        nhid=64,
        batch_size=1,
        lr=5e-4,
        weight_decay=5e-4,
        dropout_ratio=0.5,
        num_layers=2,
        num_degrees=2,
        num_channels=8,
        num_nlayers=0,
        fully_connected=False,
        div=2,
        pooling="none",
        head=1,
        si_m="1x1",
        si_e="att",
        l0_in_feat=16,
        l0_out_feat=16,
        l1_in_feat=3,
        l1_out_feat=3,
        edge_feat_dim=16,
        n_module=4,
        n_module_str=1,
        n_layer=1,
        d_msa=32,
        d_pair=32,
        d_templ=32,
        edge_d_pair=130,
        n_head_msa=2,
        n_head_pair=2,
        n_head_templ=2,
        d_hidden=32,
        r_ff=4,
        n_resblock=1,
        p_drop=0.1,
        use_templ=True,
        performer_N_opts={"nb_features": 8},
        performer_L_opts={"nb_features": 8},
        num_features=43,
        num_classes=3,
        num_bonds=0,
        seed=42,
    )
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)
    return args


class LightRoseTTARunner(BaseRunner):
    def __init__(
        self,
        device: torch.device,
        *,
        train: bool = False,
        model_overrides: dict | None = None,
        **_ignored,
    ):
        self.device = device
        overrides = dict(RUN_CONFIG.lightrosetta.model)
        if model_overrides:
            overrides.update(model_overrides)
        self.model = Predict_Network(default_model_args(device, overrides)).to(device)
        self._data_cache: dict[str, Data] = {}
        if train:
            self.model.train()
        else:
            self.model.eval()

    def _get_data(self, protein: SCNProtein) -> Data:
        key = str(getattr(protein, "id", id(protein)))
        cached = self._data_cache.get(key)
        if cached is None:
            cached = protein_to_lightrosetta_data(protein)
            self._data_cache[key] = cached
        return cached.clone().to(self.device)

    def run_batch(
        self,
        batch: list[SCNProtein],
        loss_fn: LightRoseTTALoss | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        grad_clip_norm: float | None = 1.0,
        backward: bool = False,
        include_loss: bool = True,
        include_metrics: bool = False,
        epoch: int = 0,
        **_ignored,
    ) -> tuple[dict[str, float], int]:
        include_loss = include_loss and loss_fn is not None
        proteins = list(batch)
        if not proteins:
            return {}, 0

        if backward:
            self.model.train()
            was_training = True
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        else:
            was_training = self.model.training
            self.model.eval()

        totals = defaultdict(float)
        n = 0
        grad_context = nullcontext() if backward else torch.no_grad()

        for protein in proteins:
            try:
                data = self._get_data(protein)
            except ValueError as exc:
                print(f"Skipping {getattr(protein, 'id', '?')}: feature build failed ({exc})")
                continue

            xyz = lddt_pred = logits = result_total = None
            result_log: dict[str, float] = {}
            try:
                with grad_context:
                    xyz, lddt_pred, logits = self.model(data, test_flag=not backward)
                    if include_loss and loss_fn is not None:
                        result_total, result_log = loss_fn.compute(
                            xyz, lddt_pred, logits, data, device=self.device, epoch=epoch
                        )
                if backward and result_total is not None:
                    self.apply_gradients(result_total, optimizer, scaler, grad_clip_norm)
                    optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OOM on {getattr(protein, 'id', '?')}; skipping.")
                continue

            if include_loss:
                for key, value in result_log.items():
                    totals[key] += float(value)

            if include_metrics:
                totals["plddt"] += float(lddt_pred.detach().float().mean().cpu())
                pred_ca = xyz.detach().float().cpu().numpy().reshape(-1, 3)[1::3]
                true_ca = data.ca_coords.detach().cpu().numpy()
                totals["tm_score"] += self.tm_score(pred_ca, true_ca, data.seq)

            del data, xyz, lddt_pred, logits, result_total
            n += 1

        if not backward and was_training:
            self.model.train()

        return dict(totals), n
