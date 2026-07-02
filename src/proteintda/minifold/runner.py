import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from esm.pretrained import load_model_and_alphabet
from minifold.data import data_pipeline, feature_pipeline
from minifold.data.of_data import of_inference
from minifold.model.model import MiniFoldModel
from minifold.utils.residue_constants import atom_order, restype_order_with_x_inverse
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tmtools import tm_align

from proteintda.config import CONFIG_OF
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.utils.conversions import (
    SideChainAtom,
    atom_positions_from_sidechainnet,
    sidechainnet_to_atom37,
)

MODEL_URLS = {
    "48L": "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt",
    "12L": "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_12L_final.ckpt",
}


# TODO: these are temporary metrics, remove after
def _grad_norm(model: torch.nn.Module) -> float:
    sq_sum = sum(
        p.grad.detach().float().pow(2).sum()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    if sq_sum == 0:
        return 0.0
    if isinstance(sq_sum, torch.Tensor):
        return float(torch.sqrt(sq_sum))
    return float(sq_sum**0.5)


def _measure_component_grad_norms(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    losses: dict[str, torch.Tensor],
) -> tuple[float, float]:
    fold_norm = 0.0
    topo_norm = 0.0

    loss_fold = losses.get("loss_fold")
    if loss_fold is not None and loss_fold.requires_grad:
        optimizer.zero_grad(set_to_none=True)
        loss_fold.backward(retain_graph=True)
        fold_norm = _grad_norm(model)
        optimizer.zero_grad(set_to_none=True)

    loss_topo = losses.get("loss_topo")
    if loss_topo is not None and loss_topo.requires_grad:
        loss_topo.backward(retain_graph=True)
        topo_norm = _grad_norm(model)
        optimizer.zero_grad(set_to_none=True)

    return fold_norm, topo_norm


def _download_checkpoint(cache_dir: Path, model_size: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"minifold_{model_size}.ckpt"
    if not checkpoint.exists():
        print(f"Downloading MiniFold {model_size} weights to {checkpoint}...")
        urllib.request.urlretrieve(MODEL_URLS[model_size], checkpoint)
    return checkpoint


class MiniFoldRunner:
    def __init__(
        self,
        cache_dir: Path,
        *,
        model_size: str = "48L",
        device: torch.device | None = None,
        train: bool = False,
        kernels: bool = False,
        compile: bool = False,
        unfreeze_fold_blocks: int = 0,
        unfreeze_structure_module: bool = True,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading MiniFold ({model_size}) on {device}...")

        checkpoint = _download_checkpoint(cache_dir, model_size)
        torch.hub.set_dir(cache_dir)

        if kernels:
            torch._dynamo.config.cache_size_limit = 64

        ckpt = torch.load(checkpoint, map_location="cpu")
        hparams = ckpt["hyper_parameters"]
        model = MiniFoldModel(
            esm_model_name=hparams["esm_model_name"],
            num_blocks=hparams["num_blocks"],
            no_bins=hparams["no_bins"],
            config_of=CONFIG_OF,
            use_structure_module=True, # Note: They only used structure module in second stage
            kernels=kernels,
        )
        _, alphabet = load_model_and_alphabet(hparams["esm_model_name"])

        state_dict = ckpt["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
        state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        if compile:
            model.fold.miniformer = torch.compile(
                model.fold.miniformer,
                dynamic=True,
                fullgraph=True,
            )

        self.alphabet = alphabet
        self.model = model.to(device)
        self.config_of = CONFIG_OF
        self.device = device
        self.cache_dir = cache_dir
        self.model_size = model_size

        if train:
            self.prepare_for_training(
                unfreeze_fold_blocks=unfreeze_fold_blocks,
                unfreeze_structure_module=unfreeze_structure_module,
            )
        else:
            self.model.eval()

    def _register_distogram_bins(self, *, max_dist: float = 25.0, no_bins: int | None = None) -> None:
        """Adapted from minifold.train.model.MiniFold.__init__."""
        if no_bins is None:
            no_bins = self.model.fold.disto_bins
        device = next(self.model.parameters()).device
        boundaries = torch.linspace(2, max_dist, no_bins - 1, device=device)
        lower = torch.tensor([1.0], device=device)
        upper = torch.tensor([max_dist + 5.0], device=device)
        exp_boundaries = torch.cat((lower, boundaries, upper))
        mid_points = (exp_boundaries[:-1] + exp_boundaries[1:]) / 2
        self.model.register_buffer("boundaries", boundaries)
        self.model.register_buffer("mid_points", mid_points)

    def prepare_for_training(
        self,
        *,
        unfreeze_fold_blocks: int = 0,
        unfreeze_structure_module: bool = True,
    ) -> None:
        self._register_distogram_bins(no_bins=self.model.fold.disto_bins)
        self._configure_finetuning(
            unfreeze_fold_blocks=unfreeze_fold_blocks,
            unfreeze_structure_module=unfreeze_structure_module,
        )
        self.model.train()

    def _configure_finetuning(
        self,
        *,
        unfreeze_fold_blocks: int = 0,
        unfreeze_structure_module: bool = True,
    ) -> None:
        model = self.model
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc_s.parameters():
            param.requires_grad = True
        for param in model.fc_z.parameters():
            param.requires_grad = True
        for param in model.fold.fc_out.parameters():
            param.requires_grad = True
        for param in model.fold.recycle.parameters():
            param.requires_grad = True

        if unfreeze_fold_blocks > 0:
            for block in model.fold.miniformer.blocks[-unfreeze_fold_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        if unfreeze_structure_module and model.use_structure_module:
            for module in (model.structure_module, model.sz_project):
                for param in module.parameters():
                    param.requires_grad = True
            for param in model.aux_heads.plddt.parameters():
                param.requires_grad = True

        self._unfreeze_enabled_aux_heads()

    def _unfreeze_enabled_aux_heads(self) -> None:
        if not self.model.use_structure_module:
            return

        loss_cfg = self.config_of.loss
        heads = self.model.aux_heads

        if loss_cfg.experimentally_resolved.weight != 0.0 and hasattr(heads, "experimentally_resolved"):
            for param in heads.experimentally_resolved.parameters():
                param.requires_grad = True

        if (
            loss_cfg.tm.enabled
            and loss_cfg.tm.weight != 0.0
            and getattr(heads, "tm_enabled", False)
            and hasattr(heads, "tm")
        ):
            for param in heads.tm.parameters():
                param.requires_grad = True

    @property
    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def prepare_batch(self, protein: SCNProtein, *, train: bool = False) -> dict:
        """Build a batched model input from a SidechainNet protein."""
        seq = str(protein.seq)
        config = self.config_of.data

        if train:
            num_res = len(seq)
            raw_features = {
                **data_pipeline.make_sequence_features(seq, protein.id, num_res),
                **data_pipeline.make_dummy_msa_feats(seq),
            }
            positions, atom_mask, _ = sidechainnet_to_atom37(protein, self.device)
            positions_np = positions.detach().cpu().numpy().astype(np.float32)
            mask_np = atom_mask.detach().cpu().numpy().astype(np.float32)
            raw_features["all_atom_positions"] = positions_np
            raw_features["all_atom_mask"] = mask_np
            raw_features["all_atom_mask_true"] = mask_np.copy()
            raw_features["resolution"] = np.array(
                [float(protein.resolution) if protein.resolution is not None else 0.0],
                dtype=np.float32,
            )
            raw_features["is_distillation"] = np.array(0.0, dtype=np.float32)
            batch_of = feature_pipeline.FeaturePipeline(config).process_features(
                raw_features,
                "train",
            )
            seq_length = batch_of["seq_length"]
            if isinstance(seq_length, torch.Tensor):
                seq_length = int(seq_length.reshape(-1)[0].item())
            else:
                seq_length = int(seq_length)
            aatype = batch_of["aatype"]
            if aatype.ndim > 1:
                aatype = aatype[:, 0]
            of_seq = "".join(
                restype_order_with_x_inverse[x.item()] for x in aatype
            )[:seq_length]
            encoded_seq = torch.tensor(self.alphabet.encode(of_seq), dtype=torch.long)
            coords = batch_of["all_atom_positions"][:, 0:3, :, 0]
        else:
            batch_of = of_inference(seq, "predict", config)
            of_seq = "".join(
                restype_order_with_x_inverse[x.item()] for x in batch_of["aatype"]
            )[: batch_of["seq_length"]]
            encoded_seq = torch.tensor(self.alphabet.encode(of_seq), dtype=torch.long)
            batch_of = {k: v for k, v in batch_of.items() if k in (
                "aatype",
                "seq_mask",
                "residx_atom37_to_atom14",
                "atom37_atom_exists",
            )}
            coords = None

        mask = batch_of["seq_mask"][:, 0].bool()
        model_batch = {
            "seq": encoded_seq.unsqueeze(0).to(self.device),
            "mask": mask.unsqueeze(0).to(self.device),
            "batch_of": {k: v.unsqueeze(0).to(self.device) for k, v in batch_of.items()},
        }
        if coords is not None:
            model_batch["coords"] = coords.unsqueeze(0).to(self.device)
        return model_batch

    @torch.inference_mode()
    def predict(self, protein: SCNProtein, *, num_recycling: int = 3) -> dict[str, torch.Tensor]:
        seq = str(protein.seq)
        model_batch = self.prepare_batch(protein, train=False)

        # ESM backbone applies dropout in training mode, which we don't want for inference
        was_training = self.model.training
        self.model.eval()
        autocast_device = "cuda" if self.device.type == "cuda" else self.device.type
        try:
            with torch.autocast(autocast_device, dtype=torch.bfloat16):
                out = self.model(model_batch, num_recycling=num_recycling)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM during MiniFold forward pass; skipping protein.")
            return None
        finally:
            if was_training:
                self.model.train()

        length = len(seq)
        return {
            "positions": out["final_atom_positions"][0, :length].float().cpu(),
            "atom_mask": out["final_atom_mask"][0, :length].float().cpu(),
            "plddt": out["plddt"][0, :length].float().cpu(),
            "sequence": seq,
        }

    def build_optimizer(self, *, base_lr: float, struct_lr: float) -> torch.optim.Optimizer:
        """Adapted from minifold.train.model.MiniFold.configure_optimizers."""
        model = self.model
        return torch.optim.Adam(
            [
                {
                    "params": [
                        p
                        for name, p in model.named_parameters()
                        if p.requires_grad
                        and ("structure_module" not in name)
                        and ("aux_heads" not in name)
                        and ("sz_project" not in name)
                    ],
                    "lr": base_lr,
                },
                {
                    "params": [
                        p
                        for name, p in model.named_parameters()
                        if p.requires_grad
                        and (
                            ("structure_module" in name)
                            or ("aux_heads" in name)
                            or ("sz_project" in name)
                        )
                    ],
                    "lr": struct_lr,
                },
            ],
            lr=base_lr,
        )

    def training_step(
        self,
        batch,
        optimizer: torch.optim.Optimizer,
        loss_fn: MiniFoldLoss,
        scaler: torch.amp.GradScaler,
        *,
        train_recycles: int | None = None,
        randomize_recycles: bool = True,
        use_amp: bool = False,
        grad_clip_norm: float | None = 1.0,
    ) -> tuple[dict[str, float], int]:
        self.model.train()
        totals = defaultdict(float)
        n = 0

        for protein in batch:
            optimizer.zero_grad(set_to_none=True)
            if randomize_recycles and train_recycles is not None and train_recycles > 0:
                num_recycling = MiniFoldLoss.sample_recycles(train_recycles)
            else:
                num_recycling = train_recycles or 0

            model_batch = self.prepare_batch(protein, train=True)
            autocast_device = "cuda" if self.device.type == "cuda" else self.device.type
            with torch.autocast(autocast_device, dtype=torch.bfloat16, enabled=use_amp):
                losses = loss_fn.compute(self.model, model_batch, num_recycling=num_recycling)

            if losses is None:
                continue

            # TODO: remove this after ensuring that gradients are roughly in the same order of magnitude
            fold_grad_norm, topo_grad_norm = _measure_component_grad_norms(
                self.model,
                optimizer,
                losses,
            )
            totals["fold_grad_norm"] += fold_grad_norm
            totals["topo_grad_norm"] += topo_grad_norm

            if scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                optimizer.step()

            log = losses.get("log", {})
            for key, value in log.items():
                totals[key] += value
            n += 1

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return dict(totals), n

    def evaluation_step(
        self,
        batch,
        *,
        num_recycling: int,
    ) -> tuple[list[float], list[float]]:
        ca_idx = atom_order["CA"]
        plddt_scores: list[float] = []
        tm_scores: list[float] = []

        for protein in batch:
            result = self.predict(protein, num_recycling=num_recycling)
            if result is None:
                continue
            plddt_scores.append(float(result["plddt"].mean()))
            pred_ca = result["positions"][:, ca_idx].numpy()
            exp_ca = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).numpy()
            alignment = tm_align(pred_ca, exp_ca, str(protein.seq), str(protein.seq))
            tm_scores.append(alignment.tm_norm_chain2)

        return plddt_scores, tm_scores

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
