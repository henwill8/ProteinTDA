import urllib.request
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from esm.pretrained import load_model_and_alphabet
from minifold.data import data_pipeline, feature_pipeline
from minifold.data.config import NUM_RES
from minifold.data.of_data import of_inference
from minifold.model.model import MiniFoldModel
from minifold.utils.residue_constants import atom_order, restype_order_with_x_inverse
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tmtools import tm_align

from proteintda.config import CONFIG_OF, RUN_CONFIG
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


def _download_checkpoint(cache_dir: Path, model_size: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"minifold_{model_size}.ckpt"
    if not checkpoint.exists():
        print(f"Downloading MiniFold {model_size} weights to {checkpoint}...")
        urllib.request.urlretrieve(MODEL_URLS[model_size], checkpoint)
    return checkpoint


def _as_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)


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
        self._frozen_modules: list[torch.nn.Module] = []

        if train:
            self.prepare_for_training(
                unfreeze_fold_blocks=unfreeze_fold_blocks,
                unfreeze_structure_module=unfreeze_structure_module,
            )
        else:
            self.model.eval()


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
        self._set_training_mode()
    
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
        self._configure_dropout()

    def _unfreeze_enabled_aux_heads(self) -> None:
        if not self.model.use_structure_module:
            return

        loss_cfg = self.config_of.loss
        heads = self.model.aux_heads

        # TM and Experimentally Resolved heads are initialized randomly and need to be trained

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

    def _configure_dropout(self) -> None:
        # TODO: ensure this actually works
        frozen_only = RUN_CONFIG.training.dropout
        self._frozen_modules = []
        for module in self.model.modules():
            params = tuple(module.parameters())
            if not params:
                continue
            if not any(p.requires_grad for p in params):
                self._frozen_modules.append(module)
            elif not frozen_only:
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
                dropout = getattr(module, "dropout", None)
                if isinstance(dropout, float):
                    module.dropout = 0.0
                dropout_prob = getattr(module, "dropout_prob", None)
                if isinstance(dropout_prob, float):
                    module.dropout_prob = 0.0

    def _set_training_mode(self) -> None:
        self.model.train()
        for module in self._frozen_modules:
            module.eval()
    

    def prepare_batch(
        self,
        proteins: SCNProtein | list[SCNProtein],
        *,
        train: bool = False,
    ) -> dict:
        if not isinstance(proteins, list):
            proteins = [proteins]
        singles = [self._prepare_single(protein, train=train) for protein in proteins]
        return self._collate_batches(singles)


    def _prepare_single(self, protein: SCNProtein, *, train: bool = False) -> dict:
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
            coords = _as_tensor(batch_of["all_atom_positions"][:, 0:3, :, 0])
            batch_of = {k: _as_tensor(v) for k, v in batch_of.items()}
        else:
            batch_of = of_inference(seq, "predict", config)
            seq_length = batch_of["seq_length"]
            if isinstance(seq_length, torch.Tensor):
                seq_length = int(seq_length.reshape(-1)[0].item())
            else:
                seq_length = int(seq_length)
            of_seq = "".join(
                restype_order_with_x_inverse[x.item()] for x in batch_of["aatype"]
            )[:seq_length]
            encoded_seq = torch.tensor(self.alphabet.encode(of_seq), dtype=torch.long)
            batch_of = {
                k: _as_tensor(v)
                for k, v in batch_of.items()
                if k in (
                    "aatype",
                    "seq_mask",
                    "residx_atom37_to_atom14",
                    "atom37_atom_exists",
                )
            }
            coords = None

        mask = batch_of["seq_mask"][:, 0].bool()
        single = {
            "seq": encoded_seq,
            "mask": mask,
            "batch_of": batch_of,
            "seq_length": seq_length,
        }
        if coords is not None:
            single["coords"] = coords
        return single

    @staticmethod
    def _pad_residue_tensor(tensor: torch.Tensor, schema: list, target_len: int) -> torch.Tensor:
        res_dims = [i for i, dim in enumerate(schema) if dim == NUM_RES]
        if not res_dims:
            return tensor

        pad_len = target_len - tensor.shape[res_dims[0]]
        if pad_len <= 0:
            return tensor

        ndim = tensor.ndim
        padding = [0, 0] * ndim
        for res_dim in res_dims:
            reverse_idx = ndim - 1 - res_dim
            padding[2 * reverse_idx + 1] = pad_len
        return F.pad(tensor, tuple(padding))

    def _collate_batch_of(self, items: list[dict], max_len: int) -> dict:
        schema = self.config_of.data.common.feat
        keys = items[0].keys()
        collated: dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [item[key] for item in items]
            if key in schema and NUM_RES in schema[key]:
                collated[key] = torch.stack(
                    [
                        self._pad_residue_tensor(tensor, schema[key], max_len)
                        for tensor in tensors
                    ],
                    dim=0,
                )
            else:
                collated[key] = torch.stack(tensors, dim=0)
        return collated

    def _collate_batches(self, singles: list[dict]) -> dict:
        max_len = max(single["seq_length"] for single in singles)
        pad_idx = self.alphabet.padding_idx

        seqs = [
            F.pad(single["seq"], (0, max_len - single["seq"].shape[0]), value=pad_idx)
            if max_len - single["seq"].shape[0] > 0 else single["seq"]
            for single in singles
        ]
        masks = [
            F.pad(single["mask"], (0, max_len - single["mask"].shape[0]), value=False)
            if max_len - single["mask"].shape[0] > 0 else single["mask"]
            for single in singles
        ]
        coords_list = [
            F.pad(single["coords"], (0, 0, 0, 0, 0, max_len - single["coords"].shape[0]))
            if max_len - single["coords"].shape[0] > 0 else single["coords"]
            for single in singles if "coords" in single
        ]

        model_batch = {
            "seq": torch.stack(seqs, dim=0).to(self.device),
            "mask": torch.stack(masks, dim=0).to(self.device),
            "batch_of": {
                k: v.to(self.device) for k, v in self._collate_batch_of(
                    [single["batch_of"] for single in singles],
                    max_len,
                ).items()
            },
        }
   
        if coords_list:
            model_batch["coords"] = torch.stack(coords_list, dim=0).to(self.device)
        return model_batch

    def run_batch(
        self,
        batch: list[SCNProtein],
        loss_fn: MiniFoldLoss | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        num_recycling: int = 0,
        randomize_recycles: bool = False,
        use_amp: bool = False,
        grad_clip_norm: float | None = 1.0,
        backward: bool = False,
        include_loss: bool = True,
        include_metrics: bool = False,
    ) -> tuple[dict[str, float], int]:
        """Single forward pass with an optional backward pass and optionally reports loss and/or metrics."""
        include_loss = include_loss and loss_fn is not None
        proteins = list(batch)
        if not proteins:
            return {}, 0

        if backward:
            self._set_training_mode()
            was_training = True
            optimizer.zero_grad(set_to_none=True)
        else:
            was_training = self.model.training
            self.model.eval()

        totals = defaultdict(float)
        autocast_device = "cuda" if self.device.type == "cuda" else self.device.type
        autocast_enabled = use_amp if backward else True
        grad_context = nullcontext() if backward else torch.no_grad()

        if randomize_recycles and num_recycling > 0:
            recycles = MiniFoldLoss.sample_recycles(num_recycling)
        else:
            recycles = num_recycling

        outputs = None
        try:
            with grad_context, torch.autocast(
                autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                outputs = self._forward_batch(proteins, loss_fn, recycles)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM during MiniFold forward pass; skipping batch.")

        if outputs is None:
            if not backward and was_training:
                self._set_training_mode()
            return dict(totals), 0

        batch_n = outputs.get("batch_size", len(proteins))

        if backward:
            self._apply_gradients(outputs["total"], optimizer, scaler, grad_clip_norm)
        if include_loss:
            for key, value in outputs.get("log", {}).items():
                totals[key] += value * batch_n
        if include_metrics:
            self._accumulate_metrics(totals, proteins, outputs)

        if not backward and was_training:
            self._set_training_mode()

        return dict(totals), batch_n

    def _forward_batch(
        self,
        proteins: list[SCNProtein],
        loss_fn: MiniFoldLoss | None,
        num_recycling: int,
    ) -> dict | None:
        train = loss_fn is not None
        model_batch = self.prepare_batch(proteins, train=train)
        if loss_fn is not None:
            result = loss_fn.compute(self.model, model_batch, num_recycling=num_recycling)
            if result is not None:
                result["batch_size"] = len(proteins)
            return result

        out = self.model(model_batch, num_recycling=num_recycling)
        ca_idx = atom_order["CA"]
        return {
            "plddt": out["plddt"].detach(),
            "pred_ca": out["final_atom_positions"][:, :, ca_idx].detach().float().cpu(),
            "batch_size": len(proteins),
        }

    def _accumulate_metrics(
        self,
        totals: dict[str, float],
        proteins: list[SCNProtein],
        outputs: dict,
    ) -> None:
        plddt = outputs.get("plddt")
        pred_ca = outputs.get("pred_ca")
        for i, protein in enumerate(proteins):
            length = len(str(protein.seq))
            if plddt is not None:
                totals["plddt"] += float(plddt[i, :length].mean().detach().cpu())
            if pred_ca is not None:
                exp_ca = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).cpu().numpy()
                alignment = tm_align(
                    pred_ca[i, :length].numpy(),
                    exp_ca,
                    str(protein.seq),
                    str(protein.seq),
                )
                totals["tm_score"] += alignment.tm_norm_chain2

    def _apply_gradients(
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

    
    @property
    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total


    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
