import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from minifold.utils.residue_constants import atom_order
from minifold.utils.tensor_utils import tensor_tree_map

from proteintda.config import LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.pipeline import build_loss_fn
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph, wasserstein_distance
from proteintda.utils.dataset import load_dataset, set_seed

BATCH_SIZES = [1, 2, 4, 8]
WARMUP_STEPS = 2
TIMED_STEPS = 8
TIMED_BATCHES = 4

PROFILE_KEYS = (
    "prepare_batch",
    "model_forward",
    "distogram_loss",
    "structure_loss",
    "tda_distance_matrix",
    "persistence_diagram",
    "wasserstein",
    "vpd",
    "metrics",
    "backward",
    "optimizer",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release_device_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


@dataclass
class ProfileTotals:
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    steps: int = 0
    proteins: int = 0

    def add(self, other: "ProfileTotals") -> None:
        self.steps += other.steps
        self.proteins += other.proteins
        for key, value in other.seconds.items():
            self.seconds[key] += value

    def per_step_ms(self, key: str) -> float:
        if self.steps == 0:
            return 0.0
        return 1000.0 * self.seconds[key] / self.steps

    def per_protein_ms(self, key: str) -> float:
        if self.proteins == 0:
            return 0.0
        return 1000.0 * self.seconds[key] / self.proteins

    def total_per_step_ms(self) -> float:
        return sum(self.per_step_ms(key) for key in PROFILE_KEYS)


class _BatchTimer:
    def __init__(self, totals: ProfileTotals, device: torch.device) -> None:
        self.totals = totals
        self.device = device

    @contextmanager
    def section(self, name: str):
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.totals.seconds[name] += time.perf_counter() - t0


def _profile_tda_terms(tda_loss, pred_adj, target_adj, timer: _BatchTimer):
    cfg = tda_loss.config
    hom_dim = cfg.pd.hom_dim

    with timer.section("persistence_diagram"):
        target_diags = pd_from_graph(target_adj, **cfg.pd)
        pred_diags = pd_from_graph(pred_adj, **cfg.pd)

    terms: dict[str, torch.Tensor] = {}
    enabled = tda_loss._enabled

    if "wasserstein_h0" in enabled or "wasserstein_h1" in enabled:
        with timer.section("wasserstein"):
            wasserstein = wasserstein_distance(pred_diags, target_diags, hom_dim)
        for i, name in enumerate(("wasserstein_h0", "wasserstein_h1")):
            if name in enabled and i < len(wasserstein):
                terms[name] = wasserstein[i]

    if "vpd_h0" in enabled or "vpd_h1" in enabled:
        with timer.section("vpd"):
            if "vpd_h0" in enabled:
                terms["vpd_h0"] = tda_loss.h0rff.vpd_loss(pred_diags[0], target_diags[0])
            if "vpd_h1" in enabled:
                terms["vpd_h1"] = tda_loss.h1rff.vpd_loss(pred_diags[1], target_diags[1])

    return terms


def _profile_batch(
    runner: MiniFoldRunner,
    proteins: list,
    loss_fn: MiniFoldLoss,
    *,
    device: torch.device,
    training,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    backward: bool,
    include_metrics: bool,
) -> ProfileTotals:
    totals = ProfileTotals()
    if not proteins:
        return totals

    timer = _BatchTimer(totals, device)
    use_amp = training.amp and device.type == "cuda"
    grad_context = nullcontext() if backward else torch.no_grad()

    if backward:
        runner._set_training_mode()
        optimizer.zero_grad(set_to_none=True)
    else:
        runner.model.eval()

    recycles = training.train_recycles
    autocast_device = "cuda" if device.type == "cuda" else device.type

    with grad_context, torch.autocast(
        autocast_device, dtype=torch.bfloat16, enabled=use_amp if backward else True
    ):
        with timer.section("prepare_batch"):
            model_batch = runner.prepare_batch(proteins, train=True)

        with timer.section("model_forward"):
            r_dict = runner.model(model_batch, num_recycling=recycles)

        preds = r_dict["preds"]
        total = preds.new_zeros(())

        if loss_fn.loss_config.distogram.enabled:
            with timer.section("distogram_loss"):
                disto_loss = MiniFoldLoss._distogram_loss(
                    preds,
                    model_batch["coords"],
                    model_batch["mask"],
                    runner.model.boundaries,
                    no_bins=preds.shape[-1],
                )
                total = total + loss_fn.loss_config.distogram.weight * disto_loss

        needs_structure = (
            loss_fn.loss_config.structure.enabled
            or (loss_fn.tda_enabled and loss_fn.loss_config.tda.enabled)
        )
        batch_of = None
        if needs_structure:
            batch_of = tensor_tree_map(lambda t: t[..., -1], model_batch["batch_of"])

            if loss_fn.loss_config.structure.enabled:
                with timer.section("structure_loss"):
                    struct_loss, _ = loss_fn.structure_loss(
                        r_dict, batch_of, _return_breakdown=True,
                    )
                    total = total + loss_fn.loss_config.structure.weight * struct_loss

            if loss_fn.tda_enabled and loss_fn.loss_config.tda.enabled:
                tda = loss_fn._tda
                tda_losses = []
                for index in range(len(proteins)):
                    with timer.section("tda_distance_matrix"):
                        pred_adj, target_adj = tda._create_adjs(r_dict, batch_of, index)
                    terms = _profile_tda_terms(tda, pred_adj, target_adj, timer)
                    ref = pred_adj
                    loss_i = ref.new_zeros(())
                    for name, loss in terms.items():
                        loss_i = loss_i + tda.config[name].weight * loss
                    tda_losses.append(loss_i)
                total = total + loss_fn.loss_config.tda.weight * torch.stack(tda_losses).mean()

        if include_metrics:
            with timer.section("metrics"):
                outputs = {
                    "plddt": r_dict.get("plddt"),
                    "pred_ca": None,
                }
                if "final_atom_positions" in r_dict:
                    ca_idx = atom_order["CA"]
                    outputs["pred_ca"] = (
                        r_dict["final_atom_positions"][:, :, ca_idx].detach().float().cpu()
                    )
                metric_totals: dict[str, float] = defaultdict(float)
                runner._accumulate_metrics(metric_totals, proteins, outputs)

    if backward and optimizer is not None:
        with timer.section("backward"):
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()
        with timer.section("optimizer"):
            trainable = [p for p in runner.model.parameters() if p.requires_grad]
            if scaler is not None and scaler.is_enabled():
                if training.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, training.grad_clip_norm)
                optimizer.step()

    totals.steps = 1
    totals.proteins = len(proteins)
    _release_device_memory(device)
    return totals


def _pick_proteins(proteins, target_length):
    def _sort_key(protein):
        if target_length is None:
            return (0, protein.id)
        return (abs(len(protein.seq) - target_length), protein.id)

    return sorted(proteins, key=_sort_key)


def _make_batches(proteins, batch_size):
    return [
        proteins[i : i + batch_size]
        for i in range(0, len(proteins), batch_size)
    ]


def _profile_subset(proteins: list, batch_size: int) -> list:
    limit = batch_size * TIMED_BATCHES
    return proteins[: min(len(proteins), limit)]


def _profile_batch_size(
    runner,
    proteins,
    loss_fn,
    batch_size: int,
    device: torch.device,
) -> ProfileTotals | None:
    training = RUN_CONFIG.training
    subset = _profile_subset(proteins, batch_size)
    batches = _make_batches(subset, batch_size)
    if not batches:
        raise ValueError("no batches to profile")

    optimizer = torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    use_amp = training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_pass() -> ProfileTotals:
        pass_totals = ProfileTotals()
        for batch in batches:
            try:
                pass_totals.add(
                    _profile_batch(
                        runner,
                        batch,
                        loss_fn,
                        device=device,
                        training=training,
                        optimizer=optimizer,
                        scaler=scaler,
                        backward=True,
                        include_metrics=True,
                    )
                )
            except torch.cuda.OutOfMemoryError:
                _release_device_memory(device)
                print(f"  OOM on batch_size={batch_size} ({len(batch)} proteins); skipping batch")
        return pass_totals

    for _ in range(WARMUP_STEPS):
        run_pass()
        _release_device_memory(device)

    aggregate = ProfileTotals()
    for _ in range(TIMED_STEPS):
        aggregate.add(run_pass())
        _release_device_memory(device)

    if aggregate.steps == 0:
        return None
    return aggregate


def _print_batch_report(batch_size: int, totals: ProfileTotals) -> None:
    total_ms = totals.total_per_step_ms()
    print(f"\nbatch_size={batch_size}  steps={totals.steps}  proteins={totals.proteins}")
    print(f"{'section':<22} {'ms/step':>10} {'ms/protein':>12} {'%':>7}")
    for key in PROFILE_KEYS:
        ms_step = totals.per_step_ms(key)
        if ms_step <= 0:
            continue
        pct = 100.0 * ms_step / total_ms if total_ms > 0 else 0.0
        print(
            f"{key:<22} {ms_step:>10.2f} {totals.per_protein_ms(key):>12.2f} {pct:>6.1f}"
        )
    print(f"{'TOTAL':<22} {total_ms:>10.2f} {1000.0 * sum(totals.seconds.values()) / totals.proteins:>12.2f} {100.0:>6.1f}")


def _resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _warm_prepare_cache(runner, proteins):
    for protein in proteins:
        runner.prepare_batch([protein], train=True)


def _profile_protein_pool(proteins: list) -> list:
    max_batch = max(BATCH_SIZES)
    return proteins[: min(len(proteins), max_batch * TIMED_BATCHES)]


def main():
    data = RUN_CONFIG.data
    training = RUN_CONFIG.training
    runtime = RUN_CONFIG.runtime
    set_seed(training.seed)
    device = _resolve_device()

    proteins = _pick_proteins(load_dataset(), data.max_protein_length)
    if not proteins:
        raise ValueError("dataset is empty")

    profile_proteins = _profile_protein_pool(proteins)

    print(
        f"Pipeline profile: {len(profile_proteins)}/{len(proteins)} proteins, lengths "
        f"{min(len(p.seq) for p in profile_proteins)}-"
        f"{max(len(p.seq) for p in profile_proteins)}, "
        f"{device}, model={runtime.model_size}"
    )
    print(
        f"training: batch_sizes={BATCH_SIZES}, timed_batches={TIMED_BATCHES}, "
        f"warmup={WARMUP_STEPS}, timed_passes={TIMED_STEPS}, "
        f"train_recycles={training.train_recycles}, amp={training.amp}"
    )
    enabled_losses = [
        name
        for name in ("distogram", "structure", "tda")
        if LOSS_CONFIG[name].enabled
    ]
    if LOSS_CONFIG.tda.enabled:
        enabled_losses.extend(
            term
            for term in ("wasserstein_h0", "wasserstein_h1", "vpd_h0", "vpd_h1")
            if LOSS_CONFIG[term].enabled
        )
    print(f"loss terms: {', '.join(enabled_losses)}")

    loss_fn = build_loss_fn()
    runner = MiniFoldRunner(
        Path(runtime.minifold_cache_dir),
        model_size=runtime.model_size,
        device=device,
        train=True,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
    )
    _warm_prepare_cache(runner, profile_proteins)

    for batch_size in BATCH_SIZES:
        if batch_size > len(profile_proteins):
            continue
        totals = _profile_batch_size(runner, profile_proteins, loss_fn, batch_size, device)
        if totals is None:
            print(f"\nbatch_size={batch_size}  skipped (OOM on all batches)")
            continue
        _print_batch_report(batch_size, totals)
        _release_device_memory(device)


if __name__ == "__main__":
    main()
