import time
from pathlib import Path

import ml_collections as mlc
import torch

from proteintda.config import CONFIG_OF, HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.vpd_kernels import create_vpd_kernels
from proteintda.utils.dataset import load_dataset

N_PROTEINS = 100
BATCH_SIZES = [1, 2, 4, 8]
WARMUP_STEPS = 2
TIMED_STEPS = 5
TARGET_LENGTH = 100
RECYCLES = 3
CACHE_DIR = Path("cache/minifold")
MODEL_SIZE = "12L"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pick_proteins(proteins, n, target_len):
    ranked = sorted(proteins, key=lambda p: abs(len(p.seq) - target_len))
    return ranked[:n]


def _make_loss(with_tda):
    cfg = mlc.ConfigDict(LOSS_CONFIG.to_dict())
    cfg.distogram.enabled = False
    cfg.vpd_h0.enabled = False
    cfg.vpd_h1.enabled = False
    cfg.tda.enabled = with_tda
    cfg.wasserstein_h0.enabled = with_tda
    cfg.wasserstein_h1.enabled = with_tda
    cfg.structure.enabled = not with_tda
    h0rff, h1rff = create_vpd_kernels(cfg, HEAT_RFF_CONFIG)
    return MiniFoldLoss(CONFIG_OF, loss_config=cfg, h0rff=h0rff, h1rff=h1rff)


def _time_mode(runner, proteins, loss_fn, batch_size, device):
    batches = [
        proteins[i : i + batch_size]
        for i in range(0, len(proteins), batch_size)
    ]
    optimizer = torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=RUN_CONFIG.training.lr,
    )
    use_amp = RUN_CONFIG.training.amp and device.type == "cuda"

    def run_pass():
        proteins_seen = 0
        for batch in batches:
            _, n = runner.run_batch(
                batch,
                loss_fn,
                optimizer=optimizer,
                num_recycling=RECYCLES,
                randomize_recycles=False,
                use_amp=use_amp,
                backward=True,
                include_metrics=False,
            )
            proteins_seen += n
        return proteins_seen

    for _ in range(WARMUP_STEPS):
        run_pass()

    _sync(device)
    t0 = time.perf_counter()
    proteins_seen = 0
    for _ in range(TIMED_STEPS):
        proteins_seen += run_pass()
    _sync(device)
    elapsed = time.perf_counter() - t0

    steps = TIMED_STEPS * len(batches)
    return elapsed / steps, elapsed / proteins_seen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proteins = _pick_proteins(load_dataset(), N_PROTEINS, TARGET_LENGTH)
    print(f"{len(proteins)} proteins, lengths {min(len(p.seq) for p in proteins)}-{max(len(p.seq) for p in proteins)}, {device}")

    runner = MiniFoldRunner(
        CACHE_DIR,
        model_size=MODEL_SIZE,
        device=device,
        train=True,
        unfreeze_structure_module=True,
    )
    for protein in proteins:
        runner.prepare_batch([protein], train=True)

    print(f"\n{'mode':<10} {'bs':>4}  {'s/step':>8}  {'s/protein':>10}")
    for label, loss_fn in [("no_tda", _make_loss(False)), ("with_tda", _make_loss(True))]:
        for batch_size in BATCH_SIZES:
            if batch_size > len(proteins):
                continue
            sec_step, sec_protein = _time_mode(runner, proteins, loss_fn, batch_size, device)
            print(f"{label:<10} {batch_size:>4}  {sec_step:>8.3f}  {sec_protein:>10.3f}")


if __name__ == "__main__":
    main()
