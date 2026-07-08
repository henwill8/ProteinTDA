import time
from pathlib import Path

import ml_collections as mlc
import torch

from proteintda.config import CONFIG_OF, HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.pipeline import build_loss_fn
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.vpd_kernels import create_vpd_kernels
from proteintda.utils.dataset import load_proteins, make_loader, set_seed

MATCH_PIPELINE = True
N_PROTEINS = 100
BATCH_SIZES = [1, 2, 4, 8]
WARMUP_STEPS = 2
TIMED_STEPS = 16
TARGET_LENGTH = 100

CASP_VERSION = None
SCN_DIR = None
CASP_THINNING = None
MAX_PROTEIN_LENGTH = TARGET_LENGTH + 20
ALLOW_INCOMPLETE = None

CACHE_DIR = Path("cache/minifold")
MODEL_SIZE = "12L"
SEED = 42
LR = 1e-5
WEIGHT_DECAY = 0.01
TRAIN_RECYCLES = 3
INFER_RECYCLES = 3
RANDOMIZE_RECYCLES = True
AMP = True
GRAD_CLIP_NORM = 1.0
LENGTH_BUCKETING = True
LENGTH_BUCKET_SIZE = 10
UNFREEZE_FOLD_BLOCKS = 0
UNFREEZE_STRUCTURE_MODULE = True
DEVICE = None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pick_proteins(proteins, n, target_len):
    ranked = sorted(proteins, key=lambda p: abs(len(p.seq) - target_len))
    return ranked[:n]


def _load_proteins():
    data = RUN_CONFIG.data
    return load_proteins(
        casp_version=data.casp_version if CASP_VERSION is None else CASP_VERSION,
        scn_dir=data.scn_dir if SCN_DIR is None else SCN_DIR,
        casp_thinning=data.casp_thinning if CASP_THINNING is None else CASP_THINNING,
        max_protein_length=(
            data.max_protein_length if MAX_PROTEIN_LENGTH is None else MAX_PROTEIN_LENGTH
        ),
        allow_incomplete=(
            data.allow_incomplete if ALLOW_INCOMPLETE is None else ALLOW_INCOMPLETE
        ),
        max_proteins=None,
    )


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


def _make_runner(device):
    return MiniFoldRunner(
        CACHE_DIR,
        model_size=MODEL_SIZE,
        device=device,
        train=True,
        unfreeze_fold_blocks=UNFREEZE_FOLD_BLOCKS,
        unfreeze_structure_module=UNFREEZE_STRUCTURE_MODULE,
    )


def _make_optimizer(runner):
    return torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


def _make_loader(proteins, batch_size, *, shuffle):
    return make_loader(
        proteins,
        batch_size,
        shuffle=shuffle,
        length_bucketing=LENGTH_BUCKETING,
        length_bucket_size=LENGTH_BUCKET_SIZE,
    )


def _time_batches(run_pass, device, *, warmup_steps, timed_steps):
    for _ in range(warmup_steps):
        run_pass()

    _sync(device)
    t0 = time.perf_counter()
    proteins_seen = 0
    steps = 0
    for _ in range(timed_steps):
        batch_proteins, batch_steps = run_pass()
        proteins_seen += batch_proteins
        steps += batch_steps
    _sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed / steps, elapsed / proteins_seen


def _time_microbenchmark(runner, proteins, loss_fn, batch_size, device):
    batches = [
        proteins[i : i + batch_size]
        for i in range(0, len(proteins), batch_size)
    ]
    optimizer = _make_optimizer(runner)
    use_amp = AMP and device.type == "cuda"

    def run_pass():
        proteins_seen = 0
        for batch in batches:
            _, n = runner.run_batch(
                batch,
                loss_fn,
                optimizer=optimizer,
                num_recycling=TRAIN_RECYCLES,
                randomize_recycles=False,
                use_amp=use_amp,
                backward=True,
                include_metrics=False,
            )
            proteins_seen += n
        return proteins_seen, len(batches)

    return _time_batches(
        run_pass,
        device,
        warmup_steps=WARMUP_STEPS,
        timed_steps=TIMED_STEPS,
    )


def _time_pipeline_train(runner, proteins, loss_fn, batch_size, device):
    loader = _make_loader(proteins, batch_size, shuffle=True)
    optimizer = _make_optimizer(runner)
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_pass():
        proteins_seen = 0
        steps = 0
        for batch in loader:
            _, n = runner.run_batch(
                batch,
                loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                num_recycling=TRAIN_RECYCLES,
                randomize_recycles=RANDOMIZE_RECYCLES,
                use_amp=use_amp,
                grad_clip_norm=GRAD_CLIP_NORM,
                backward=True,
                include_loss=True,
                include_metrics=False,
            )
            proteins_seen += n
            steps += 1
        return proteins_seen, steps

    return _time_batches(
        run_pass,
        device,
        warmup_steps=WARMUP_STEPS,
        timed_steps=TIMED_STEPS,
    )


def _time_pipeline_eval(runner, proteins, loss_fn, batch_size, device):
    loader = _make_loader(proteins, batch_size, shuffle=False)

    def run_pass():
        proteins_seen = 0
        steps = 0
        for batch in loader:
            _, n = runner.run_batch(
                batch,
                loss_fn,
                num_recycling=INFER_RECYCLES,
                backward=False,
                include_loss=True,
                include_metrics=True,
            )
            proteins_seen += n
            steps += 1
        return proteins_seen, steps

    return _time_batches(
        run_pass,
        device,
        warmup_steps=WARMUP_STEPS,
        timed_steps=TIMED_STEPS,
    )


def _warm_prepare_cache(runner, proteins):
    for protein in proteins:
        runner.prepare_batch([protein], train=True)


def _run_microbenchmark(runner, proteins, device):
    print(f"\n{'mode':<10} {'bs':>4}  {'s/step':>8}  {'s/protein':>10}")
    for label, loss_fn in [("no_tda", _make_loss(False)), ("with_tda", _make_loss(True))]:
        for batch_size in BATCH_SIZES:
            if batch_size > len(proteins):
                continue
            sec_step, sec_protein = _time_microbenchmark(
                runner, proteins, loss_fn, batch_size, device,
            )
            print(f"{label:<10} {batch_size:>4}  {sec_step:>8.3f}  {sec_protein:>10.3f}")


def _run_pipeline_benchmark(runner, proteins, loss_fn, device):
    print(
        "\nPipeline-matched settings:"
        f" full LOSS_CONFIG, length_bucketing={LENGTH_BUCKETING},"
        f" train_recycles={TRAIN_RECYCLES},"
        f" randomize_recycles={RANDOMIZE_RECYCLES},"
        f" amp={AMP}, grad_clip={GRAD_CLIP_NORM}"
    )
    print(f"\n{'mode':<10} {'bs':>4}  {'s/step':>8}  {'s/protein':>10}")
    for label, timer in [
        ("train", _time_pipeline_train),
        ("eval", _time_pipeline_eval),
    ]:
        for batch_size in BATCH_SIZES:
            if batch_size > len(proteins):
                continue
            sec_step, sec_protein = timer(runner, proteins, loss_fn, batch_size, device)
            print(f"{label:<10} {batch_size:>4}  {sec_step:>8.3f}  {sec_protein:>10.3f}")


def main():
    set_seed(SEED)
    if DEVICE is not None:
        device = torch.device(DEVICE)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proteins = _pick_proteins(_load_proteins(), N_PROTEINS, TARGET_LENGTH)
    print(
        f"{len(proteins)} proteins, lengths "
        f"{min(len(p.seq) for p in proteins)}-{max(len(p.seq) for p in proteins)}, "
        f"{device}, match_pipeline={MATCH_PIPELINE}"
    )

    runner = _make_runner(device)
    _warm_prepare_cache(runner, proteins)

    if MATCH_PIPELINE:
        loss_fn = build_loss_fn()
        _run_pipeline_benchmark(runner, proteins, loss_fn, device)
    else:
        _run_microbenchmark(runner, proteins, device)


if __name__ == "__main__":
    main()
