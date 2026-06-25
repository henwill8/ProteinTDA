import sys
import time
import threading
from pathlib import Path

import torch
from tqdm import tqdm

from config import LOSS_CONFIG, HEAT_RFF_CONFIG
from vpd import _cpp

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "heat_rff"


def _heat_rff_cache_path(n, axis_dim, resolution, R, tau, seed):
    return _CACHE_DIR / (f"n-{n}_axisdim-{axis_dim}_res-{resolution}_tau-{tau}-R-{R}_seed-{seed}.pt")


def _validate_cached_kernel(cached: dict, *, n, axis_dim, resolution, R, seed) -> None:
    expected = {
        "n": n,
        "axis_dim": axis_dim,
        "resolution": resolution,
        "R": R,
        "seed": seed,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            raise ValueError(
                f"Heat kernel cache {key} mismatch: file has {cached.get(key)}, expected {value}"
            )


def _format_kernel_config(n, axis_dim, resolution, R, tau, seed) -> str:
    parts = [
        f"n={n}",
        f"R={R}",
        f"axis_dim={axis_dim}",
        f"resolution={resolution}",
        f"tau={tau}",
        f"seed={seed}",
    ]
    return ", ".join(parts)


def _format_acceptance(builder) -> str:
    if builder.attempts_completed <= 0:
        return "n/a"
    rate = f"{builder.acceptance_rate * 100:.1f}%"
    return rate


def _update_kernel_progress(pbar: tqdm, builder) -> None:
    pbar.set_postfix_str(
        f"w={builder.weights_completed}/{builder.total_weights}, a={_format_acceptance(builder)}",
        refresh=False,
    )


def _build_kernel_with_progress(n, axis_dim, resolution, R, tau, seed, progress_batch):
    builder = _cpp.Heat_KernelBuilder(n, axis_dim, resolution, R, tau, seed, progress_batch)
    error = {"exc": None}

    def run_build() -> None:
        try:
            builder.build()
        except Exception as exc:
            error["exc"] = exc

    print(_format_kernel_config(n, axis_dim, resolution, R, tau, seed))

    thread = threading.Thread(target=run_build, daemon=True)
    thread.start()

    pbar = None
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} est. "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    try:
        while thread.is_alive():
            if builder.total_ops > 0 and pbar is None:
                pbar = tqdm(
                    total=builder.total_ops,
                    desc="heat kernel",
                    unit="op",
                    unit_scale=True,
                    smoothing=0.05,
                    bar_format=bar_format,
                )
            if pbar is not None:
                completed = builder.completed_ops
                total = builder.total_ops
                if total != pbar.total:
                    pbar.total = total
                delta = completed - pbar.n
                if delta > 0:
                    pbar.update(delta)
                _update_kernel_progress(pbar, builder)
            time.sleep(0.2)
    except KeyboardInterrupt:
        if pbar is not None:
            pbar.close()
        print("\nKernel build interrupted.", file=sys.stderr)
        raise
    thread.join()

    if pbar is not None:
        delta = builder.completed_ops - pbar.n
        if delta > 0:
            pbar.update(delta)
        pbar.total = builder.total_ops
        pbar.refresh()
        pbar.close()

    if error["exc"] is not None:
        raise error["exc"]
    return builder.kernel()


def create_heat_random_fourier_features(
    n, axis_dim, resolution, R=100, tau=1, seed=42, show_progress=True, progress_batch=100,
):
    cache_path = _heat_rff_cache_path(n, axis_dim, resolution, R, tau, seed)
    if cache_path.is_file():
        cached = torch.load(cache_path, weights_only=False)
        _validate_cached_kernel(
            cached, n=n, axis_dim=axis_dim, resolution=resolution, R=R, seed=seed
        )
        kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, tau, cached["thetas"], cached["weights"])
        return _cpp.VPD(kernel)

    if show_progress:
        kernel = _build_kernel_with_progress(
            n, axis_dim, resolution, R, tau, seed, progress_batch,
        )
    else:
        kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, tau, seed)

    vpd = _cpp.VPD(kernel)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": n,
            "axis_dim": axis_dim,
            "resolution": resolution,
            "R": R,
            "seed": seed,
            "thetas": vpd.thetas,
            "weights": vpd.weights,
        },
        cache_path,
    )
    return vpd


def create_vpd_kernels(loss_config, heat_rff_config):
    """Create VPD heat kernels only when the corresponding loss term is enabled."""
    h0rff = (
        create_heat_random_fourier_features(**heat_rff_config["h0rff"])
        if loss_config.vpd_h0.enabled
        else None
    )
    h1rff = (
        create_heat_random_fourier_features(**heat_rff_config["h1rff"])
        if loss_config.vpd_h1.enabled
        else None
    )
    return h0rff, h1rff


if __name__ == "__main__":
    print("Creating VPD Kernels")
    print(create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG))
