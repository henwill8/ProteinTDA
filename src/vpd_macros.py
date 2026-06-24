import sys
import os
import time
import threading
from pathlib import Path

import torch
from tqdm import tqdm

from config import LOSS_CONFIG, HEAT_RFF_CONFIG
from vpd import _cpp

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "heat_rff"


def _mask_cache_label(mask) -> str:
    if mask is None:
        return "nomask"
    # tbh I don't know what is in mask, if this is a bad way to turn it into a string pls change
    return "mask-" + "-".join(str(i) for i in mask)

def _heat_rff_cache_path(n, axis_dim, resolution, R, tau, mask, seed):
    return _CACHE_DIR / (f"n-{n}_axisdim-{axis_dim}_res-{resolution}_tau-{tau}-R-{R}_seed-{seed}_{_mask_cache_label(mask)}.pt")


def _validate_cached_kernel(cached: dict, *, n, axis_dim, resolution, R, mask, seed) -> None:
    expected = {
        "n": n,
        "axis_dim": axis_dim,
        "resolution": resolution,
        "R": R,
        "mask": mask,
        "seed": seed,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            raise ValueError(
                f"Heat kernel cache {key} mismatch: file has {cached.get(key)}, expected {value}"
            )

def _count_scale(value: int) -> tuple[float, str]:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return 1_000_000_000, "B"
    if abs_value >= 1_000_000:
        return 1_000_000, "M"
    if abs_value >= 1_000:
        return 1_000, "K"
    return 1, ""


def _format_count_pair(completed: int, total: int) -> str:
    scale, suffix = _count_scale(max(completed, total))
    if scale == 1:
        return f"{completed}/{total}"
    return f"{completed / scale:.2f}{suffix}/{total / scale:.2f}{suffix}"


def _update_kernel_progress(pbar: tqdm, builder) -> int:
    completed = builder.completed_ops
    if builder.phase == "thetas":
        done = _format_count_pair(builder.thetas_completed, builder.total_thetas)
        pbar.set_postfix(phase="thetas", done=done, refresh=False)
    elif builder.phase == "lambdas":
        done = _format_count_pair(builder.lambdas_completed, builder.total_lambdas)
        pbar.set_postfix(phase="lambdas", done=done, refresh=False)
    else:
        pbar.set_postfix(phase=builder.phase, refresh=False)
    return completed


def _build_kernel_with_progress(n, axis_dim, resolution, R, tau, mask, seed, label: str, progress_batch: int = 100):
    builder = _cpp.Heat_KernelBuilder(n, axis_dim, resolution, R, tau, mask, seed, progress_batch)
    error = {"exc": None}

    def run_build() -> None:
        try:
            builder.build()
        except Exception as exc:
            error["exc"] = exc

    thread = threading.Thread(target=run_build, daemon=True)
    thread.start()

    pbar = None
    last_ops = 0
    try:
        while thread.is_alive():
            if builder.total_ops > 0 and pbar is None:
                pbar = tqdm(total=builder.total_ops, desc=label, unit="op", unit_scale=True)
            if pbar is not None:
                completed = _update_kernel_progress(pbar, builder)
                if completed > last_ops:
                    pbar.update(completed - last_ops)
                    last_ops = completed
            time.sleep(0.2)
    except KeyboardInterrupt:
        if pbar is not None:
            pbar.close()
        print("\nKernel build interrupted.", file=sys.stderr)
        raise
    thread.join()

    if pbar is not None:
        remaining = builder.total_ops - last_ops
        if remaining > 0:
            pbar.update(remaining)
        pbar.close()

    if error["exc"] is not None:
        raise error["exc"]
    return builder.kernel()


def create_heat_random_fourier_features(
    n, axis_dim, resolution, R=100, tau=1, mask=None, seed=42, show_progress=True, progress_batch=100,
):
    cache_path = _heat_rff_cache_path(n, axis_dim, resolution, R, tau, mask, seed)
    if cache_path.is_file():
        cached = torch.load(cache_path, weights_only=False)
        _validate_cached_kernel(
            cached, n=n, axis_dim=axis_dim, resolution=resolution, R=R, mask=mask, seed=seed
        )
        kernel = _cpp.Heat_Kernel(            n, axis_dim, resolution, R, tau, cached["thetas"], cached["weights"]
        )
        return _cpp.VPD(kernel)

    label = f"n={n} R={R} axis_dim={axis_dim} resolution={resolution}"
    if show_progress:
        kernel = _build_kernel_with_progress(
            n, axis_dim, resolution, R, tau, mask, seed, label, progress_batch,
        )
    else:
        kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, tau, mask, seed)

    vpd = _cpp.VPD(kernel)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": n,
            "axis_dim": axis_dim,
            "resolution": resolution,
            "R": R,
            "mask": mask,
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
    print("Crreating VPD Kernels");
    create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG);
