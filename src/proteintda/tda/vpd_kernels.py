import sys
import threading
import time
from pathlib import Path

import torch
from tqdm import tqdm
from vpd import _cpp

from proteintda.config import GraphRepresentation, SamplingMethod, HEAT_RFF_CONFIG, LOSS_CONFIG

_CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "heat_rff"

_GRAPH_REPRESENTATION_CPP = {
    GraphRepresentation.COMPLETE: _cpp.Graph_Representation.COMPLETE,
    GraphRepresentation.LATTICE: _cpp.Graph_Representation.LATTICE,
}

_SAMPLING_METHOD_CPP = {
    SamplingMethod.RANDOM: _cpp.RandomSamplingKernel,
    SamplingMethod.REJECTION: _cpp.RejectionSamplingKernel,
    SamplingMethod.MCMC: lambda: _cpp.MALASamplingKernel(
        sigma=0.1, burn_in=300, thinning=30
    ),
    SamplingMethod.MALA: lambda: _cpp.MALASamplingKernel(
        sigma=0.1, burn_in=300, thinning=30, tune_sigma=True
    ),
}


def _to_cpp_graph(graph_representation_type: GraphRepresentation | _cpp.Graph_Representation):
    if isinstance(graph_representation_type, GraphRepresentation):
        return _GRAPH_REPRESENTATION_CPP[graph_representation_type]
    return graph_representation_type


def _make_sampler(sampling_method: SamplingMethod):
    factory = _SAMPLING_METHOD_CPP[sampling_method]
    return factory()


def _heat_rff_cache_path(n, axis_dim, resolution, R, s, t, seed, sampler: str, graph_representation_type):
    graph_name = graph_representation_type.name.lower()
    return _CACHE_DIR / (
        f"n-{n}_axisdim-{axis_dim}_res-{resolution}_s-{s}_t-{t}_R-{R}_seed-{seed}"
        f"_sampler-{sampler}_graph-{graph_name}.pt"
    )


def _validate_cached_kernel(cached: dict, *, n, axis_dim, resolution, R, seed, graph_representation_type) -> None:
    expected = {
        "n": n,
        "axis_dim": axis_dim,
        "resolution": resolution,
        "R": R,
        "seed": seed,
        "graph_representation_type": graph_representation_type.name,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            raise ValueError(
                f"Heat kernel cache {key} mismatch: file has {cached.get(key)}, expected {value}"
            )


def _format_kernel_config(n, axis_dim, resolution, R, s, t, seed, graph_representation_type) -> str:
    parts = [
        f"n={n}",
        f"R={R}",
        f"axis_dim={axis_dim}",
        f"resolution={resolution}",
        f"s={s}",
        f"t={t}",
        f"seed={seed}",
        f"graph_representation_type={graph_representation_type.name}",
    ]
    return ", ".join(parts)


def _update_sampler_progress(pbar: tqdm, sampler) -> None:
    pbar.set_postfix_str(sampler.progress_postfix(), refresh=False)


def _build_kernel_with_progress(sampler, config_line: str):
    result = {"kernel": None}
    error = {"exc": None}

    def run_build() -> None:
        try:
            result["kernel"] = sampler.build()
        except Exception as exc:
            error["exc"] = exc

    print(config_line)

    thread = threading.Thread(target=run_build, daemon=True)
    thread.start()

    pbar = None
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} est. "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    try:
        while thread.is_alive():
            if sampler.total_ops > 0 and pbar is None:
                pbar = tqdm(
                    total=sampler.total_ops,
                    desc="heat kernel",
                    unit="op",
                    unit_scale=True,
                    smoothing=0.05,
                    bar_format=bar_format,
                )
            if pbar is not None:
                completed = sampler.completed_ops
                total = sampler.total_ops
                if total != pbar.total:
                    pbar.total = total
                delta = completed - pbar.n
                if delta > 0:
                    pbar.update(delta)
                _update_sampler_progress(pbar, sampler)
            time.sleep(0.2)
    except KeyboardInterrupt:
        if pbar is not None:
            pbar.close()
        print("\nKernel build interrupted.", file=sys.stderr)
        raise
    thread.join()

    if pbar is not None:
        delta = sampler.completed_ops - pbar.n
        if delta > 0:
            pbar.update(delta)
        pbar.total = sampler.total_ops
        pbar.refresh()
        pbar.close()

    if error["exc"] is not None:
        raise error["exc"]
    return result["kernel"]


def create_heat_random_fourier_features(
    n,
    axis_dim,
    resolution,
    R=100,
    s=1.0,
    t=1,
    seed=42,
    device=_cpp.Device.CPU,
    sampling_method: SamplingMethod = SamplingMethod.MALA,
    graph_representation_type: GraphRepresentation = GraphRepresentation.LATTICE,
    show_progress=True,
):
    cpp_graph = _to_cpp_graph(graph_representation_type)
    cache_path = _heat_rff_cache_path(
        n, axis_dim, resolution, R, s, t, seed, sampling_method.name, graph_representation_type
    )
    if cache_path.is_file():
        print(f"Loading cached heat kernel from {cache_path}...", flush=True)
        cached = torch.load(cache_path, weights_only=False)
        _validate_cached_kernel(
            cached, n=n, axis_dim=axis_dim, resolution=resolution, R=R, seed=seed,
            graph_representation_type=graph_representation_type,
        )
        kernel = _cpp.Heat_Kernel(
            n, axis_dim, resolution, R, s, t, cached["thetas"], cached["weights"],
        )
        print(f"Loaded heat kernel cache: {cache_path.name}", flush=True)
        return _cpp.VPD(kernel)

    kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, s, t)
    sampler = _make_sampler(sampling_method)
    sampler.init(
        kernel,
        True,
        seed=seed,
        graph_representation_type=cpp_graph,
        device=device,
    )
    if show_progress:
        _build_kernel_with_progress(
            sampler,
            f"Building heat kernel: {_format_kernel_config(n, axis_dim, resolution, R, s, t, seed, graph_representation_type)}",
        )
    else:
        sampler.build()

    vpd = _cpp.VPD(kernel)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": n,
            "axis_dim": axis_dim,
            "resolution": resolution,
            "R": R,
            "seed": seed,
            "graph_representation_type": graph_representation_type.name,
            "thetas": vpd.thetas,
            "weights": vpd.weights,
        },
        cache_path,
    )
    return vpd


def create_vpd_kernels(loss_config, heat_rff_config):
    """Create VPD heat kernels only when the corresponding loss term is enabled."""
    terms = loss_config.tda.terms
    vpd_dims = {
        int(name.rsplit("_h", 1)[1])
        for name in terms
        if name.startswith("vpd_") and terms[name].enabled
    }
    h0rff = None
    if 0 in vpd_dims:
        print("Preparing VPD h0 kernel...", flush=True)
        h0rff = create_heat_random_fourier_features(**heat_rff_config["h0rff"])
    h1rff = None
    if any(dim > 0 for dim in vpd_dims):
        print("Preparing VPD h1 kernel...", flush=True)
        h1rff = create_heat_random_fourier_features(**heat_rff_config["h1rff"])
    return h0rff, h1rff


if __name__ == "__main__":
    print("Creating VPD Kernels")
    print(create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG))
