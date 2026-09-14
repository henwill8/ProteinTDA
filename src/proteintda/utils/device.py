import torch

from proteintda.config import RUN_CONFIG


def resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"RUN_CONFIG.runtime.device={device!r} but CUDA is unavailable")
        index = resolved.index if resolved.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"RUN_CONFIG.runtime.device={device!r} but only "
                f"{torch.cuda.device_count()} GPU(s) are visible"
            )
        torch.cuda.set_device(index)
        resolved = torch.device("cuda", index)
    return resolved
