import torch

from proteintda.config import RUN_CONFIG


def resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
