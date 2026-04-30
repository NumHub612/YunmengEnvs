# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Parse GPU device strings into torch.device objects.
"""
import torch


def parse_gpu(gpu: str | int | None, device: str = "cuda") -> torch.device:
    """Parse single GPU device strings into torch.device object.

    Args:
        gpu (str | int | None): GPU device string or index.
        device (str, optional): Device type. Defaults to "cuda".

    Returns:
        torch.device: Parsed GPU device.
    """
    if device.lower() == "cpu":
        return torch.device("cpu")
    if gpu is None:
        return torch.device("cpu")

    if isinstance(gpu, int):
        gpu = f"cuda:{gpu}"
        return torch.device(gpu)
    elif isinstance(gpu, str):
        gpu = gpu.lower()
        return torch.device(gpu)
    else:
        raise ValueError(f"Invalid GPU specification: {gpu}")
