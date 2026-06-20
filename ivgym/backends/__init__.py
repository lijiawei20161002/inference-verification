from .synthetic import SyntheticBackend  # noqa: F401

# HFGPUBackend / VLLMBackend import torch/transformers/vllm; only available on a
# CUDA host with those installed. They are imported lazily by `make_backend`
# (and can still be imported directly, e.g.
# `from ivgym.backends.hf_gpu import HFGPUBackend`).

# Names the CLI / `make_backend` understand. The synthetic backend runs
# anywhere; the others are resolved lazily so importing this package never
# requires torch/vllm.
BACKENDS = ("synthetic", "hf_gpu", "vllm")


def make_backend(name: str, **kwargs):
    """Instantiate a backend by name. `kwargs` are forwarded to its constructor.

    synthetic : runs anywhere, no GPU (default).
    hf_gpu    : real model via HuggingFace transformers (needs CUDA + torch).
    vllm      : higher-throughput production path (needs CUDA + vLLM).
    """
    if name == "synthetic":
        return SyntheticBackend(**kwargs)
    if name == "hf_gpu":
        from .hf_gpu import HFGPUBackend
        return HFGPUBackend(**kwargs)
    if name == "vllm":
        from .vllm_adapter import VLLMBackend
        return VLLMBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
