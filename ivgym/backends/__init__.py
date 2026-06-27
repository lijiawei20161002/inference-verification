# HFGPUBackend / VLLMBackend import torch/transformers/vllm; only available on a
# CUDA host with those installed. They are imported lazily by `make_backend`
# (and can still be imported directly, e.g.
# `from ivgym.backends.hf_gpu import HFGPUBackend`), so importing this package
# never requires torch/vllm.

# Names the CLI / `make_backend` understand.
BACKENDS = ("hf_gpu", "vllm")


def make_backend(name: str, **kwargs):
    """Instantiate a backend by name. `kwargs` are forwarded to its constructor.

    hf_gpu : real model via HuggingFace transformers (needs CUDA + torch). Default.
    vllm   : higher-throughput production path (needs CUDA + vLLM).
    """
    if name == "hf_gpu":
        from .hf_gpu import HFGPUBackend
        return HFGPUBackend(**kwargs)
    if name == "vllm":
        from .vllm_adapter import VLLMBackend
        return VLLMBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
