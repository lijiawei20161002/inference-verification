from .synthetic import SyntheticBackend  # noqa: F401

# HFGPUBackend imports torch/transformers; only available on a CUDA host with
# those installed. Import it directly: `from ivgym.backends.hf_gpu import HFGPUBackend`.
