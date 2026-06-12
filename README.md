# inference-verification

Experiments on the verifiability of LLM inference.

| module | paper | what it shows |
|---|---|---|
| [`e4_stego/`](e4_stego/) | Weight-exfiltration verifier (arXiv 2511.02620) | A concrete steganographic encoder/decoder measured against the authors' real GLS scorer — realized covert capacity vs their information-theoretic bound. |
| [`e5_slack/`](e5_slack/) | Weight-exfiltration verifier (arXiv 2511.02620) | Maps per-token "slack" (the SAFE set the paper bounds) across a real generation — where covert freedom actually lives. |
| [`bitexact_a100/`](bitexact_a100/) | Bit-Exact Inference Verification (arXiv 2606.00279) | A100 pilot reproducing the Pillar-A claims: GPU inference is bit-deterministic given fixed conditions, *non-invariant* across shape/batch (non-associativity), with genuine non-determinism confined to float `atomicAdd`. |
