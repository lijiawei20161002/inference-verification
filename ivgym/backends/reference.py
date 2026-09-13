"""Reference-only backend owned by the verification service, never the provider.

This adapter feeds VContext to the existing token verifiers. It has no remote
provider client and accepts neither provider logits nor provider calibration.
"""

from __future__ import annotations
import hashlib
import json
import time
import numpy as np
from ..core import SamplingSpec, VContext
from ..sampling import gumbel_noise, position_seed
from ..verifiers import CrossEntropy, TokenDiFR, TokenTOPLOC

TOKEN_VERIFIERS = {v.name: v for v in (CrossEntropy(), TokenDiFR(), TokenTOPLOC())}


class HFReferenceModel:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config, self.torch = config, torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.snapshot, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config.snapshot,
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, config.dtype),
                attn_implementation=config.attention,
            )
            .to(config.device)
            .eval()
        )
        self.identity = dict(
            repository=config.repository,
            revision=config.revision,
            dtype=config.dtype,
            attention=config.attention,
            torch=torch.__version__,
            device=torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
        )
        # The service operator pins the snapshot; file hashes make the served reference identifiable.
        from pathlib import Path

        files = {}
        for p in sorted(Path(config.snapshot).rglob("*")):
            if p.is_file():
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(block)
                files[str(p.relative_to(config.snapshot))] = h.hexdigest()
        self.identity["files_sha256"] = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare(self, capture):
        if capture.prompt_token_ids is not None:
            prompt, output = capture.prompt_token_ids, capture.output_token_ids
            mode = "exact_ids"
        else:
            prompt, output = self.encode(capture.prompt), self.encode(capture.output)
            mode = "retokenized_text"
        if not prompt or not output:
            raise ValueError("empty tokenized prompt or output")
        if max(prompt + output) >= self.model.config.vocab_size:
            raise ValueError("token ID outside reference vocabulary")
        if len(prompt) + len(output) > self.model.config.max_position_embeddings:
            raise ValueError("reference context exceeded")
        return prompt, output, mode

    def score(self, prepared, capture, sampling, detectors):
        torch = self.torch
        prompt, output, mode = prepared
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            x = torch.tensor(
                [prompt + output[:-1]], device=self.config.device, dtype=torch.long
            )
            logits = (
                self.model(input_ids=x, use_cache=False)
                .logits[0, len(prompt) - 1 :]
                .float()
                .cpu()
                .numpy()
            )
        shared = None
        if "token_difr" in detectors:
            shared = np.stack(
                [
                    gumbel_noise(
                        logits.shape[1],
                        position_seed(sampling.seed, capture.prompt_id, i),
                    )
                    for i in range(len(output))
                ]
            )
        ctx = VContext(
            capture.prompt_id,
            output,
            SamplingSpec(**sampling.model_dump()),
            ref_logits=logits,
            gumbel=shared,
        )
        scores = {
            name: TOKEN_VERIFIERS[name].evidence(ctx).tolist() for name in detectors
        }
        return scores, time.perf_counter() - started
