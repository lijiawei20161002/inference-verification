"""Pinned BF16 reference and actual HF generation on CUDA.

Unfiltered NLL uses claimed temperature. Rank is log1p(min(strict rank, 50)).
No noise injection or seed synchronization. Score the provider-returned prefix.
"""
from __future__ import annotations

import importlib.metadata
import math
import platform
from pathlib import Path

from .crypto import digest, require
from .contract import DETECTORS


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def software():
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), transformers=importlib.metadata.version("transformers"),
                scoring_source_sha256=file_hash(__file__))


def model_manifest(snapshot, repository, revision):
    snapshot = Path(snapshot)
    files = {str(p.relative_to(snapshot)): file_hash(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    require(any(p.endswith(".safetensors") for p in files), "model weights missing")
    tokenizer = {k: v for k, v in files.items() if "token" in k or k in ("vocab.json", "merges.txt")}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    return dict(repository=repository, revision=revision, files_sha256=digest(files),
                tokenizer_sha256=digest(tokenizer), chat_template_sha256=digest(tok.chat_template),
                adapters=[], claimed_dtype="bfloat16"), files


class HFModel:
    def __init__(self, snapshot, repository, revision, attention="eager", quantization=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        require(torch.cuda.is_available(), "CUDA reference required")
        self.torch = torch
        self.manifest, self.files = model_manifest(snapshot, repository, revision)
        self.attention = attention
        self.quantization = quantization
        require(quantization in (None, "nf4"), "unsupported quantization")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        kwargs = dict(dtype=torch.bfloat16, attn_implementation=attention,
                      local_files_only=True, trust_remote_code=False)
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                          bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True), device_map={"": 0})
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
        if quantization is None:
            self.model = self.model.to("cuda")
        self.model.eval()

    def profile(self, executor_id="reference"):
        require(self.quantization is None, "quantized provider cannot act as BF16 reference")
        return dict(executor_id=executor_id, software_sha256=digest(software()),
                    hardware=self.torch.cuda.get_device_name(), attention=self.attention,
                    dtype="bfloat16", replay_atol=1e-5)

    def generate(self, request, temperature_override=None):
        torch = self.torch
        g = request["generation"]
        temperature = g["temperature"] if temperature_override is None else temperature_override
        x = torch.tensor([request["prompt_token_ids"]], device="cuda", dtype=torch.long)
        eos, result, cache = set(g["eos_token_ids"]), [], None
        require(x.shape[1] + g["max_output_tokens"] <= self.model.config.max_position_embeddings,
                "model context exceeded")
        with torch.inference_mode():
            for _ in range(g["max_output_tokens"]):
                out = self.model(input_ids=x, past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1, :].float() / temperature
                token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                value = int(token.item())
                result.append(value)
                if value in eos:
                    break
                cache, x = out.past_key_values, token
        return result, self.tokenizer.decode(result, skip_special_tokens=False), "eos" if result[-1] in eos else "length"


class HFReferenceExecutor:
    """Own model instance, with no provider cache or provider logit input."""
    def __init__(self, model):
        self.model = model

    def score(self, records, spec):
        model, torch = self.model, self.model.torch
        require(model.manifest == spec["model"], "reference weights/tokenizer digest mismatch")
        require(model.profile(spec["reference"]["executor_id"]) == spec["reference"], "unsupported reference profile")
        per_token, prefill = [], 0
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for row in records:
                prompt = row["request"]["prompt_token_ids"]
                output = row["capture"]["payload"]["output_token_ids"]
                require(all(t < model.model.config.vocab_size for t in prompt + output), "token outside vocabulary")
                ids = prompt + output[:-1]
                prefill += len(ids)
                x = torch.tensor([ids], device="cuda", dtype=torch.long)
                logits = model.model(input_ids=x, use_cache=False).logits[0, len(prompt) - 1:].float()
                target = torch.tensor(output, device="cuda", dtype=torch.long)
                scaled = logits / spec["generation"]["temperature"]
                nll = -torch.log_softmax(scaled, dim=-1).gather(1, target[:, None]).squeeze(1)
                chosen = logits.gather(1, target[:, None])
                rank = (logits > chosen).sum(dim=-1).clamp(max=50)
                values = zip(nll.cpu().tolist(), rank.cpu().tolist())
                per_token.extend({DETECTORS[0]: n, DETECTORS[1]: math.log1p(r)} for n, r in values)
        end.record()
        torch.cuda.synchronize()
        require(per_token, "empty reference block")
        return dict(model=model.manifest, profile=model.profile(spec["reference"]["executor_id"]),
                    per_token_scores=per_token,
                    scores={d: sum(row[d] for row in per_token) / len(per_token) for d in DETECTORS},
                    token_count=len(per_token), prefill_tokens=prefill,
                    gpu_seconds=start.elapsed_time(end) / 1000)
