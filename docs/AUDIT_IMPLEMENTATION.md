# Experimental audit protocol v1

The [RFC](AUDIT_PROTOCOL_RFC.md) now has an executable, deliberately bounded
implementation in `ivgym/audit/`. It provides signed artifacts, a real HTTP
receipt extension, independent Hugging Face CUDA scoring, offline verification,
and GPU replay. The research harness and its existing detectors are unchanged.
This release does not certify a deployment operating point.

![Implemented protocol sequence with a post-closure selection boundary and distinct verify and replay paths.](figures/fig_audit_protocol.png)

[SVG](figures/fig_audit_protocol.svg) · [PNG](figures/fig_audit_protocol.png) ·
[PDF](figures/fig_audit_protocol.pdf). The figure shows receipted mode; missing
required capabilities or evidence prevent a completed statistical decision.

## Tested status: 2026-09-13

The [H100 pilot](results/audit_protocol_h100.md) completed 3 development, 63
calibration and 40 held-out blocks, with a further separate-process CLI audit.
At `alpha=0.05`, it flagged 1/20 honest BF16 blocks, 10/10 temperature overrides
and 0/10 actual NF4 blocks. NF4 detection was not demonstrated at this budget;
the honest sample does not certify a deployment false-positive rate.

The complete suite passed **112 tests**, including 45 protocol and conformance
tests. Its one warning comes from the existing test of the research harness's
batch/pool ratio ceiling. All **107 artifacts** verified. Four replay checks
(one per provider condition plus the separate-process CLI check) had zero score
delta and no decision-boundary disagreement. Full block scores, signed evidence,
confidence intervals and measured costs are linked from the pilot report.

The tested provider is Hugging Face on one H100, using the `/ivgym/` extension.
There is no tested vLLM path or deployment calibration release. The unresolved
RFC template is not an executable spec; use the provisioned v1 spec described
below.

## Run it

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[audit,gpu,test]' bitsandbytes scipy
.venv/bin/python -m pytest tests/test_audit_protocol.py tests/test_audit_conformance.py -q

# Download the exact snapshot used in the pilot; copy the printed path below.
HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
))
PY

.venv/bin/python -m experiments.exp_audit_protocol_gpu \
  --snapshot /path/to/models--Qwen--Qwen3-0.6B/snapshots/IMMUTABLE_COMMIT \
  --out runs/audit-h100

.venv/bin/ivgym verify runs/audit-h100/honest/block-0000 \
  --trust runs/audit-h100/trust.json
.venv/bin/ivgym replay runs/audit-h100/honest/block-0000 \
  --trust runs/audit-h100/trust.json \
  --reference runs/audit-h100/reference-config.json
```

Use a new `--out` directory for each run; the experiment refuses to overwrite
an existing run. Its defaults reproduce the pilot block counts and include real
NF4 generation. Run `.venv/bin/python -m pytest tests/ -q` for the complete suite.

The experiment provisions separate auditor/provider/reference/calibration keys,
hashes all checkpoint files, records software and hardware profiles, and writes
resolved specs, public trust roots, signed calibration, every full transcript,
and an aggregate report. Private keys are created exclusively with permissions
0600; run directories use 0700 and are gitignored. Trust roots must be distributed
through an independent trusted channel. Public keys arriving with an artifact
are not an authenticated identity.

For a subsequent audit using the provisioned configuration, start the local
test provider in one terminal:

```sh
.venv/bin/ivgym serve --spec runs/audit-h100/spec.json \
  --reference runs/audit-h100/reference-config.json \
  --provider-key runs/audit-h100/keys/provider.pem \
  --trust runs/audit-h100/trust.json
```

Save exactly four prompts as an array of token-ID arrays in `prompts.json`, using
the pinned tokenizer and the calibrated traffic policy, then run:

```sh
.venv/bin/ivgym audit http://127.0.0.1:8765 --model Qwen/Qwen3-0.6B \
  --spec runs/audit-h100/spec.json --calibration runs/audit-h100/calibration.json \
  --prompts prompts.json --reference runs/audit-h100/reference-config.json \
  --trust runs/audit-h100/trust.json \
  --auditor-key runs/audit-h100/keys/auditor.pem \
  --reference-key runs/audit-h100/keys/reference.pem --out runs/new-audit
```

`python -m ivgym` is equivalent to the installed command. `keygen PATH` creates
an Ed25519 private key and prints its public key. The CLI exits 1 for an integrity,
configuration or replay failure, 2 for an unsupported/inconclusive audit, and 0
for a completed statistical decision. An `inconsistent` decision is a successful
audit execution; automation must inspect the outcome field.

## Executable scope

`ivgym.audit.v1` is a new, strict wire schema. The unresolved draft template is
intentionally rejected. Its supported generation contract is raw token input,
positive temperature, full softmax (`top_p=1`, `top_k=null`), explicit EOS IDs and
output cap, and rejection on context overflow. There are no inherited generation
defaults, chat preprocessing, hidden reasoning, custom logits processors, or
automatic retries. Such features need new contracts and calibration profiles.

The two frozen detector scores are unfiltered reference NLL at the claimed
temperature and `log1p(min(strict token rank, 50))`. Each block score is the mean
over the selected output tokens. Rank ties count only strictly greater logits.
Each output position is scored from its exact prompt and returned prefix.
Neither feature is a sampler proof. Seed-synchronized and activation detectors
are not exposed by this audit API.

The calibration profile binds model, generation, reference and the entire block
design, including traffic policy, sample counts, detectors and alpha. For each
detector the checker recomputes `(1 + count(calibration >= observed))/(m+1)`, takes
the maximum across nuisance conditions and uses `alpha/2`. Calibration uses
fresh complete blocks; no token shuffling or bootstrap creates new trials.
Insufficient rank-test resolution, unmatched profiles and incomplete evidence
produce `inconclusive`. Provisioning bundles contain no null scores and always
produce `inconclusive`, while retaining reference measurements for calibration.

All operating points are labeled exploratory. Setting
`require_validated_operating_point=true` returns inconclusive unconditionally in
this release; a publisher-supplied boolean cannot establish deployment validity.
`alpha_scope` supports one fixed-horizon audit only. Persistent campaign alpha
accounting, sequential stopping and rare-error certification are not implemented.

## Wire and verification contract

The `/ivgym/deployment`, `/ivgym/start`, `/ivgym/completions` and `/ivgym/close`
routes implement the negotiated exact-token extension. Receipt middleware takes
an engine generation callable and signs its outputs; the bundled runnable
engine is Hugging Face. These routes are **not** ordinary OpenAI-compatible
routes. An ordinary endpoint without the extension returns `unsupported` when
receipts are required. There is no tested vLLM adapter in this release.

The middleware authenticates the plan, rejects replayed sessions and out-of-order
requests, signs chained final receipts, checks that the closure contains every
response it served, and refuses reopened closures. Requests and responses are
captured as exact HTTP body bytes encoded in strict base64, alongside exact IDs.
Transport headers and TLS packet evidence are not captured. Streaming/SSE is not
implemented. The server is for local experiments and has no durable session
store, authentication layer beyond plan signatures, or production load controls.

All signatures use [Ed25519](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/).
JSON uses [RFC 8785 canonicalization](https://www.rfc-editor.org/rfc/rfc8785).
The parser rejects duplicate keys, nonfinite numbers, unsafe JSON integers and
malformed Unicode/base64. A signed envelope has exactly `version`, `role`,
`kind`, `key_id`, `payload`, `signature`. Remove `signature`, then sign this byte
sequence, where `frame(x) = uint64_be(len(x)) || x`:

```
frame(UTF8("IVGYM-SIGNATURE")) || frame(UTF8("ivgym.audit.v1")) ||
frame(UTF8(role)) || frame(UTF8(kind)) || frame(JCS(unsigned_envelope))
```

The selection commitment is SHA256 of separately framed domain
`ivgym-selection-v1`, UTF8 session, and 32 secret bytes. After provider closure,
derive `key = HMAC-SHA256(seed, frame("ivgym-selection-kdf-v1") || frame(root) ||
frame(session))`, where root is the lowercase hex digest of canonical closure
before execution results. For counters starting at zero, interpret
`HMAC-SHA256(key, uint64_be(counter))` as an unsigned big-endian integer. At each
partial Fisher–Yates step with `n` remaining entries, reject values at or above
`2^256 - (2^256 mod n)`; otherwise choose offset `value mod n`. Draw order is the
scoring order. This is uniform sampling without replacement. Generation randomness
is independent of the secret selection seed.

The fixed artifact file list is in `protocol.FILES`. `manifest.json` maps these
names to SHA256(JCS(parsed file)). `transcript.sig` signs that manifest outside
itself. Exact wire bodies remain hash-bound through their base64 fields.
Verification rejects symlinks, extra/missing files, wrong locally pinned signers,
bad signatures, altered chains/bindings/selection, wrong counts and changed
decisions. Missing trusted keys return `unverifiable`; malformed evidence returns
`invalid`. These are separate from statistical outcomes.

`verify` needs no GPU and does not import torch. It authenticates reference scores
but cannot establish that the reference ran correctly. `replay` loads the pinned
weights under the recorded numerical profile, recomputes every selected token
score and reports numerical or decision-boundary disagreement. It never rewrites
the original verdict. HF profiles bind the scorer source, package versions,
CUDA/cuDNN, attention implementation, dtype and GPU model. Container attestation,
cross-driver calibration and hardware identity proofs are outside this release.

## Budget and trust limits

Request/output limits are checked before sending the fixed schedule. Exact
selected reference-prefill cost is checked before scoring. Unknown monetary
pricing returns inconclusive before requests are sent; costs in dollars stay
null. Wire-body bytes, actual reference-prefill tokens and CUDA-event elapsed
scoring time are measured. CUDA-event elapsed time can include device idle time
between launches; it is not hardware utilization or a bill. Experiment wall time
includes collection, signing, reference and verification work.

Provider endorsement is an assertion about returned bytes, not GPU execution.
The reference runs on adopter-controlled hardware. Local signatures cannot prove
that an auditor did not discard whole sessions or that traffic profiles are
representative. The current implementation does not offer a private-inference
backend, fixed-point relation proof, external plan witness, cross-hardware
validation, production receipts for arbitrary endpoints, or deployment-grade
false-positive certification. These remain RFC follow-on work.

## Regenerate the protocol figure

```sh
.venv/bin/pip install 'matplotlib>=3.7'
.venv/bin/python -m experiments.plot_audit_protocol
```

The [figure source](../experiments/plot_audit_protocol.py) exports editable vector
SVG, a 2560 × 2272 PNG, and a vector PDF under `docs/figures/fig_audit_protocol.*`.
It requires no GPU, model download, private keys or captured traffic.
