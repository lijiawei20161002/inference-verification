"""Provider-side inference commitment -- the real implementation of the vLLM PR.

This is the code the PR proposal (docs/VLLM_PROVIDER_COMMITMENT_PR.md) drops into
`vllm/entrypoints/commitment.py`. It is written in pure Python (stdlib hashlib +
the `cryptography` Ed25519 primitive) so it runs unchanged in this repo and would
run unchanged inside a vLLM server.

The design constraint is the whole point: **this module never verifies anything.**
The provider is the untrusted party, so a verifier it runs is one it can stub. All
this code does is *bind* what was served -- form a leaf over (model, spec, prompt,
output, seed, sampling params), accumulate leaves into a Merkle tree, and SIGN and
publish the root each epoch. Catching a cheat is done entirely off-box by an
auditor that re-computes the real model (ivgym defenses); see
`verify_inclusion` for the one thing the *client* checks against the signed root.

Soundness rests on two facts, both external to any value the provider computes:
  1. The root is signed and published BEFORE the auditor draws its random audit
     set, so leaves cannot be swapped post-hoc (predict-the-audit-set is dead).
  2. The opening is returned to the caller, so the leaf is over the tokens the
     user actually received -- a provider cannot commit-honest / serve-cheap.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .core import SamplingSpec

# Ed25519: real signatures. `cryptography` is a hard dep of transformers/vLLM.
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


# ---------------------------------------------------------------------------
# Canonical serialization + leaf hashing (domain-separated SHA-256)
# ---------------------------------------------------------------------------
def canonical_sampling_params(spec: SamplingSpec) -> bytes:
    """Deterministic, key-sorted encoding of phi. Byte-identical for equal specs
    across processes (unlike repr/pickle)."""
    d = {
        "temperature": spec.temperature,
        "top_k": spec.top_k,
        "top_p": spec.top_p,
        "seed": spec.seed,
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def _i32(a) -> bytes:
    return np.asarray(a, np.int32).tobytes()


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def leaf_hash(model_id: str, spec_version: str, prompt_ids, output_ids,
              seed: int, spec: SamplingSpec) -> bytes:
    """The committed leaf. Note there are NO logits here: soundness comes from the
    auditor recomputing logits from the real model, so the provider is not trusted
    to report them (an optional logit-hash accelerator can be layered on top, but
    is never a source of trust)."""
    h = hashlib.sha256()
    for part in (
        b"\x00", model_id.encode(),
        b"\x00", spec_version.encode(),
        b"\x00", _i32(prompt_ids),
        b"\x00", _i32(output_ids),
        b"\x00", str(int(seed)).encode(),
        b"\x00", canonical_sampling_params(spec),
    ):
        h.update(part)
    return h.digest()


# ---------------------------------------------------------------------------
# Merkle tree (mirrors the MVP's commit.py, kept self-contained for the PR)
# ---------------------------------------------------------------------------
class MerkleTree:
    def __init__(self, leaves):
        self.leaves = list(leaves)
        self.levels = [self.leaves[:]]
        cur = self.leaves[:]
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]  # duplicate last
                nxt.append(_h(b"\x01" + left + right))
            self.levels.append(nxt)
            cur = nxt
        self.root = cur[0] if cur else _h(b"")

    def proof(self, index):
        """Authentication path for leaf `index`: list of (sibling, sib_is_right)."""
        path, idx = [], index
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                sib = level[idx + 1] if idx + 1 < len(level) else level[idx]
                path.append((sib, True))
            else:
                path.append((level[idx - 1], False))
            idx //= 2
        return path

    @staticmethod
    def verify_path(leaf, path, root) -> bool:
        h = leaf
        for sib, sib_is_right in path:
            h = _h(b"\x01" + h + sib) if sib_is_right else _h(b"\x01" + sib + h)
        return h == root


# ---------------------------------------------------------------------------
# Signed roots, openings, and the emitter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignedRoot:
    epoch_id: int
    root: bytes
    signature: bytes           # Ed25519 over (epoch_id ‖ root)

    def signed_message(self) -> bytes:
        return str(self.epoch_id).encode() + b"\x1f" + self.root


@dataclass(frozen=True)
class Opening:
    """Returned to the caller in RequestOutput.commitment. `leaf` and `path` let
    the holder prove inclusion under the epoch's signed root."""
    epoch_id: int
    leaf_index: int
    leaf: bytes
    path: list


class CommitmentEmitter:
    """Append-only Merkle accumulator + signed-root publisher. NEVER verifies.

    In vLLM this is invoked once per finished request from the output processor.
    Here `record(...)` takes the same fields the output path has in hand.
    """

    def __init__(self, model_id: str, spec_version: str = "gumbel-max/v1",
                 signing_key: Ed25519PrivateKey | None = None, epoch: int = 1024):
        self.model_id = model_id
        self.spec_version = spec_version
        self.signing_key = signing_key or Ed25519PrivateKey.generate()
        self.epoch = epoch
        self._buf: list[bytes] = []
        self._records: list[tuple] = []       # (prompt_ids, output_ids, seed, spec)
        self._epoch_id = 0
        self.published: list[SignedRoot] = []  # the append-only transparency sink
        self._sealed_trees: dict[int, MerkleTree] = {}

    # -- provider side: bind one served request, return its opening --------
    def record(self, prompt_ids, output_ids, seed: int, spec: SamplingSpec) -> Opening:
        leaf = leaf_hash(self.model_id, self.spec_version, prompt_ids,
                         output_ids, seed, spec)
        self._buf.append(leaf)
        self._records.append((prompt_ids, output_ids, seed, spec))
        idx = len(self._buf) - 1
        epoch_id = self._epoch_id
        # NOTE: the path is only meaningful once the epoch is sealed; we return a
        # lazy Opening and fill `path` at seal time via `opening_for`.
        return Opening(epoch_id=epoch_id, leaf_index=idx, leaf=leaf, path=[])

    def seal(self) -> SignedRoot:
        """Build the tree over the current epoch, SIGN the root, publish it, and
        start a new epoch. In vLLM this fires every `epoch` requests / T seconds."""
        tree = MerkleTree(self._buf)
        sig = self.signing_key.sign(
            str(self._epoch_id).encode() + b"\x1f" + tree.root)
        sr = SignedRoot(epoch_id=self._epoch_id, root=tree.root, signature=sig)
        self.published.append(sr)
        self._sealed_trees[self._epoch_id] = tree
        self._epoch_id += 1
        self._buf, self._records = [], []
        return sr

    def opening_for(self, epoch_id: int, leaf_index: int) -> Opening:
        """The full opening (with authentication path) for a sealed epoch."""
        tree = self._sealed_trees[epoch_id]
        return Opening(epoch_id=epoch_id, leaf_index=leaf_index,
                       leaf=tree.leaves[leaf_index], path=tree.proof(leaf_index))

    def public_key(self) -> Ed25519PublicKey:
        return self.signing_key.public_key()


# ---------------------------------------------------------------------------
# Client side: the ONE thing the auditor checks about the commitment itself.
# (Everything else -- catching a cheat -- is recompute + an ivgym defense.)
# ---------------------------------------------------------------------------
def verify_signed_root(sr: SignedRoot, public_key: Ed25519PublicKey) -> bool:
    try:
        public_key.verify(sr.signature, sr.signed_message())
        return True
    except InvalidSignature:
        return False


def verify_inclusion(model_id: str, spec_version: str, prompt_ids, output_ids,
                     seed: int, spec: SamplingSpec,
                     opening: Opening, signed_root: SignedRoot,
                     public_key: Ed25519PublicKey) -> bool:
    """Prove the (prompt, output, seed, spec) the holder actually received is the
    one bound under a validly-signed root. Fails if the signature is bad, or the
    recomputed leaf is not under the root (tamper / wrong opening)."""
    if not verify_signed_root(signed_root, public_key):
        return False
    if opening.epoch_id != signed_root.epoch_id:
        return False
    leaf = leaf_hash(model_id, spec_version, prompt_ids, output_ids, seed, spec)
    if leaf != opening.leaf:
        return False
    return MerkleTree.verify_path(leaf, opening.path, signed_root.root)
