"""Tests for the provider-side inference commitment (ivgym.commitment).

Dependency-free (stdlib + numpy + cryptography), same style as test_smoke.py:
    python tests/test_commitment.py            # or: python -m pytest tests/ -q

Covers the crypto binding the vLLM PR relies on: leaf determinism +
canonicalization, honest inclusion under a signed root, and that every
provider-side lie (tampered output, forged root, wrong claimed spec, post-publish
leaf swap) is rejected. These properties are what make "predict the audit set"
and "commit-honest / serve-cheap" fail; no model is needed to test them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import commitment as C
from ivgym.core import SamplingSpec

SPEC = SamplingSpec(temperature=1.0, top_k=50, top_p=0.95, seed=42)
MODEL = "test/model"


def _emitter_with(n, epoch=None):
    """Emit n deterministic served requests and seal the epoch."""
    em = C.CommitmentEmitter(model_id=MODEL, epoch=epoch or n)
    ops = []
    for i in range(n):
        prompt = np.arange(i, i + 5, dtype=np.int32)
        out = np.array([100 + i, 7, 42, 9], dtype=np.int32)
        ops.append((prompt, out, SPEC.seed, em.record(prompt, out, SPEC.seed, SPEC)))
    sr = em.seal()
    return em, ops, sr


def test_leaf_determinism_and_canonicalization():
    p = np.array([1, 2, 3], np.int32)
    o = np.array([9, 8], np.int32)
    a = C.leaf_hash(MODEL, "v1", p, o, 42, SPEC)
    b = C.leaf_hash(MODEL, "v1", p, o, 42, SPEC)
    assert a == b, "leaf hash must be deterministic"
    # any field change changes the leaf
    assert a != C.leaf_hash(MODEL, "v1", p, np.array([9, 7], np.int32), 42, SPEC)
    assert a != C.leaf_hash(MODEL, "v1", p, o, 43, SPEC)
    assert a != C.leaf_hash(MODEL, "v1", p, o, 42, SPEC.replace(temperature=0.9))
    assert a != C.leaf_hash("other/model", "v1", p, o, 42, SPEC)
    # canonical params are key-order independent and byte-stable
    assert (C.canonical_sampling_params(SPEC)
            == C.canonical_sampling_params(SamplingSpec(seed=42, top_p=0.95,
                                                        top_k=50, temperature=1.0)))
    print("ok  leaf determinism + canonicalization")


def test_signed_root_and_honest_inclusion():
    em, ops, sr = _emitter_with(6)
    pk = em.public_key()
    assert C.verify_signed_root(sr, pk)
    for p, o, s, op0 in ops:
        op = em.opening_for(op0.epoch_id, op0.leaf_index)
        assert C.verify_inclusion(MODEL, em.spec_version, p, o, s, SPEC, op, sr, pk)
    print("ok  signed root verifies + all honest openings include")


def test_tampered_output_rejected():
    em, ops, sr = _emitter_with(6)
    pk = em.public_key()
    p, o, s, op0 = ops[3]
    op = em.opening_for(op0.epoch_id, op0.leaf_index)
    tam = o.copy(); tam[0] ^= 1
    assert not C.verify_inclusion(MODEL, em.spec_version, p, tam, s, SPEC, op, sr, pk)
    # claiming a different spec/seed than committed also fails
    assert not C.verify_inclusion(MODEL, em.spec_version, p, o, 43, SPEC, op, sr, pk)
    assert not C.verify_inclusion(MODEL, em.spec_version, p, o, s,
                                  SPEC.replace(temperature=0.5), op, sr, pk)
    print("ok  tampered output / wrong claimed spec rejected")


def test_forged_root_rejected():
    em, ops, sr = _emitter_with(4)
    pk = em.public_key()
    forged = C.SignedRoot(sr.epoch_id,
                          sr.root[:-1] + bytes([sr.root[-1] ^ 1]), sr.signature)
    assert not C.verify_signed_root(forged, pk)
    # a different key cannot validate a root it did not sign
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    other_pk = Ed25519PrivateKey.generate().public_key()
    assert not C.verify_signed_root(sr, other_pk)
    print("ok  forged root + wrong-key rejected")


def test_post_publish_leaf_swap_changes_root():
    """After the root is signed+published, replacing any committed leaf changes
    the root -- so a provider cannot retroactively make an audited request honest
    without forging the signature. This is what pins per-request catch prob to q."""
    em, ops, sr = _emitter_with(8)
    tree = em._sealed_trees[sr.epoch_id]
    for idx in range(len(ops)):
        alt = list(tree.leaves)
        # swap for a leaf over some *different* served content
        alt[idx] = C.leaf_hash(MODEL, em.spec_version,
                               np.array([999], np.int32), np.array([1], np.int32),
                               SPEC.seed, SPEC)
        assert C.MerkleTree(alt).root != sr.root
    print("ok  post-publish leaf swap always changes the signed root")


def test_wrong_opening_index_rejected():
    em, ops, sr = _emitter_with(6)
    pk = em.public_key()
    p, o, s, op0 = ops[2]
    # present request 2's data but request 5's authentication path
    wrong = em.opening_for(op0.epoch_id, 5)
    assert not C.verify_inclusion(MODEL, em.spec_version, p, o, s, SPEC, wrong, sr, pk)
    print("ok  mismatched opening path rejected")


def main():
    test_leaf_determinism_and_canonicalization()
    test_signed_root_and_honest_inclusion()
    test_tampered_output_rejected()
    test_forged_root_rejected()
    test_post_publish_leaf_swap_changes_root()
    test_wrong_opening_index_rejected()
    print("\nall commitment tests passed.")


if __name__ == "__main__":
    main()
