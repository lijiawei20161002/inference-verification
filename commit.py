"""SHA-256 Merkle commitment over per-request transcripts.

Soundness role: the provider must publish the Merkle root *before* learning
which requests the client will audit. Because the root binds every leaf, the
provider cannot, after seeing the audit set, swap a cheated transcript for an
honest one without changing the root (which would be detected). This is what
makes the per-request catch probability exactly q rather than something the
provider can drive to zero by predicting the audit set.
"""
import hashlib
import numpy as np

def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def leaf_bytes(prompt, out_tokens, out_logits, quant=1e-3):
    """Serialize a transcript. Logits are quantized so the commitment defines a
    reproducible value the client can re-derive; the quantization step is the
    floor of the tolerance band."""
    q = np.round(out_logits / quant).astype(np.int32)
    parts = [
        np.asarray(prompt, np.int32).tobytes(),
        b'|',
        np.asarray(out_tokens, np.int32).tobytes(),
        b'|',
        q.tobytes(),
    ]
    return b''.join(parts)

def leaf_hash(prompt, out_tokens, out_logits, quant=1e-3):
    return _h(b'\x00' + leaf_bytes(prompt, out_tokens, out_logits, quant))

class MerkleTree:
    """Standard binary Merkle tree with domain-separated node hashing."""
    def __init__(self, leaves):
        self.leaves = list(leaves)
        self.levels = [self.leaves[:]]
        cur = self.leaves[:]
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]  # duplicate last
                nxt.append(_h(b'\x01' + left + right))
            self.levels.append(nxt)
            cur = nxt
        self.root = cur[0] if cur else _h(b'')

    def proof(self, index):
        """Authentication path for leaf `index`: list of (sibling, is_right)."""
        path = []
        idx = index
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                sib = level[idx + 1] if idx + 1 < len(level) else level[idx]
                path.append((sib, True))       # sibling is on the right
            else:
                path.append((level[idx - 1], False))
            idx //= 2
        return path

    @staticmethod
    def verify(leaf, index, path, root):
        h = leaf
        for sib, sib_is_right in path:
            h = _h(b'\x01' + h + sib) if sib_is_right else _h(b'\x01' + sib + h)
        return h == root
