"""
Reed-Solomon ECC wrapper for the noisy-channel study.

The covert channel is noisy whenever the decoder's logits diverge from the encoder's
(hardware / prefill-vs-decode / quantization). ECC buys reliability at a throughput cost:
net rate = data / (data + parity). We charge that cost explicitly so the reported
"net reliable bits/token" is honest rather than the raw (pre-ECC) capacity.
"""
from __future__ import annotations

from reedsolo import RSCodec, ReedSolomonError


def bits_to_bytes(bits) -> bytes:
    bits = list(int(b) for b in bits)
    pad = (-len(bits)) % 8
    bits = bits + [0] * pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | b
        out.append(v)
    return bytes(out)


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        bits.extend((byte >> (7 - i)) & 1 for i in range(8))
    return bits


def rs_encode(data: bytes, nsym: int) -> bytes:
    """Add `nsym` parity bytes per (255-nsym)-byte block. Corrects up to nsym//2 byte errors/block."""
    return bytes(RSCodec(nsym).encode(data))


def rs_decode(encoded: bytes, nsym: int):
    """Returns (data_bytes_or_None, ok). ok=False if uncorrectable."""
    try:
        decoded = RSCodec(nsym).decode(encoded)[0]
        return bytes(decoded), True
    except ReedSolomonError:
        return None, False


def net_rate(nsym: int, block: int = 255) -> float:
    """Fraction of channel bits that carry payload after RS overhead."""
    data = block - nsym
    return data / block
