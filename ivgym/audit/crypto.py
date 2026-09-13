"""RFC 8785 / Ed25519 wire primitives; no model correctness is implied."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

VERSION = "ivgym.audit.v1"


class Invalid(ValueError):
    pass


class Untrusted(Invalid):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def canonical(value):
    return rfc8785.dumps(value)


def loads(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    def bad(value):
        raise Invalid("invalid JSON number: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    canonical(value)  # Also reject overflow, unsafe integers and lone surrogates.
    return value


def read(path):
    return loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def blob_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def unb64(value):
    raw = base64.b64decode(value, validate=True)
    require(b64(raw) == value, "noncanonical base64")
    return raw


def framed(*parts):
    return b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def signing_bytes(envelope):
    return framed(b"IVGYM-SIGNATURE", VERSION.encode(), envelope["role"].encode(),
                  envelope["kind"].encode(), canonical(envelope))


def new_key(path):
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return key


def load_key(path):
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    require(isinstance(key, Ed25519PrivateKey), "expected Ed25519 key")
    return key


def public(key):
    return b64(key.public_key().public_bytes(serialization.Encoding.Raw,
                                            serialization.PublicFormat.Raw))


def sign(payload, key, key_id, role, kind):
    env = dict(version=VERSION, role=role, kind=kind, key_id=key_id, payload=payload)
    return dict(env, signature=b64(key.sign(signing_bytes(env))))


def authenticate(env, trust, role, kind, key_id=None):
    require(set(env) == {"version", "role", "kind", "key_id", "payload", "signature"},
            "invalid signature envelope fields")
    require(env["version"] == VERSION, "unsupported protocol version")
    require(env["role"] == role and env["kind"] == kind, "signature role/type mismatch")
    require(key_id is None or env["key_id"] == key_id, "wrong pinned signer")
    encoded = trust.get(role, {}).get(env["key_id"])
    if encoded is None:
        raise Untrusted(f"no local trust root for {role}/{env['key_id']}")
    unsigned = {k: v for k, v in env.items() if k != "signature"}
    Ed25519PublicKey.from_public_bytes(unb64(encoded)).verify(
        unb64(env["signature"]), signing_bytes(unsigned))
    return env["payload"]
