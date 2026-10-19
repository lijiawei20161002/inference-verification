"""Install a pinned NVIDIA CLI package under /workspace without changing CUDA/PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

VERSION = "2026.5.1"
PACKAGE = "NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb"
URL = "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/" + PACKAGE
SHA256 = "61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def install(root):
    destination = root / "nsight-systems" / VERSION
    cache = root / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / PACKAGE
    if not package.exists() or digest(package) != SHA256:
        partial = package.with_suffix(".partial")
        print(f"Downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if digest(partial) != SHA256:
            raise RuntimeError("NVIDIA package SHA256 mismatch; refusing to extract")
        partial.replace(package)
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["dpkg-deb", "--extract", str(package), str(destination)], check=True)
    candidates = list(destination.glob("opt/nvidia/nsight-systems*/**/bin/nsys"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one nsys executable, found {candidates}")
    binary = candidates[0].resolve()
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if VERSION not in version:
        raise RuntimeError(f"Unexpected version: {version}")
    link = root / "bin/nsys"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != binary:
            raise RuntimeError(f"Refusing to replace existing executable: {link}")
    else:
        link.symlink_to(binary)
    manifest = {"url": URL, "package_sha256": SHA256, "version": version,
                "executable": str(binary), "launcher": str(link)}
    (destination / "installation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/.tools"))
    args = parser.parse_args()
    install(args.root.resolve())


if __name__ == "__main__":
    main()
