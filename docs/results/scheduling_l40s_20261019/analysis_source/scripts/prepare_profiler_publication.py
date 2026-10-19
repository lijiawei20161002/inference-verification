"""Remove captured environment metadata from public Nsight SQLite exports.

Original archives are copied to an ignored private directory first. Proprietary
.nsys-rep files remain private because they cannot be safely edited as SQLite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3


SENSITIVE = re.compile(r"SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY", re.I)
PROTECTED = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_digest(connection, table):
    digest = hashlib.sha256()
    for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def sanitize_database(path):
    original_sha = sha(path)
    secrets = {v for k,v in os.environ.items() if SENSITIVE.search(k) and len(v)>=12}
    with sqlite3.connect(path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        before = {t:table_digest(conn,t) for t in PROTECTED if t in tables}
        environment = {}
        if "TARGET_INFO_SYSTEM_ENV" in tables:
            environment.update(conn.execute("SELECT name,value FROM TARGET_INFO_SYSTEM_ENV"))
        if "META_DATA_CAPTURE" in tables:
            for (value,) in conn.execute("SELECT value FROM META_DATA_CAPTURE WHERE name LIKE '%ENVIRONMENT_VARIABLE%'"):
                if value and "=" in value:
                    key, val = value.split("=",1)
                    environment[key] = val
        secrets.update(v for k,v in environment.items() if SENSITIVE.search(k) and v and len(v)>=12)
        prefixes = tuple(k+"=" for k in environment)
        changed = []
        if "StringIds" in tables:
            for identity,value in conn.execute("SELECT id,value FROM StringIds"):
                if value and (any(p in value for p in prefixes) or any(v in value for v in secrets)):
                    changed.append(identity)
        for table, columns in (("CUPTI_ACTIVITY_KIND_KERNEL", ("demangledName", "shortName", "mangledName")),
                               ("CUPTI_ACTIVITY_KIND_RUNTIME", ("nameId",)), ("NVTX_EVENTS", ("textId",))):
            if table not in tables:
                continue
            present = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            for column in columns:
                referenced = {r[0] for r in conn.execute(f'SELECT DISTINCT "{column}" FROM "{table}"')} if column in present else set()
                if referenced.intersection(changed):
                    raise ValueError("Environment string is referenced by a measurement event")
        conn.execute("PRAGMA secure_delete=ON")
        if changed:
            conn.executemany("UPDATE StringIds SET value='[environment metadata removed]' WHERE id=?",
                             ((i,) for i in changed))
        if "TARGET_INFO_SYSTEM_ENV" in tables:
            conn.execute("DELETE FROM TARGET_INFO_SYSTEM_ENV")
        if "META_DATA_CAPTURE" in tables:
            conn.execute("DELETE FROM META_DATA_CAPTURE WHERE name LIKE '%ENVIRONMENT_VARIABLE%'")
        conn.commit()
        conn.execute("VACUUM")
        after = {t:table_digest(conn,t) for t in PROTECTED if t in tables}
        if before != after:
            raise ValueError("Sanitization changed measurement records")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
    data = path.read_bytes()
    if any(v.encode() in data for v in secrets):
        raise ValueError("Credential data remains after sanitization")
    return dict(original_sha256=original_sha, published_sha256=sha(path),
                environment_variable_names_removed=len(environment), string_rows_redacted=len(changed),
                preserved_measurement_table_sha256=after)


def refresh_checksums(root):
    (root / "SHA256SUMS").write_text("".join(
        f"{sha(p)}  {p.relative_to(root)}\n" for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"))


def prepare(root, private_backup):
    if root.resolve() == private_backup.resolve() or root.resolve() in private_backup.resolve().parents:
        raise ValueError("Private backup must be outside the publication directory")
    shutil.copytree(root, private_backup)
    changes = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("profiler") != "nsys":
            continue
        capture = manifest_path.parent
        record = dict(databases={}, omitted_reports={})
        for p in sorted((capture / "traces").glob("*.sqlite")):
            relative = str(p.relative_to(capture))
            record["databases"][relative] = sanitize_database(p)
            manifest["hashes"][relative] = sha(p)
        for p in sorted((capture / "traces").glob("*.nsys-rep")):
            relative = str(p.relative_to(capture))
            expected = sha(p)
            if sha(private_backup / p.relative_to(root)) != expected:
                raise ValueError("Original capture backup verification failed")
            record["omitted_reports"][relative] = dict(original_sha256=expected,
                reason="Original proprietary capture retained locally; use the sanitized SQLite export.")
            p.unlink()
            manifest["hashes"].pop(relative)
        manifest["publication_sanitization"] = record
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        changes[str(capture.relative_to(root))] = record
    publication = dict(policy="Environment metadata removed; original .nsys-rep files retained privately.",
                       capture_changes=changes, measurement_records_preserved=True)
    (root / "PUBLICATION.json").write_text(json.dumps(publication,indent=2)+"\n")
    refresh_checksums(root)
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--private-backup",type=Path,required=True)
    args = parser.parse_args()
    result = prepare(args.root,args.private_backup)
    print(json.dumps(dict(sanitized_captures=len(result["capture_changes"]),
                          measurement_records_preserved=True)))


if __name__ == "__main__":
    main()
