#!/usr/bin/env python3
"""Validate the complete experimental series before editing a clean checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
parser.add_argument("--check", action="store_true", help="validate without editing sources")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "manifests/source.json").read_text())


def git(*command, env=None):
    return subprocess.check_output(["git", "-C", str(args.checkout), *command], env=env).decode().strip()


if git("rev-parse", "HEAD") != manifest["upstream_commit"]:
    parser.error("checkout must be at the exact upstream_commit in manifests/source.json")
if git("status", "--porcelain", "--untracked-files=all"):
    parser.error("checkout must be clean, including untracked files")
patches = []
for entry in manifest["patches"]:
    patch = root / "patches" / entry["file"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
        parser.error("patch checksum mismatch: " + entry["file"])
    patches.append(patch)
with tempfile.TemporaryDirectory(prefix="apex-v2-patch-check-") as temp:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
    git("read-tree", "HEAD", env=env)
    for patch in patches:
        git("apply", "--cached", str(patch), env=env)
print(f"Validated {len(patches)} patches against the pinned base.")
if not args.check:
    for patch in patches:
        git("apply", str(patch))
    print("Applied experimental sources. No build or service was started.")
