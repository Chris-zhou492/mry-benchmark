"""Download and verify the pinned public benchmark files in this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "datasets" / "manifest.json"
USER_AGENT = "MRY-Benchmark-dataset-manager/0.1"
CHUNK_SIZE = 1024 * 1024


def read_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_destination(relative_path: str) -> Path:
    """Resolve a manifest path and reject path traversal outside the repository."""
    candidate = (ROOT / Path(*PurePosixPath(relative_path).parts)).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"Path escapes repository root: {relative_path}")
    return candidate


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def download(url: str, destination: Path, metadata: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    print(f"Downloading {destination.relative_to(ROOT)}")
    try:
        with urllib.request.urlopen(request(url), timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(CHUNK_SIZE):
                out.write(chunk)
                size += len(chunk)
                digest.update(chunk)
        verify_metadata(metadata, size, digest.hexdigest(), temporary)
        temporary.replace(destination)
        print(f"  verified {size:,} bytes; SHA-256 {digest.hexdigest()}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_metadata(metadata: dict[str, Any], size: int, sha256: str, path: Path) -> None:
    expected_size = metadata.get("size")
    expected_hash = metadata.get("sha256")
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Size mismatch for {path.name}: expected {expected_size}, got {size}")
    if expected_hash is not None and sha256.lower() != expected_hash.lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected_hash}, got {sha256}")


def bundled_path(file_entry: dict[str, Any]) -> Path | None:
    target = file_entry.get("bundle")
    return safe_destination(f"datasets/bundled/{target}") if target else None


def download_path(dataset_key: str, file_entry: dict[str, Any]) -> Path:
    rel = file_entry.get("download_path") or PurePosixPath(dataset_key, file_entry["path"])
    return safe_destination(f"datasets/downloads/{rel}")


def get_json(url: str) -> Any:
    with urllib.request.urlopen(request(url), timeout=60) as response:
        return json.load(response)


def git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_personamem_histories(dataset_key: str, dataset: dict[str, Any]) -> None:
    settings = dataset["dynamic_history"]
    # This pinned tree snapshot returns all 981 128K-history entries in its
    # 1,000-entry default page. The exact count is checked below to catch drift.
    items = get_json(settings["api"])
    files = [item for item in items if item.get("type") == "file" and item.get("path", "").startswith(settings["prefix"])]
    files.sort(key=lambda item: item["path"])
    expected_count = settings["expected_count"]
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} pinned histories, upstream returned {len(files)}")

    revision = dataset["revision"]
    repo_id = "bowen-upenn/PersonaMem-v2"
    destination_root = safe_destination(f"datasets/downloads/{settings['destination']}")
    print(f"Downloading {len(files)} PersonaMem v2 history files from revision {revision}")
    for index, item in enumerate(files, start=1):
        relative = item["path"][len(settings["prefix"]) :]
        destination = destination_root / Path(*PurePosixPath(relative).parts)
        encoded_path = urllib.parse.quote(item["path"], safe="/")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}?download=true"
        temporary = destination.with_suffix(destination.suffix + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request(url), timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(CHUNK_SIZE):
                    out.write(chunk)
                    size += len(chunk)
                    digest.update(chunk)
            if size != item["size"]:
                raise ValueError(f"Size mismatch for {item['path']}: expected {item['size']}, got {size}")
            lfs = item.get("lfs")
            if lfs:
                actual, expected = digest.hexdigest(), lfs["oid"]
            else:
                actual, expected = git_blob_sha1(temporary, size), item["oid"]
            if actual.lower() != expected.lower():
                raise ValueError(f"Upstream object hash mismatch for {item['path']}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if index % 25 == 0 or index == len(files):
            print(f"  downloaded {index}/{len(files)} histories")


def list_datasets(manifest: dict[str, Any]) -> None:
    for key, dataset in manifest["datasets"].items():
        bundled = sum("bundle" in item for item in dataset["files"])
        extra = " + pinned history tree" if "dynamic_history" in dataset else ""
        print(f"{key:24} {dataset['title']} ({bundled} bundled file(s){extra})")


def plan_dataset(manifest: dict[str, Any], key: str) -> None:
    if key not in manifest["datasets"]:
        raise ValueError(f"Unknown dataset key: {key}")
    dataset = manifest["datasets"][key]
    files = dataset["files"]
    total_size = sum(item.get("size", 0) for item in files)
    file_count = len(files)
    if "dynamic_history" in dataset:
        settings = dataset["dynamic_history"]
        items = get_json(settings["api"])
        histories = [item for item in items if item.get("type") == "file" and item.get("path", "").startswith(settings["prefix"])]
        if len(histories) != settings["expected_count"]:
            raise ValueError(f"Expected {settings['expected_count']} pinned histories, upstream returned {len(histories)}")
        file_count += len(histories)
        total_size += sum(item.get("size", 0) for item in histories)
    print(f"{key}: {file_count} file(s), approximately {total_size / (1024 ** 3):.2f} GiB")


def verify_bundled(manifest: dict[str, Any], selected: str | None = None) -> None:
    checked = 0
    missing: list[str] = []
    mismatches: list[str] = []
    for key, dataset in manifest["datasets"].items():
        if selected and key != selected:
            continue
        for item in dataset["files"]:
            destination = bundled_path(item)
            if destination is None:
                continue
            if not destination.is_file():
                missing.append(str(destination.relative_to(ROOT)))
                continue
            size, sha256 = digest_file(destination)
            try:
                verify_metadata(item, size, sha256, destination)
            except ValueError as error:
                mismatches.append(str(error))
                continue
            checked += 1
            print(f"OK {destination.relative_to(ROOT)} ({size:,} bytes, SHA-256 {sha256})")
    if missing or mismatches:
        for path in missing:
            print(f"MISSING {path}", file=sys.stderr)
        for mismatch in mismatches:
            print(f"ERROR {mismatch}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {checked} bundled file(s). Downloaded files are checked during retrieval.")


def download_dataset(manifest: dict[str, Any], key: str) -> None:
    datasets = manifest["datasets"]
    if key not in datasets:
        raise ValueError(f"Unknown dataset key: {key}")
    dataset = datasets[key]
    for item in dataset["files"]:
        target = bundled_path(item) or download_path(key, item)
        if target.is_file():
            size, sha256 = digest_file(target)
            verify_metadata(item, size, sha256, target)
            print(f"Already present and verified: {target.relative_to(ROOT)}")
            continue
        url = item["url"]
        # Quote URL path spaces while leaving the scheme, separators, and query unchanged.
        split = urllib.parse.urlsplit(url)
        quoted_path = urllib.parse.quote(urllib.parse.unquote(split.path), safe="/:@")
        download(urllib.parse.urlunsplit((split.scheme, split.netloc, quoted_path, split.query, split.fragment)), target, item)
    if "dynamic_history" in dataset:
        download_personamem_histories(key, dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list available dataset keys")
    plan_parser = subparsers.add_parser("plan", help="show the pinned file count and approximate download size")
    plan_parser.add_argument("dataset", help="dataset key shown by the list command")
    verify_parser = subparsers.add_parser("verify", help="verify bundled files")
    verify_parser.add_argument("--dataset", help="verify one dataset key")
    download_parser = subparsers.add_parser("download", help="download one pinned dataset")
    download_parser.add_argument("dataset", help="dataset key shown by the list command")
    args = parser.parse_args()

    manifest = read_manifest()
    if args.command == "list":
        list_datasets(manifest)
    elif args.command == "plan":
        plan_dataset(manifest, args.dataset)
    elif args.command == "verify":
        verify_bundled(manifest, args.dataset)
    elif args.command == "download":
        download_dataset(manifest, args.dataset)


if __name__ == "__main__":
    try:
        main()
    except (OSError, urllib.error.URLError, ValueError, KeyError) as error:
        print(f"dataset manager error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
