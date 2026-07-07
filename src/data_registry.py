"""Content-addressed data version manifests."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import resolve_path
from src.utils import read_json, write_json


def _utc_now() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _project_relative(path: Path) -> str:
    """Format a resolved path relative to the project root when possible."""
    resolved = path.resolve()
    project_root = resolve_path(".").resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _remote_uri(config: dict[str, Any], relative_path: str) -> str | None:
    """Map a project-relative artifact path to configured remote storage."""
    versioning_config = config.get("data_versioning", {})
    remote_base = os.getenv("DATA_REMOTE_URI", str(versioning_config.get("remote_uri", ""))).strip()
    if not remote_base:
        return None
    return f"{remote_base.rstrip('/')}/{relative_path}"


def _sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible identity metadata for a local file."""
    resolved_path = resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {resolved_path}")

    stat = resolved_path.stat()
    relative_path = _project_relative(resolved_path)
    fingerprint: dict[str, Any] = {
        "path": relative_path,
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved_path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    remote_uri = _remote_uri(config, relative_path)
    if remote_uri:
        fingerprint["remote_uri"] = remote_uri
    return fingerprint


def write_data_version_manifest(
    config: dict[str, Any],
    dataset_version: str,
    dataset_path: str | Path,
    source_paths: list[str | Path] | None = None,
    metadata_path: str | Path | None = None,
    stage: str | None = None,
    rows: int | None = None,
    columns: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write per-version and aggregate data registry manifests."""
    versioning_config = config.get("data_versioning", {})
    checksum_algorithm = str(versioning_config.get("checksum_algorithm", "sha256"))
    if checksum_algorithm != "sha256":
        raise ValueError("Only sha256 data version checksums are currently supported.")

    registry_dir = resolve_path(versioning_config.get("registry_dir", "data/registry"))
    registry_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "stage": stage,
        "created_utc": _utc_now(),
        "checksum_algorithm": checksum_algorithm,
        "artifact": file_fingerprint(dataset_path, config),
        "sources": [
            file_fingerprint(source_path, config)
            for source_path in source_paths or []
            if resolve_path(source_path).exists()
        ],
    }
    if metadata_path is not None and resolve_path(metadata_path).exists():
        manifest["metadata"] = file_fingerprint(metadata_path, config)
    if rows is not None:
        manifest["rows"] = int(rows)
    if columns is not None:
        manifest["columns"] = int(columns)
    if extra:
        manifest["extra"] = extra

    version_manifest_path = registry_dir / f"{dataset_version}.json"
    write_json(version_manifest_path, manifest)

    aggregate_path = registry_dir / "manifest.json"
    if aggregate_path.exists():
        aggregate_manifest = read_json(aggregate_path)
    else:
        aggregate_manifest = {
            "registry_format": "telco-data-registry-v1",
            "checksum_algorithm": checksum_algorithm,
            "versions": {},
        }

    aggregate_manifest["updated_utc"] = _utc_now()
    aggregate_manifest["latest_version"] = dataset_version
    aggregate_manifest.setdefault("versions", {})[dataset_version] = manifest
    write_json(aggregate_path, aggregate_manifest)
    return manifest
