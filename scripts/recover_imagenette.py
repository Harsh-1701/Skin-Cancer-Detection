"""Safely restore the ImageNette files referenced by the V3 dataset CSV.

The script never downloads data and never changes the dataset CSV. First run a
dry run against an already-extracted archive, for example:

    python scripts\recover_imagenette.py --source-root D:\datasets\imagenette2 --dry-run

Recovery is intentionally Windows-only because promotion relies on Windows
os.rename() refusing an existing destination. Dry-run performs no writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_ROOT / "dataset" / "combined_dataset.csv"
DESTINATION = PROJECT_ROOT / "dataset" / "train" / "unknown"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "paper2" / "imagenette_recovery_manifest.csv"
STAGING_PARENT = PROJECT_ROOT / ".imagenette_recovery_staging"
REQUIRED_COLUMNS = {"image_name", "dx", "source"}
SHA256_CHUNK_SIZE = 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
OWNER_MARKER_NAME = ".imagenette_recovery_owner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and safely restore the CSV-defined ImageNette subset."
    )
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of an already-extracted ImageNette directory to scan recursively.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform matching and source-hash checks without writing anything.")
    parser.add_argument("--archive-sha256", default=None,
                        help="Optional known SHA-256 of the source archive (64 hexadecimal characters).")
    parser.add_argument("--archive-id", default="unavailable",
                        help="Optional archive URL or identifier, recorded as supplied.")
    return parser.parse_args()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def paths_overlap(a: Path, b: Path) -> bool:
    """True if either path is equal to, inside, or an ancestor of the other."""
    a_resolved, b_resolved = a.resolve(), b.resolve()
    return is_within(a_resolved, b_resolved) or is_within(b_resolved, a_resolved)


def assert_project_path(path: Path) -> None:
    if not is_within(path.resolve(), PROJECT_ROOT.resolve()):
        raise ValueError(f"Project-managed path escapes V3 project root: {path}")


def validate_archive_sha256(value: str | None) -> str:
    if value is None:
        return "unavailable"
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError("--archive-sha256 must be exactly 64 hexadecimal characters.")
    return value.lower()


def required_filenames(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError("Dataset CSV is missing columns: " + ", ".join(sorted(missing_columns)))
    subset = dataframe.loc[
        (dataframe["source"] == "IMAGENETTE") & (dataframe["dx"] == "unknown"),
        "image_name",
    ]
    if len(subset) != 1000:
        raise ValueError(f"Expected exactly 1,000 ImageNette rows; found {len(subset)}.")
    if subset.isna().any():
        raise ValueError("ImageNette rows contain missing image_name values.")
    names = subset.astype(str).tolist()
    if len(set(names)) != 1000:
        raise ValueError("Expected exactly 1,000 unique ImageNette filenames.")
    # Filenames must be basenames, not paths or traversal expressions.
    if any(Path(n).name != n or n in {"", ".", ".."} for n in names):
        raise ValueError("CSV image_name values must be plain filenames, not paths.")
    return names


def build_source_index(source_root: Path, expected_names: set[str]) -> DefaultDict[str, list[Path]]:
    if not source_root.is_dir():
        raise NotADirectoryError(f"--source-root is not a directory: {source_root}")
    matches: DefaultDict[str, list[Path]] = defaultdict(list)
    for directory, _, filenames in os.walk(source_root):
        directory_path = Path(directory)
        for filename in filenames:
            if filename in expected_names:
                matches[filename].append(directory_path / filename)
    return matches


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_destination_entries(destination: Path) -> list[Path]:
    if not destination.exists():
        return []
    if not destination.is_dir():
        raise NotADirectoryError(f"Destination exists but is not a directory: {destination}")
    return list(destination.iterdir())


def verify_direct_filename_set(
    directory: Path, expected_names: set[str], expected_hashes: dict[str, str]
) -> None:
    entries = direct_destination_entries(directory)
    # The destination may contain our unique ownership marker until recovery
    # and manifest publication have both succeeded.
    image_entries = [entry for entry in entries if entry.name != OWNER_MARKER_NAME]
    non_files = [entry for entry in image_entries if not entry.is_file()]
    names = {entry.name for entry in image_entries if entry.is_file()}
    if non_files or names != expected_names or len(image_entries) != len(expected_names):
        raise RuntimeError("Directory verification failed: expected exactly the required files.")
    for name in expected_names:
        if sha256(directory / name) != expected_hashes[name]:
            raise RuntimeError(f"SHA-256 verification failed for {directory / name}")


def print_validation_summary(expected_count: int, matched_count: int,
                             ambiguous: dict[str, list[Path]], missing: list[str],
                             destination_entries: list[Path]) -> None:
    print("Recovery validation summary")
    print(f"  Expected files: {expected_count}")
    print(f"  Matched files: {matched_count}")
    print(f"  Ambiguous names: {len(ambiguous)}")
    print(f"  Missing names: {len(missing)}")
    print(f"  Existing destination entries: {len(destination_entries)}")
    if missing:
        print("\nMissing source basenames (first 20):")
        for name in missing[:20]:
            print(f"  {name}")
    if ambiguous:
        print("\nAmbiguous source basenames (first 20):")
        for name, paths in list(sorted(ambiguous.items()))[:20]:
            print(f"  {name}")
            for path in paths:
                print(f"    {path}")


def make_directory_tree(path: Path, created_directories: list[Path]) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing parent found for {path}")
        current = parent
    if not current.is_dir():
        raise NotADirectoryError(f"Parent path is not a directory: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory}")
        else:
            created_directories.append(directory)


def cleanup_created_directories(created_directories: list[Path]) -> None:
    for directory in reversed(created_directories):
        try:
            directory.rmdir()  # Empty directories only.
        except OSError:
            pass


def same_file_identity(path: Path, expected_stat: os.stat_result) -> bool:
    try:
        current = path.stat()
    except OSError:
        return False
    # st_ino is available on supported Windows Python versions; include device
    # to reduce the chance of treating a replacement as the original artifact.
    return current.st_dev == expected_stat.st_dev and current.st_ino == expected_stat.st_ino


def promote_staging_directory(stage_dir: Path, destination: Path) -> None:
    if os.name != "nt":
        raise OSError("Safe no-replace directory promotion is implemented only for Windows.")
    if stage_dir.drive.casefold() != destination.drive.casefold():
        raise OSError("Staging and destination must be on the same Windows volume.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final destination: {destination}")
    # Windows os.rename() fails if the destination already exists. Staging and
    # destination share a project-root volume. This is not a defense against a
    # hostile process with permission to alter the parent directory.
    os.rename(stage_dir, destination)


def write_manifest_atomically(rows: list[dict[str, str]],
                              created_directories: list[Path]) -> tuple[os.stat_result, bool]:
    """Publish manifest without replacement; return its identity and temp-cleanup status."""
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {MANIFEST_PATH}")
    make_directory_tree(MANIFEST_PATH.parent, created_directories)
    temporary_manifest = MANIFEST_PATH.parent / f".{MANIFEST_PATH.name}.{uuid.uuid4().hex}.tmp"
    fieldnames = [
        "image_name", "source_relative_path", "destination_path", "source_sha256",
        "destination_sha256", "archive_sha256", "archive_url_or_identifier",
        "recovery_timestamp_utc",
    ]
    published = False
    try:
        with temporary_manifest.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())

        # Hard-link creation is atomic and fails if the final name already exists.
        os.link(temporary_manifest, MANIFEST_PATH)
        published = True
        manifest_stat = MANIFEST_PATH.stat()
        try:
            temporary_manifest.unlink()
            temp_cleanup_ok = True
        except OSError as error:
            # The published manifest is already valid. Do not turn a temporary
            # cleanup failure into a false failed-recovery state.
            print(f"Warning: manifest was published, but temporary file cleanup failed: {error}",
                  file=sys.stderr)
            temp_cleanup_ok = False
        return manifest_stat, temp_cleanup_ok
    except Exception:
        # Never unlink a final manifest in this handler. If publication occurred,
        # preserve it and let the caller reconcile based on the returned/observed state.
        try:
            if temporary_manifest.exists():
                temporary_manifest.unlink()
        except OSError:
            pass
        if published:
            raise RuntimeError(
                "Manifest publication may have succeeded; preserve destination and inspect manifest."
            )
        raise


def remove_owned_destination(destination: Path, owner_token: str,
                             expected_stat: os.stat_result) -> bool:
    """Delete only if directory identity and its unique owner marker both match."""
    if not same_file_identity(destination, expected_stat):
        return False
    marker = destination / OWNER_MARKER_NAME
    try:
        if marker.read_text(encoding="utf-8") != owner_token:
            return False
    except OSError:
        return False
    # Recheck identity immediately before deletion. Filesystem races cannot be
    # eliminated portably between this check and rmtree; the marker is a guard,
    # not a security boundary against a hostile process with write access.
    if not same_file_identity(destination, expected_stat):
        return False
    shutil.rmtree(destination)
    return True


def main() -> int:
    args = parse_args()
    stage_dir: Path | None = None
    promoted_destination = False
    destination_stat: os.stat_result | None = None
    owner_token = uuid.uuid4().hex
    created_directories: list[Path] = []
    manifest_published = False

    try:
        for project_path in (DATASET_CSV, DESTINATION, MANIFEST_PATH, STAGING_PARENT):
            assert_project_path(project_path)

        archive_sha256 = validate_archive_sha256(args.archive_sha256)
        if not args.source_root.exists():
            raise FileNotFoundError(f"Supplied --source-root does not exist: {args.source_root}")
        source_root = args.source_root.resolve()

        if source_root == PROJECT_ROOT.resolve():
            raise ValueError("--source-root must not be the V3 project root.")
        protected_paths = (DESTINATION.resolve(), STAGING_PARENT.resolve(), DATASET_CSV.resolve(),
                           MANIFEST_PATH.resolve())
        overlaps = [p for p in protected_paths if paths_overlap(source_root, p)]
        if overlaps:
            raise ValueError(
                "--source-root overlaps protected V3 paths: " + ", ".join(str(p) for p in overlaps)
            )

        expected_names = set(required_filenames(DATASET_CSV))
        index = build_source_index(source_root, expected_names)
        missing = sorted(expected_names.difference(index))
        ambiguous = {name: sorted(paths) for name, paths in index.items() if len(paths) > 1}
        matched = {name: paths[0] for name, paths in index.items() if len(paths) == 1}
        existing_entries = direct_destination_entries(DESTINATION) if DESTINATION.exists() else []

        print_validation_summary(len(expected_names), len(matched), ambiguous, missing, existing_entries)
        if missing or ambiguous:
            raise RuntimeError("No recovery performed: every expected basename must have exactly one match.")
        if DESTINATION.exists():
            raise RuntimeError(f"No recovery performed: final destination already exists: {DESTINATION}")
        if MANIFEST_PATH.exists():
            raise FileExistsError(f"No recovery performed: manifest already exists: {MANIFEST_PATH}")

        source_hashes = {name: sha256(path) for name, path in matched.items()}
        if args.dry_run:
            print("\nDry run succeeded: source matching and hashing passed; nothing was written.")
            return 0

        make_directory_tree(STAGING_PARENT, created_directories)
        stage_dir = STAGING_PARENT / f"recovery-{uuid.uuid4().hex}"
        stage_dir.mkdir()
        marker = stage_dir / OWNER_MARKER_NAME
        marker.write_text(owner_token, encoding="utf-8")

        for name in sorted(expected_names):
            staged_path = stage_dir / name
            shutil.copy2(matched[name], staged_path)
            if sha256(staged_path) != source_hashes[name]:
                raise RuntimeError(f"Staging SHA-256 verification failed for {name}")

        # Verify image set separately from the ownership marker.
        marker.unlink()
        verify_direct_filename_set(stage_dir, expected_names, source_hashes)
        marker.write_text(owner_token, encoding="utf-8")

        make_directory_tree(DESTINATION.parent, created_directories)
        promote_staging_directory(stage_dir, DESTINATION)
        stage_dir = None
        promoted_destination = True
        destination_stat = DESTINATION.stat()

        verify_direct_filename_set(DESTINATION, expected_names, source_hashes)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest_rows = [
            {
                "image_name": name,
                "source_relative_path": str(matched[name].relative_to(source_root)),
                "destination_path": str(DESTINATION / name),
                "source_sha256": source_hashes[name],
                "destination_sha256": source_hashes[name],
                "archive_sha256": archive_sha256,
                "archive_url_or_identifier": args.archive_id,
                "recovery_timestamp_utc": timestamp,
            }
            for name in sorted(expected_names)
        ]

        manifest_stat, _ = write_manifest_atomically(manifest_rows, created_directories)
        manifest_published = same_file_identity(MANIFEST_PATH, manifest_stat)
        if not manifest_published:
            raise RuntimeError("Published manifest identity could not be verified; inspect recovery outputs.")

        # Manifest is now published and destination verified. Remove marker only
        # after successful publication; failure to remove it is harmless and reported.
        try:
            (DESTINATION / OWNER_MARKER_NAME).unlink()
        except OSError as error:
            print(f"Warning: recovery succeeded but owner marker could not be removed: {error}",
                  file=sys.stderr)

        print("\nRecovery completed successfully.")
        print(f"  Copied and hash-verified files: {len(expected_names)}")
        print(f"  Destination: {DESTINATION}")
        print(f"  Manifest: {MANIFEST_PATH}")
        return 0

    except Exception as error:
        # If a manifest was published, preserve both it and the verified destination
        # for manual reconciliation. Never delete a destination after a manifest
        # may have been published.
        if promoted_destination and not manifest_published and not MANIFEST_PATH.exists():
            if destination_stat is not None:
                try:
                    removed = remove_owned_destination(DESTINATION, owner_token, destination_stat)
                    if not removed:
                        print(
                            "\nRollback skipped: destination ownership could not be verified. "
                            "Inspect it manually.",
                            file=sys.stderr,
                        )
                except OSError as cleanup_error:
                    print(f"\nRollback failed; inspect destination manually: {cleanup_error}",
                          file=sys.stderr)
        elif promoted_destination:
            print(
                "\nDestination preserved because a manifest may exist or publication state is uncertain. "
                "Inspect destination and manifest manually.",
                file=sys.stderr,
            )
        print(f"\nRecovery failed: {error}", file=sys.stderr)
        return 1

    finally:
        if stage_dir is not None:
            try:
                shutil.rmtree(stage_dir)
            except OSError as cleanup_error:
                print(f"\nStaging cleanup failed; inspect {stage_dir}: {cleanup_error}",
                      file=sys.stderr)
        cleanup_created_directories(created_directories)


if __name__ == "__main__":
    raise SystemExit(main())
