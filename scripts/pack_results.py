"""Package run metadata and metrics; model weights are opt-in."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def pack_runs(runs_dir: Path, output: Path, include_model: bool = False) -> int:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(runs_dir.rglob("*")):
            if not file_path.is_file() or output.resolve() == file_path.resolve():
                continue
            relative = file_path.relative_to(runs_dir)
            if any(part in {"features", ".cache"} for part in relative.parts):
                continue
            if not include_model and (
                "best_model" in relative.parts or "tokenizer" in relative.parts
            ):
                continue
            archive.write(file_path, Path("runs") / relative)
            added += 1
    if not added:
        raise ValueError(f"No run artifacts were found under {runs_dir}.")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/run_bundle.zip"))
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    count = pack_runs(args.runs, args.out, args.include_model)
    print(f"Packed {count} files to {args.out}")


if __name__ == "__main__":
    main()
