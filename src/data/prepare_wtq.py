"""
prepare_wtq.py — Download and convert WikiTableQuestions to RSAT format.

Downloads the original zip from GitHub, extracts to data/raw/, reads the
TSV files directly, and writes unified JSONL to data/processed/.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

from src.data.data_formats import RSATExample, Table

_DATA_URL = (
    "https://github.com/ppasupat/WikiTableQuestions/releases/"
    "download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
)

_ZIP_NAME = "WikiTableQuestions-1.0.2-compact.zip"
_ROOT_FOLDER = "WikiTableQuestions"

# Mapping from our split names to file names inside data/
_SPLIT_FILES = {
    "train": "random-split-1-train.tsv",
    "validation": "random-split-1-dev.tsv",
    "test": "pristine-unseen-tables.tsv",
}


# ═════════════════════════════════════════════════════════════════════════════
# Download + extraction
# ═════════════════════════════════════════════════════════════════════════════


def download_and_extract(raw_dir: Path) -> Path:
    """
    Download the WTQ zip and extract it to raw_dir.
    Returns the path to the WikiTableQuestions root directory.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / _ZIP_NAME
    root_path = raw_dir / _ROOT_FOLDER

    # Skip download if already extracted
    if root_path.exists() and (root_path / "data").exists():
        print(f"  Already extracted at {root_path}")
        return root_path

    # Download
    if not zip_path.exists():
        print(f"  Downloading WTQ from GitHub ...")
        urllib.request.urlretrieve(_DATA_URL, str(zip_path))
        print(f"  Saved → {zip_path}")
    else:
        print(f"  Zip already exists at {zip_path}")

    # Extract
    print(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(raw_dir))
    print(f"  Extracted → {root_path}")

    return root_path


# ═════════════════════════════════════════════════════════════════════════════
# Table reading
# ═════════════════════════════════════════════════════════════════════════════


def read_table(table_name: str, root_dir: Path) -> Table:
    """
    Read a table file from the WTQ directory.

    table_name is something like "csv/200-csv/42.csv" — the loading script
    replaces .csv → .tsv to use the normalised version.
    """
    table_name = table_name.replace(".csv", ".tsv")
    table_path = root_dir / table_name

    with open(table_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def parse_line(line: str) -> list[str]:
        return [cell.replace("\n", " ").strip() for cell in line.strip("\n").split("\t")]

    headers = parse_line(lines[0])
    rows = [parse_line(line) for line in lines[1:] if line.strip()]

    return Table(headers=headers, rows=rows)


# ═════════════════════════════════════════════════════════════════════════════
# Conversion
# ═════════════════════════════════════════════════════════════════════════════


def convert_split(split: str, root_dir: Path) -> list[RSATExample]:
    """Read a WTQ split TSV and convert every row to RSATExample."""
    tsv_name = _SPLIT_FILES[split]
    tsv_path = root_dir / "data" / tsv_name

    print(f"  Reading {tsv_path} ...")

    examples: list[RSATExample] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        # First line is the TSV header — skip it
        next(f)

        for i, line in enumerate(f):
            line = line.strip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            example_id, question, table_name, answer_str = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
            )

            # Answers are pipe-separated
            answers = answer_str.split("|")
            primary_answer = answers[0] if answers else ""

            # Read the table
            try:
                table = read_table(table_name, root_dir)
            except Exception as exc:
                print(f"    ⚠ Skipping {example_id}: could not read table {table_name} ({exc})")
                continue

            examples.append(
                RSATExample(
                    id=f"wtq_{split}_{i}",
                    question=question,
                    table=table,
                    answer=primary_answer,
                    source_dataset="wtq",
                    reasoning_trace=None,
                    metadata={
                        "all_answers": answers,
                        "original_id": example_id,
                    },
                )
            )

    print(f"  → {len(examples)} examples")
    return examples


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def save_jsonl(examples: list[RSATExample], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")
    print(f"  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare WTQ dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and extract
    root_dir = download_and_extract(raw_dir)

    # Step 2: Convert each split
    for split in ("train", "validation", "test"):
        examples = convert_split(split, root_dir)
        save_jsonl(examples, out_dir / f"wtq_{split}.jsonl")


if __name__ == "__main__":
    main()