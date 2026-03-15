"""
prepare_tabfact.py — Download and convert TabFact to RSAT format.

TabFact is a fact-verification dataset over Wikipedia tables. We repurpose
it for evaluating citation faithfulness: the model must verify a claim and
cite the supporting / refuting cells.

Downloads from the original GitHub repo:
    https://github.com/wenhuchen/Table-Fact-Checking
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

from src.data.data_formats import RSATExample, Table

_REPO_ZIP_URL = (
    "https://github.com/wenhuchen/Table-Fact-Checking/archive/refs/heads/master.zip"
)

_ZIP_NAME = "Table-Fact-Checking-master.zip"
_ROOT_FOLDER = "Table-Fact-Checking-master"

_SPLIT_FILES = {
    "train": "tokenized_data/train_examples.json",
    "validation": "tokenized_data/val_examples.json",
    "test": "tokenized_data/test_examples.json",
}


# ═════════════════════════════════════════════════════════════════════════════
# Download + extraction
# ═════════════════════════════════════════════════════════════════════════════


def download_and_extract(raw_dir: Path) -> Path:
    """
    Download the TabFact repo zip and extract.
    Returns path to the extracted root directory.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / _ZIP_NAME
    root_path = raw_dir / _ROOT_FOLDER

    if root_path.exists() and (root_path / "data" / "all_csv").exists():
        print(f"  Already extracted at {root_path}")
        return root_path

    if not zip_path.exists():
        print(f"  Downloading TabFact repo...")
        urllib.request.urlretrieve(_REPO_ZIP_URL, str(zip_path))
        print(f"  Saved → {zip_path}")
    else:
        print(f"  Zip already exists at {zip_path}")

    print(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(raw_dir))
    print(f"  Extracted → {root_path}")

    return root_path


# ═════════════════════════════════════════════════════════════════════════════
# Table reading
# ═════════════════════════════════════════════════════════════════════════════


def read_csv_table(csv_path: Path) -> Table | None:
    """Read a TabFact CSV file into a Table object."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="#")
            rows_raw = list(reader)

        if len(rows_raw) < 2:
            return None

        headers = [str(c).strip() for c in rows_raw[0]]
        rows = [[str(c).strip() for c in row] for row in rows_raw[1:]]

        return Table(headers=headers, rows=rows)
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Statement cleaning — strip entity linking markup
# ═════════════════════════════════════════════════════════════════════════════

# Pattern: #entity_text;row_idx,col_idx#  →  entity_text
_ENTITY_PATTERN = re.compile(r"#([^;#]+);[^#]+#")


def clean_statement(statement: str) -> str:
    """Remove entity linking markup from a tokenized statement."""
    cleaned = _ENTITY_PATTERN.sub(r"\1", statement)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
# Conversion
# ═════════════════════════════════════════════════════════════════════════════


def convert_split(
    split: str,
    root_dir: Path,
    max_examples: int | None = None,
) -> list[RSATExample]:
    """Read a TabFact split JSON and convert to RSATExample list."""
    json_path = root_dir / _SPLIT_FILES[split]
    csv_dir = root_dir / "data" / "all_csv"

    print(f"  Reading {json_path} ...")

    with open(json_path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)

    examples: list[RSATExample] = []
    example_idx = 0

    for table_id, entry in data.items():
        if max_examples is not None and example_idx >= max_examples:
            break

        # entry = [statements_list, labels_list, pos_tags_list, caption]
        statements: list[str] = entry[0]
        labels: list[int] = entry[1]
        # entry[2] = POS tags (unused)
        caption: str = entry[3] if len(entry) > 3 else ""

        # Read the table CSV
        csv_path = csv_dir / table_id
        table = read_csv_table(csv_path)
        if table is None:
            continue

        if caption:
            table.caption = caption

        # Create one example per (statement, label) pair
        for stmt, label in zip(statements, labels):
            if max_examples is not None and example_idx >= max_examples:
                break

            clean_stmt = clean_statement(stmt)
            answer = "entailed" if label == 1 else "refuted"

            examples.append(
                RSATExample(
                    id=f"tabfact_{split}_{example_idx}",
                    question=f"Verify: {clean_stmt}",
                    table=table,
                    answer=answer,
                    source_dataset="tabfact",
                    reasoning_trace=None,
                    metadata={
                        "table_id": table_id,
                        "label": label,
                        "raw_statement": stmt,
                        "caption": caption,
                    },
                )
            )
            example_idx += 1

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
    parser = argparse.ArgumentParser(description="Download and prepare TabFact dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw/tabfact")
    parser.add_argument(
        "--max_train",
        type=int,
        default=20000,
        help="Cap training examples (TabFact has ~92k train).",
    )
    parser.add_argument(
        "--show_example",
        action="store_true",
        help="Print one processed example per split for inspection.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and extract
    root_dir = download_and_extract(raw_dir)

    # Step 2: Convert each split
    for split, cap in [("train", args.max_train), ("validation", None), ("test", None)]:
        examples = convert_split(split, root_dir, max_examples=cap)
        save_jsonl(examples, out_dir / f"tabfact_{split}.jsonl")

        if args.show_example and examples:
            print(f"\n  ── Sample ({split}) ──")
            print(json.dumps(examples[0].to_dict(), indent=2, ensure_ascii=False))
            print()


if __name__ == "__main__":
    main()