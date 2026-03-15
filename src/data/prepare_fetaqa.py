"""
prepare_fetaqa.py — Download and convert FeTaQA to RSAT format.

FeTaQA requires free-form answers grounded in table evidence,
making it ideal for evaluating attribution quality.

Downloads JSONL files directly from the Yale-LILY GitHub repo:
    https://github.com/Yale-LILY/FeTaQA
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from src.data.data_formats import RSATExample, Table

_BASE_URL = (
    "https://raw.githubusercontent.com/Yale-LILY/FeTaQA/refs/heads/main/data"
)

_SPLIT_FILES = {
    "train": "fetaQA-v1_train.jsonl",
    "validation": "fetaQA-v1_dev.jsonl",
    "test": "fetaQA-v1_test.jsonl",
}


# ═════════════════════════════════════════════════════════════════════════════
# Download
# ═════════════════════════════════════════════════════════════════════════════


def download_split(split: str, raw_dir: Path) -> Path:
    """Download a FeTaQA split JSONL file to raw_dir. Returns local path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _SPLIT_FILES[split]
    local_path = raw_dir / filename

    if local_path.exists():
        print(f"  Already downloaded: {local_path}")
        return local_path

    url = f"{_BASE_URL}/{filename}"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, str(local_path))
    print(f"  Saved → {local_path}")
    return local_path


# ═════════════════════════════════════════════════════════════════════════════
# Conversion
# ═════════════════════════════════════════════════════════════════════════════


def convert_split(split: str, jsonl_path: Path) -> list[RSATExample]:
    """Read a FeTaQA JSONL file and convert to RSATExample list."""
    print(f"  Reading {jsonl_path} ...")

    examples: list[RSATExample] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            table_array: list[list[str]] = item.get("table_array", [])
            if not table_array or len(table_array) < 2:
                continue  # need at least headers + 1 row

            headers = [str(h) for h in table_array[0]]
            rows = [[str(c) for c in row] for row in table_array[1:]]
            caption = item.get("table_page_title") or None

            table = Table(headers=headers, rows=rows, caption=caption)

            examples.append(
                RSATExample(
                    id=f"fetaqa_{split}_{i}",
                    question=item["question"],
                    table=table,
                    answer=item["answer"],
                    source_dataset="fetaqa",
                    reasoning_trace=None,
                    metadata={
                        "feta_id": item.get("feta_id", ""),
                        "table_page_title": item.get("table_page_title", ""),
                        "table_section_title": item.get("table_section_title", ""),
                        "highlighted_cell_ids": item.get("highlighted_cell_ids", []),
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
    parser = argparse.ArgumentParser(description="Download and prepare FeTaQA dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw/fetaqa")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        jsonl_path = download_split(split, raw_dir)
        examples = convert_split(split, jsonl_path)
        save_jsonl(examples, out_dir / f"fetaqa_{split}.jsonl")


if __name__ == "__main__":
    main()