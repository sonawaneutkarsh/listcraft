#!/usr/bin/env python3
"""Build ListCraft training/eval data from the raw Etsy listing parquet.

Source: burakaktna/listybox-etsy-listing-parquet (HuggingFace), 536 rows with
`instruction` (product details) and `output` (JSON: title, description, tags).

For each row we synthesize a training pair:
  input  = rough seller notes (the instruction's key: value facts, stripped of
           the "Create Etsy listing..." wrapper, kept terse like a seller's notes)
  output = the polished listing as plain text (TITLE / DESCRIPTION / TAGS)

Writes data/train.jsonl and data/eval.jsonl (90/10 split, seed 42), each line:
  {"input": "...", "output": "..."}

Reproduces from scratch: downloads the raw parquet if data/raw_etsy.parquet
is missing.
"""

import json
import os
import random
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_PATH = os.path.join(DATA_DIR, "raw_etsy.parquet")
RAW_URL = ("https://huggingface.co/datasets/burakaktna/listybox-etsy-listing-parquet"
           "/resolve/main/data/train-00000-of-00001.parquet")
SEED = 42
EVAL_FRAC = 0.10
WRAPPER = "Create Etsy listing with the following details:"


def ensure_raw():
    if os.path.exists(RAW_PATH):
        return
    print(f"downloading raw dataset -> {RAW_PATH}")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)


def rough_notes(instruction: str) -> str:
    """Turn the instruction's key: value facts into terse seller notes."""
    text = instruction.strip()
    if text.startswith(WRAPPER):
        text = text[len(WRAPPER):].strip()
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace("_", " ")
        value = value.strip()
        if value:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def polished_listing(output_json: str) -> str:
    """Format the polished listing as plain text."""
    obj = json.loads(output_json)
    title = obj.get("title", "").strip()
    desc = obj.get("description", "").strip()
    tags = obj.get("tags", [])
    tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    return f"TITLE: {title}\n\nDESCRIPTION: {desc}\n\nTAGS: {tag_str}"


def main():
    ensure_raw()
    import pandas as pd  # local import so the script documents its deps

    df = pd.read_parquet(RAW_PATH)
    pairs = []
    for _, row in df.iterrows():
        notes = rough_notes(str(row["instruction"]))
        listing = polished_listing(str(row["output"]))
        if notes and listing:
            pairs.append({"input": notes, "output": listing})

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * EVAL_FRAC))
    eval_pairs = pairs[:n_eval]
    train_pairs = pairs[n_eval:]

    for name, subset in (("train", train_pairs), ("eval", eval_pairs)):
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in subset:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(subset)} pairs")

    print(f"total: {len(pairs)} pairs from {len(df)} raw rows")


if __name__ == "__main__":
    main()
