# Data license notes — ListCraft

Source dataset: **burakaktna/listybox-etsy-listing-parquet**
https://huggingface.co/datasets/burakaktna/listybox-etsy-listing-parquet
Checked: 2026-10-12 (via Hugging Face Hub API + dataset README).

## What the source page says

- The dataset card declares **no license** — the Hub API returns no
  `license`/`licenses` field, and the dataset README is auto-generated
  `dataset_info` YAML only (features, splits, sizes). No terms-of-use text.
- The dataset is public and ungated (not `private`, not `gated`).
- Content: 536 instruction/output pairs of AI-generated Etsy listing copy
  (titles, descriptions, tags) produced by ListyBox tooling, plus image URLs.

## Verdict

| Artifact | Publishable? |
|---|---|
| `data/raw_etsy.parquet` (raw download) | **No** — no license grants redistribution rights. |
| `data/train.jsonl`, `data/eval.jsonl` (derived pairs) | **No** — derivative of unlicensed data; do not redistribute the files. |
| Our code (`scripts/build_dataset.py`, training/eval/demo scripts) | **Yes** — original work. |
| Eval protocol, metrics, judge results, model outputs we generated | **Yes** — our own artifacts. |
| LoRA adapter weights | **Yes** (our training artifact; base model Qwen3.5-4B is Apache 2.0). |

## Conservative reproduction path (recommended)

Do not ship the data files. Instead, anyone can reproduce with:

1. Download the source dataset themselves:
   `hf_hub_download("burakaktna/listybox-etsy-listing-parquet")`
   (public, ungated — the downloader accepts the source's terms, such as they are)
2. Run `scripts/build_dataset.py` to rebuild `data/train.jsonl` / `data/eval.jsonl`
   (90/10 split, seed 42).
3. Train with `scripts/train_tinker.py`, evaluate with `scripts/eval_tinker.py`.

If a license is added to the source dataset later, re-check before publishing
any data files.
