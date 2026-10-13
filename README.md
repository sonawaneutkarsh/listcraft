# ListCraft

**Fine-tune an open-weight model to turn rough seller notes into SEO-optimized Etsy listings.**

Built for the [DEV Hacktoberfest 2026 Week 2 challenge](https://dev.to/challenges/hacktoberfest-week2-2026-10-12) — theme: **"Side Hustle"** (Oct 12–18, 2026). The idea: a tool a real Etsy seller would pay for. Sellers know their product but struggle to write titles, descriptions, and tags that rank in Etsy search. ListCraft takes bare-bones notes and produces a polished, keyword-rich listing.

## Approach

1. **Dataset** — 536 listing pairs derived from the public
   `burakaktna/listybox-etsy-listing-parquet` dataset (HuggingFace),
   split 90/10 (seed 42) → 483 train / 53 eval.
   Reproducible via `scripts/build_dataset.py`.
   ⚠️ The source dataset declares **no license**, so data files are
   **not** shipped in this repo — see `data/LICENSE_NOTES.md`.
2. **Fine-tuning** — LoRA (rank 32) fine-tune of **Qwen3.5-4B** on
   [Tinker](https://thinkingmachines.ai/tinker) (Thinking Machines):
   2 epochs, Adam 1e-4, a few dollars of a $10 credit.
   Targets the "Best Use of Tinker" category: *fine-tune a model for a
   specific task and show clear improvement over a baseline.*
3. **Eval** — base vs. fine-tuned on 53 held-out pairs
   (`eval/eval_protocol.md`): keyword coverage, title SEO heuristics,
   plus a blind LLM-judge comparison.
4. **Demo** — Gradio side-by-side: same seller notes through the base
   model and the fine-tuned model (`demo/`).

## Results

| Metric (53 held-out) | Base Qwen3.5-4B | Fine-tuned |
|---|---|---|
| Keyword coverage | 0.77 | **0.93** |
| Titles in 120–140 char sweet spot | 0% | **83%** |
| Mean title length | 2,591 chars | **136 chars** |
| Keyword phrases / title | 1.2 | **4.0** |
| Blind-judge overall wins (44 parsed) | 5 | **35** (4 ties) |

Honest caveats: the fine-tune learned Etsy's *format*, not better prose
(judge split descriptions 20–24), and it over-generates tags (~66/listing
vs. 13 reference).

## Layout

```
listcraft/
├── data/
│   ├── LICENSE_NOTES.md   # why data files aren't shipped
│   └── (rebuild with scripts/build_dataset.py)
├── scripts/
│   ├── build_dataset.py   # rebuilds data/ from the source dataset
│   ├── train_tinker.py    # LoRA training loop on Tinker
│   ├── eval_tinker.py     # base-vs-finetuned eval + metrics
│   └── judge_tinker.py    # blind LLM-judge comparison
├── eval/
│   ├── eval_protocol.md
│   ├── metrics.json       # aggregate results
│   └── judge_results.jsonl
├── demo/
│   ├── app.py             # Gradio side-by-side demo
│   └── README.md
├── checkpoint.txt         # Tinker URI of the trained LoRA weights
└── requirements.txt
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py   # downloads source dataset, builds train/eval splits
# set TINKER_API_KEY, then:
.venv/bin/python scripts/train_tinker.py
.venv/bin/python scripts/eval_tinker.py
```

## Status

- [x] Dataset built (536 pairs)
- [x] Fine-tuned on Tinker (Qwen3.5-4B, LoRA r32, 2 epochs)
- [x] Eval: base vs. fine-tuned (metrics + blind judge)
- [x] Demo UI (verified end-to-end)
- [ ] DEV submission post (due Oct 18)
