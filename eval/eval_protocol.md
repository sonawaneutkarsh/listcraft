# ListCraft eval protocol

Goal: show that the fine-tuned model writes better Etsy listings than the
base model, honestly and reproducibly, on the 53 held-out pairs in
`data/eval.jsonl` (never seen during training).

## Setup

- **Base model**: the open-weight base model, prompted with the same system
  instruction as the fine-tuned model (e.g. "You are an Etsy listing writer.
  Given rough seller notes, write a TITLE, DESCRIPTION, and TAGS.").
- **Fine-tuned model**: base + LoRA adapter trained on `data/train.jsonl`.
- Both generate one listing per eval input, greedy decoding (temperature 0),
  max ~600 tokens. No cherry-picking: all 53 pairs are scored.

## Metrics

### 1. Keyword coverage (automatic)
For each eval pair, extract content words from the reference output's title +
tags (lowercased, stopwords removed). Score = fraction of those words present
in the model's output. Measures whether the model surfaces the product's key
selling terms instead of generic filler.

### 2. Title SEO heuristics (automatic)
- Title length in the 120–140 char Etsy sweet spot (binary per listing).
- Number of `|`-separated keyword phrases in the title (Etsy titles pack
  multiple search phrases; reference average ≈ 4–5).
- Tag count: exactly 13 tags is the Etsy maximum sellers aim for.

### 3. LLM-judge comparison (qualitative)
A strong instruction model (not either contestant) blind-scores pairs of
listings (base vs fine-tuned, order randomized) on: which title would rank
better in Etsy search, which description is more persuasive, and which tags
are more specific. Report win/tie/loss rates across the 53 pairs, with the
judge prompt and raw judgments saved under `eval/judgments/`.

## Reporting

Report all three metrics side by side for base vs fine-tuned, plus 3–5
representative before/after examples (fixed, pre-registered indices, e.g.
eval rows 0, 7, 21, 33, 50). If the fine-tuned model does not beat the base
on a metric, say so — the write-up reports what happened, not what we hoped.
