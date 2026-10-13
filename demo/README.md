# ListCraft demo

Side-by-side Etsy listing generation: the fine-tuned ListCraft model vs the
base model on the same seller notes.

## Run

```bash
~/workspace/tinker-venv/bin/python ~/workspace/listcraft/demo/app.py
```

Then open **http://127.0.0.1:7860** in a browser.

Type (or paste) rough seller notes, click **Optimize listing**, and compare:
- **Left** — fine-tuned model (Qwen3.5-4B + rank-32 LoRA trained on 483
  real Etsy listings via Tinker). Titles land in the 120–140 char Etsy
  sweet spot with pipe-separated keyword phrases.
- **Right** — base Qwen3.5-4B with the identical prompt, no fine-tuning.

Three preloaded examples come from the held-out eval set (rows 0, 7, 21).

## Cost note

Each click makes two Tinker sampling calls (one per model). A handful of
demo generations costs cents; don't hammer it.

## Files

- `app.py` — the Gradio app (imports the prompt template and listing parser
  from `../scripts/eval_tinker.py` so the demo uses exactly the eval format).
