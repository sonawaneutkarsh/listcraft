"""ListCraft demo: side-by-side Etsy listing generation.

Fine-tuned ListCraft model vs the base model, same input.
Run:  ~/workspace/tinker-venv/bin/python ~/workspace/listcraft/demo/app.py
Then open http://127.0.0.1:7860 in a browser.
"""
import os
import re
import sys

# Sanitize bracketed IPv6 entries out of NO_PROXY/no_proxy before any
# library (gradio -> httpx) parses them at import time; the bracketed form
# crashes httpx's proxy-pattern parser in this runtime.
for _var in ("NO_PROXY", "no_proxy"):
    _val = os.environ.get(_var, "")
    if _val:
        _kept = [p for p in _val.split(",") if not re.fullmatch(r"\[.*\]", p.strip())]
        os.environ[_var] = ",".join(_kept)

import gradio as gr

sys.path.insert(0, "/home/hatch/workspace/skills/tinker/bin")
sys.path.insert(0, "/home/hatch/workspace/listcraft/scripts")
from tinker_client import get_service_client
from tinker import types
from eval_tinker import PROMPT_TMPL, parse_listing  # prompt format + section parser

BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_TOKENS = 600

_examples = None


def load_examples():
    global _examples
    if _examples is None:
        import json
        rows = [json.loads(l) for l in open(
            "/home/hatch/workspace/listcraft/data/eval.jsonl")]
        _examples = [rows[i]["input"] for i in (0, 7, 21)]
    return _examples


def make_clients():
    sc = get_service_client()
    ft_path = open("/home/hatch/workspace/listcraft/sampler_path.txt").read().strip()
    base_client = sc.create_sampling_client(base_model=BASE_MODEL)
    ft_client = sc.create_sampling_client(model_path=ft_path)
    tok = base_client.get_tokenizer()
    return base_client, ft_client, tok


BASE_CLIENT, FT_CLIENT, TOK = make_clients()


def generate(client, notes):
    prompt = PROMPT_TMPL.format(inp=notes.strip())
    fut = client.sample(
        types.ModelInput.from_ints(TOK.encode(prompt)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return TOK.decode(fut.result(timeout=240).sequences[0].tokens)


def format_listing(text):
    p = parse_listing(text)
    title = p["title"]
    # If the model didn't follow the TITLE/DESCRIPTION/TAGS format at all,
    # show the raw output so the contrast is visible and honest.
    if not title and not p["description"] and not p["tags"]:
        raw = text.strip()
        if len(raw) > 1500:
            raw = raw[:1500] + "\n…(truncated)"
        return ("*The model did not follow the listing format. Raw output:*\n\n"
                "```\n" + raw + "\n```")
    tags = p["tags"][:13]  # demo shows the Etsy-max 13; eval reports the raw count
    md = f"### TITLE ({len(title)} chars)\n{title}\n\n"
    md += f"### DESCRIPTION\n{p['description']}\n\n"
    md += f"### TAGS ({len(tags)} shown)\n" + "\n".join(f"- {t}" for t in tags)
    return md


def run_demo(notes):
    if not notes or not notes.strip():
        return "Enter some seller notes first.", "Enter some seller notes first."
    ft_text = generate(FT_CLIENT, notes)
    base_text = generate(BASE_CLIENT, notes)
    return format_listing(ft_text), format_listing(base_text)


with gr.Blocks(title="ListCraft — Etsy listing optimizer") as demo:
    gr.Markdown(
        "# ListCraft\n"
        "Rough seller notes in, optimized Etsy listing out. "
        "Left: **fine-tuned ListCraft model** (Qwen3.5-4B + LoRA, trained on 483 real listings). "
        "Right: **base model**, same prompt — the difference is the fine-tune."
    )
    notes = gr.Textbox(
        label="Seller notes (free text)",
        placeholder="e.g. Handmade ceramic mug, speckled glaze, 12oz, gift for coffee lovers...",
        lines=8,
    )
    with gr.Row():
        btn = gr.Button("Optimize listing", variant="primary")
        clear = gr.Button("Clear")
    with gr.Row():
        ft_out = gr.Markdown(label="Fine-tuned (ListCraft)")
        base_out = gr.Markdown(label="Base model")
    gr.Markdown("### Try an example")
    with gr.Row():
        ex = load_examples()
        for i, e in enumerate(ex):
            gr.Button(f"Example {i+1}").click(
                fn=lambda e=e: e, outputs=notes)

    btn.click(fn=run_demo, inputs=notes, outputs=[ft_out, base_out])
    clear.click(fn=lambda: ("", "", ""), outputs=[notes, ft_out, base_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
