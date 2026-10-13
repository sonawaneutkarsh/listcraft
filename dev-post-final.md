# I Fine-Tuned a 4B Model to Write Etsy Listings That Actually Rank

*An open-weights side hustle: turning rough seller notes into optimized Etsy titles, descriptions, and tags — and proving the fine-tune beats the base model.*

---

Every Etsy seller knows the feeling. You have created something wonderful — a hand-poured candle, a vintage tee, a printable wall art — and now you have to describe it on Etsy. Title: 140 characters to encompass every search term a buyer can think of. Description: persuasive but easy to scan. Tags: 13 carefully chosen terms, each one a bet on how strangers search.

Most sellers either stuff keywords until the title reads like a ransom note, or write something poetic that no search algorithm will ever find. I kept thinking: this is a narrow, well-defined text task with tons of examples floating around. It's exactly the kind of thing a small fine-tuned model should be good at.

So for DEV's Hacktoberfest Week 2 challenge (theme: Side Hustle), I built **ListCraft** — an Etsy listing optimizer powered by a LoRA fine-tune of Qwen3.5-4B on Tinker, Thinking Machines' fine-tuning API. Paste in rough notes about your product; get back an optimized title, description, and tags. And critically: I measured whether the fine-tune actually helped, instead of just assuming it did.

## Why fine-tune instead of just prompting?

The honest answer: I wanted to find out if it matters. A base instruction model can already write a decent product description. But Etsy's format has sharp edges — titles need to land in a 120-140 character sweet spot, keyword phrases are pipe-separated, and you need exactly 13 tags. These are conventions a model won't reliably follow from a prompt alone, and stuffing all of it into a system prompt eats context on every single call.

Fine-tuning bakes the format in. The bet: a 4B model with the format burned into its weights beats a much larger prompted model at this specific job — and it's cheap enough to serve that a real side hustle could afford it.

## The data

I used the public `listybox-etsy-listing-parquet` dataset (536 real listing pairs), split 90/10 with seed 42 → **483 training pairs, 53 held out for eval**. Each pair maps rough seller notes (product concept, themes, audience, use cases) to a finished listing.

One important note: the source dataset declares no license, so I'm **not redistributing the data**. The repo ships the code, the eval protocol, and a rebuild script — anyone can download the source dataset themselves and reproduce everything. Open science shouldn't need a lawyer.

## Training on Tinker (including the run that collapsed)

Tinker is Thinking Machines' fine-tuning API — you bring a script, it brings the GPUs. I trained Qwen3.5-4B with LoRA rank 32, Adam at 1e-4, batch size 4, for 2 epochs. Total cost: a few dollars of a $10 credit.

Here's what most write-ups don't tell you: my first training run produced a collapsed model. The loss decreased to ~3e-5, which sounded great until I sampled from it and got infinite repetitions of a single token ("izingizingizing..."). The problem was in how I'd constructed my datum — Tinker's `cross_entropy` doesn't shift labels internally, and I'd fed it unshifted sequences, so the model learned the identity mapping: predict the current token. A perfect loss on a meaningless task.

I caught it by reading Tinker's cookbook source, fixed the shift (`model_input=toks[:-1]`, `target_tokens=toks[1:]`), and retrained from scratch. The healthy run went 1.17 → 0.36 mean NLL, smooth, no collapse. Lesson relearned: a loss curve that looks too good is usually lying to you. Sample early, sample often.

## Does the fine-tune actually help? (measured, not vibes)

On the 53 held-out listings, base model vs. fine-tuned, greedy decoding:

| Metric | Base Qwen3.5-4B | Fine-tuned |
|---|---|---|
| Keyword coverage | 0.77 | **0.93** |
| Titles in 120–140 char sweet spot | 0% | **83%** |
| Mean title length | 2,591 chars | **136 chars** |
| Keyword phrases per title | 1.2 | **4.0** |
| Outputs with a TAGS section | 0% | **100%** |

The base model doesn't just underperform — it fails at the task structurally. Zero-shot, it rambles for thousands of characters and never emits the tag section. The fine-tuned model learned the format cold.

Then the blind test: I had Qwen3-8B judge base vs. fine-tuned outputs without knowing which was which, on title SEO, description quality, and tag relevance. The fine-tuned model won 35 of 44 overall judgments (5 for base, 4 ties). It swept title SEO 42–2 and tags 41–2. Descriptions were essentially tied (20–24) — the fine-tune learned Etsy's format, not better prose, and I'm not going to pretend otherwise.

And the honest caveat: the fine-tuned model over-generates tags — about 66 per listing vs. 13 in the references. It learned the shape of a tag list but not when to stop. That's a real limitation, and it's the next thing I'd fix (a length penalty or stop-sequence tuning).

## The demo

There's a Gradio demo in the repo with a side-by-side view: type (or pick a preloaded example of) rough seller notes, and see the base model's rambling output next to the fine-tuned model's structured listing. That comparison *is* the pitch — it's the clearest possible demonstration of what the fine-tune bought.

```bash
cd demo
pip install -r ../requirements.txt
python app.py
```

## Why open innovation matters here

Everything in this project is open: the base model weights (Apache 2.0), the training and eval scripts, the metrics, the failure story. Tinker itself is built around open-weight models. A year ago, fine-tuning a model for a niche task like this would have meant either a big GPU bill or a closed API that could change under you. Now a student can do it on a $10 credit in an afternoon, measure the improvement rigorously, and hand the whole thing to someone else to reproduce.

That's the side hustle thesis, too. The moat for a tool like ListCraft was never the model — it's the workflow, the eval, the understanding of what Etsy sellers actually need. Open weights commoditize the intelligence; the product is everything around it. If I were turning this into a real business tomorrow, I'd start with the sellers, not the model card.

## What's next

- Fix tag over-generation (stop sequences / length penalties)
- Try a larger rank or full fine-tune to see if description quality moves
- Talk to actual Etsy sellers (the genuine first user this project deserves)
- Package it as a real hosted tool

Repo, code, eval protocol, and reproduction instructions are all public. The $10 of Tinker credit bought something I couldn't have gotten from prompting: a small model that reliably does one job well — and the receipts to prove it.

---

*Tags: `#devchallenge` `#hf26challenge`*
