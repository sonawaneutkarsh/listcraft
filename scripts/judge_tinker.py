"""ListCraft LLM-judge: blind base-vs-fine-tuned comparison.

Reads eval/results.jsonl (already-generated base + finetuned outputs),
asks an independent instruction model (Qwen3-8B — stronger than either
contestant, not a contestant itself) to blind-score each pair on title SEO,
description persuasiveness, tag relevance, and overall quality, with A/B
order randomized per pair.

Prompt design note: the judge completes the verdict directly after a
"TITLE_SEO:" prefix (max 120 tokens), each verdict starting with "A:",
"B:" or "Tie:" followed by a few words of reason. Responses that don't
match this shape are kept raw and counted as 'unparsed' — verdicts are
never inferred from prose.

Writes eval/judge_results.jsonl (raw judgments) and merges a 'judge'
summary into eval/metrics.json (existing keys untouched).
"""
import sys, json, re, time, random

sys.path.insert(0, "/home/hatch/workspace/skills/tinker/bin")
from tinker_client import get_service_client
from tinker import types

JUDGE_MODEL = "Qwen/Qwen3-8B"
RESULTS = "/home/hatch/workspace/listcraft/eval/results.jsonl"
JUDGE_OUT = "/home/hatch/workspace/listcraft/eval/judge_results.jsonl"
METRICS = "/home/hatch/workspace/listcraft/eval/metrics.json"
SEED = 20261012
MAX_TOKENS = 220

JUDGE_PROMPT = """You are an Etsy SEO judge. Compare Listing A vs Listing B (you don't know which system wrote which).
Reply with exactly 4 labeled lines. Begin EACH line's verdict with exactly one of: A:  B:  Tie:  (letter + colon), then a few words of reason.

Seller notes (abridged):
{notes}

--- LISTING A ---
{a}

--- LISTING B ---
{b}

Judge on: TITLE_SEO (keyword-rich, 120-140 chars, pipe-separated phrases), DESCRIPTION (persuasive, specific, benefit-driven), TAGS (specific multi-word buyer search terms), OVERALL.

Your 4 lines (each verdict starts with A: or B: or Tie:):
TITLE_SEO:"""

LEAD_RE = re.compile(r"^\s*(A|B|Tie):", re.I)
LABELED_RE = re.compile(r"^(DESCRIPTION|TAGS|OVERALL):\s*(A|B|Tie):", re.M | re.I)


def parse_judgment(text):
    """Strict parse: leading verdict is TITLE_SEO (prompt ends with its
    label), then labeled DESCRIPTION/TAGS/OVERALL lines. None if incomplete."""
    out = {}
    m = LEAD_RE.match(text)
    if not m:
        return None
    out["TITLE_SEO"] = m.group(1).lower()
    for key, val in LABELED_RE.findall(text):
        ku = key.upper()
        if ku not in out:
            out[ku] = val.lower()
    return out if set(out) == {"TITLE_SEO", "DESCRIPTION", "TAGS", "OVERALL"} else None


def judge_call(client, tok, prompt, timeout_s=240):
    fut = client.sample(
        types.ModelInput.from_ints(tok.encode(prompt)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return tok.decode(fut.result(timeout=timeout_s).sequences[0].tokens)


def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    print(f"{len(rows)} pairs to judge", flush=True)

    done = set()
    try:
        with open(JUDGE_OUT) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["idx"])
    except FileNotFoundError:
        pass
    print(f"resuming: {len(done)} already judged", flush=True)

    sc = get_service_client()
    jc = sc.create_sampling_client(base_model=JUDGE_MODEL)
    tok = jc.get_tokenizer()
    rng = random.Random(SEED)

    outf = open(JUDGE_OUT, "a")
    for row in rows:
        idx = row["idx"]
        if idx in done:
            continue
        # Randomize A/B assignment; record it so verdicts map back honestly.
        if rng.random() < 0.5:
            a_text, b_text = row["base"], row["finetuned"]
            a_is, b_is = "base", "finetuned"
        else:
            a_text, b_text = row["finetuned"], row["base"]
            a_is, b_is = "finetuned", "base"
        notes = row["input"][:1200]
        prompt = JUDGE_PROMPT.format(notes=notes, a=a_text, b=b_text)
        t0 = time.time()
        try:
            raw = judge_call(jc, tok, prompt)
            parsed = parse_judgment(raw)
        except Exception as e:
            raw = f"[ERROR: {type(e).__name__}: {e}]"
            parsed = None
        # Map A/B verdicts back to base/finetuned.
        mapped = None
        if parsed:
            def winner(v):
                return "tie" if v == "tie" else (a_is if v == "a" else b_is)
            mapped = {k.lower(): winner(v) for k, v in parsed.items()}
        outf.write(json.dumps({
            "idx": idx, "judge_model": JUDGE_MODEL, "seed": SEED,
            "a_is": a_is, "b_is": b_is,
            "raw": raw, "parsed": parsed, "verdict": mapped,
            "elapsed_s": round(time.time() - t0, 1),
        }) + "\n")
        outf.flush()
        print(f"[{idx}] {'parsed' if mapped else 'UNPARSED'} in {time.time()-t0:.1f}s", flush=True)
    outf.close()

    # Summarize.
    judgments = [json.loads(l) for l in open(JUDGE_OUT) if l.strip()]
    verdicts = [j["verdict"] for j in judgments if j["verdict"]]
    summary = {"n_total": len(judgments), "n_parsed": len(verdicts),
               "n_unparsed": len(judgments) - len(verdicts),
               "judge_model": JUDGE_MODEL, "seed": SEED,
               "method": ("blind A/B, order randomized per pair; verdicts parsed "
                          "strictly from labeled lines, unparsable kept raw")}
    for crit in ("title_seo", "description", "tags", "overall"):
        counts = {"finetuned": 0, "base": 0, "tie": 0}
        for v in verdicts:
            counts[v[crit]] += 1
        n = len(verdicts)
        summary[crit] = {**counts,
                         "finetuned_win_rate": counts["finetuned"] / n if n else None}
    # Merge into metrics.json without clobbering existing keys.
    with open(METRICS) as f:
        metrics = json.load(f)
    metrics["judge"] = summary
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
