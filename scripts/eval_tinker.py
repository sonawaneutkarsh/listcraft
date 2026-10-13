"""ListCraft eval: base vs fine-tuned generation on held-out pairs.

Greedy decoding (temperature 0), max 600 tokens. Saves raw outputs to
eval/results.jsonl, then computes automatic metrics -> eval/metrics.json.
"""
import sys, json, re, time

sys.path.insert(0, "/home/hatch/workspace/skills/tinker/bin")
from tinker_client import get_service_client
from tinker import types

BASE_MODEL = "Qwen/Qwen3.5-4B"
EVAL = "/home/hatch/workspace/listcraft/data/eval.jsonl"
RESULTS = "/home/hatch/workspace/listcraft/eval/results.jsonl"
METRICS = "/home/hatch/workspace/listcraft/eval/metrics.json"
PROMPT_TMPL = "### Seller notes:\n{inp}\n\n### Optimized listing:\n"
MAX_TOKENS = 600

STOPWORDS = set("""a an the and or of to in on for with is are was were be been
it its this that these those as at by from we you your he she they them his her
their our i me my so do does did will would can could should have has had not no
but if then than when where which who whom whose what how all any each every
very just about into over after before between through during up down out off
s s t d ll ve re m""".split())


def load_eval():
    rows = []
    with open(EVAL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gen(client, tok, prompt, max_tokens=MAX_TOKENS, timeout_s=240):
    fut = client.sample(
        types.ModelInput.from_ints(tok.encode(prompt)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    res = fut.result(timeout=timeout_s)
    toks = res.sequences[0].tokens
    return tok.decode(toks)


def parse_listing(text):
    """Split TITLE:/DESCRIPTION:/TAGS: sections (best effort)."""
    title = re.search(r"TITLE:\s*(.*?)(?=\n\s*DESCRIPTION:|\Z)", text, re.S | re.I)
    desc = re.search(r"DESCRIPTION:\s*(.*?)(?=\n\s*TAGS:|\Z)", text, re.S | re.I)
    tags = re.search(r"TAGS:\s*(.*)", text, re.S | re.I)
    t = title.group(1).strip() if title else ""
    d = desc.group(1).strip() if desc else ""
    g = tags.group(1).strip() if tags else ""
    tag_list = [x.strip().strip('"').strip("'") for x in re.split(r"[,\n]", g) if x.strip()]
    return {"title": t, "description": d, "tags": tag_list}


def content_words(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def keyword_coverage(reference_output, model_output):
    ref = parse_listing(reference_output)
    ref_words = content_words(ref["title"] + " " + " ".join(ref["tags"]))
    if not ref_words:
        return None
    out_words = content_words(model_output)
    return len(ref_words & out_words) / len(ref_words)


def title_seo(parsed):
    title = parsed["title"]
    n = len(title)
    sweet = 120 <= n <= 140
    phrases = [p.strip() for p in title.split("|") if p.strip()]
    return {
        "title_len": n,
        "title_sweet_spot": sweet,
        "title_pipe_phrases": len(phrases),
        "tag_count": len(parsed["tags"]),
    }


def main():
    rows = load_eval()
    print(f"{len(rows)} eval pairs", flush=True)
    sc = get_service_client()
    ft_path = open("/home/hatch/workspace/listcraft/sampler_path.txt").read().strip()
    print("finetuned sampler:", ft_path, flush=True)

    base_client = sc.create_sampling_client(base_model=BASE_MODEL)
    ft_client = sc.create_sampling_client(model_path=ft_path)
    tok = base_client.get_tokenizer()

    out = open(RESULTS, "a")
    done = set()
    try:
        with open(RESULTS) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["idx"])
    except FileNotFoundError:
        pass
    print(f"resuming: {len(done)} pairs already done", flush=True)
    agg = {"base": [], "ft": []}
    for i, r in enumerate(rows):
        if i in done:
            continue
        prompt = PROMPT_TMPL.format(inp=r["input"])
        t0 = time.time()
        try:
            base_text = gen(base_client, tok, prompt)
        except Exception as e:
            base_text = f"[ERROR: {type(e).__name__}: {e}]"
        try:
            ft_text = gen(ft_client, tok, prompt)
        except Exception as e:
            ft_text = f"[ERROR: {type(e).__name__}: {e}]"
        dt = time.time() - t0
        out.write(json.dumps({
            "idx": i, "input": r["input"], "reference": r["output"],
            "base": base_text, "finetuned": ft_text,
        }) + "\n")
        out.flush()
        for key, text in (("base", base_text), ("ft", ft_text)):
            if text.startswith("[ERROR"):
                agg[key].append({"idx": i, "coverage": None, "title_len": 0,
                                 "title_sweet_spot": False, "title_pipe_phrases": 0,
                                 "tag_count": 0, "error": True})
                continue
            parsed = parse_listing(text)
            cov = keyword_coverage(r["output"], text)
            seo = title_seo(parsed)
            agg[key].append({"idx": i, "coverage": cov, **seo})
        print(f"[{i+1}/{len(rows)}] done in {dt:.1f}s", flush=True)
    out.close()

    # Recompute metrics from the FULL results file (includes resumed rows).
    agg = {"base": [], "ft": []}
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = next((r["output"] for r in rows if r["input"] == row["input"]), None)
            for key, text in (("base", row["base"]), ("ft", row["finetuned"])):
                if text.startswith("[ERROR"):
                    agg[key].append({"idx": row["idx"], "coverage": None, "title_len": 0,
                                     "title_sweet_spot": False, "title_pipe_phrases": 0,
                                     "tag_count": 0, "error": True})
                    continue
                parsed = parse_listing(text)
                cov = keyword_coverage(ref, text) if ref else None
                seo = title_seo(parsed)
                agg[key].append({"idx": row["idx"], "coverage": cov, **seo})

    def summarize(lst):
        covs = [x["coverage"] for x in lst if x["coverage"] is not None]
        errs = sum(1 for x in lst if x.get("error"))
        return {
            "n": len(lst),
            "errors": errs,
            "keyword_coverage_mean": sum(covs) / len(covs) if covs else None,
            "title_sweet_spot_rate": sum(1 for x in lst if x["title_sweet_spot"]) / len(lst) if lst else 0,
            "title_len_mean": sum(x["title_len"] for x in lst) / len(lst) if lst else 0,
            "title_pipe_phrases_mean": sum(x["title_pipe_phrases"] for x in lst) / len(lst) if lst else 0,
            "tag_count_mean": sum(x["tag_count"] for x in lst) / len(lst) if lst else 0,
            "tags_eq_13_rate": sum(1 for x in lst if x["tag_count"] == 13) / len(lst) if lst else 0,
        }

    metrics = {
        "base": summarize(agg["base"]),
        "finetuned": summarize(agg["ft"]),
        "per_item": agg,
        "note": "reference rows: eval indices 0,7,21,33,50 for before/after examples",
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"base": metrics["base"], "finetuned": metrics["finetuned"]}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
