"""ListCraft LoRA fine-tune on Tinker.

SFT on Qwen/Qwen3.5-4B (rank-32 LoRA): prompt = rough seller notes,
target = optimized listing. Loss on target tokens only (prompt masked).
"""
import sys, json, random, time

sys.path.insert(0, "/home/hatch/workspace/skills/tinker/bin")
from tinker_client import get_service_client
from tinker import types

BASE_MODEL = "Qwen/Qwen3.5-4B"
RANK = 32
EPOCHS = 2
BATCH = 4
LR = 1e-4
LOG_EVERY = 10
DATA = "/home/hatch/workspace/listcraft/data/train.jsonl"
LOG = "/home/hatch/workspace/listcraft/train_log.jsonl"
CKPT_TXT = "/home/hatch/workspace/listcraft/checkpoint.txt"

PROMPT_TMPL = "### Seller notes:\n{inp}\n\n### Optimized listing:\n"


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                pairs.append((r["input"], r["output"]))
    return pairs


def make_datum(tok, prompt, target):
    p = tok.encode(prompt)
    t = tok.encode(target)
    allt = p + t
    # Next-token shift (per tinker-cookbook sl_loop): model_input drops the
    # last token, targets drop the first token, weights shift with targets.
    # Passing unshifted sequences trains the identity mapping (copy input).
    w = [0.0] * len(p) + [1.0] * len(t)
    tw = w[1:]
    tot = sum(tw)
    tw = [x / tot for x in tw]  # mean reduction: stable grads across lengths
    return types.Datum(
        model_input=types.ModelInput.from_ints(allt[:-1]),
        loss_fn_inputs={
            "target_tokens": allt[1:],
            "weights": tw,
        },
    ), len(t)


def main():
    pairs = load_pairs(DATA)
    print(f"loaded {len(pairs)} train pairs", flush=True)

    sc = get_service_client()
    print("console:", sc.get_console_url(), flush=True)
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print("training client ready", flush=True)
    tok = tc.get_tokenizer()

    step = 0
    logf = open(LOG, "w")
    t0 = time.time()
    for epoch in range(EPOCHS):
        random.seed(1234 + epoch)
        order = list(range(len(pairs)))
        random.shuffle(order)
        for i in range(0, len(order), BATCH):
            batch_idx = order[i:i + BATCH]
            datums, ntoks = [], 0
            for j in batch_idx:
                inp, out = pairs[j]
                d, n = make_datum(tok, PROMPT_TMPL.format(inp=inp), out)
                datums.append(d)
                ntoks += n
            fb = tc.forward_backward(datums, "cross_entropy").result(timeout=900)
            loss_sum = fb.metrics.get("loss:sum", float("nan"))
            tc.optim_step(types.AdamParams(learning_rate=LR)).result(timeout=900)
            step += 1
            if step % LOG_EVERY == 0:
                entry = {
                    "step": step, "epoch": epoch + 1,
                    "loss_sum": loss_sum, "batch_target_tokens": ntoks,
                    "loss_per_token": loss_sum / ntoks if ntoks else None,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                logf.write(json.dumps(entry) + "\n")
                logf.flush()
                print(json.dumps(entry), flush=True)
    logf.close()

    print("saving state...", flush=True)
    sr = tc.save_state(name="listcraft-fixed").result(timeout=900)
    ckpt_path = getattr(sr, "path", None) or str(sr)
    print("save_state path:", ckpt_path, flush=True)
    with open(CKPT_TXT, "w") as f:
        f.write(str(ckpt_path) + "\n")

    print("saving sampler weights...", flush=True)
    sw = tc.save_weights_for_sampler(name="listcraft-fixed-sampler").result(timeout=900)
    sampler_path = sw.path
    print("sampler path:", sampler_path, flush=True)
    with open("/home/hatch/workspace/listcraft/sampler_path.txt", "w") as f:
        f.write(sampler_path + "\n")

    # spend check
    try:
        tel = tc.get_telemetry()
        print("telemetry:", tel, flush=True)
    except Exception as e:
        print("telemetry unavailable:", e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
