"""Sustained load: one question after another on the NPU for a fixed time, reporting decisions per
second and graph latency per time window (does the rate drop over time?).

    python sustained.py --model english --binary ../models/english.bin --L 128 --seconds 300 \
        --out ../results/sustained_english_l128.json
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime"))
import numpy as np  # noqa: E402

from bench_lengths import CHECKPOINTS, QUESTIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="english", choices=list(CHECKPOINTS))
    ap.add_argument("--binary", required=True)
    ap.add_argument("--text", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_en.txt"))
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--out")
    args = ap.parse_args()

    from laya_npu import NPUAgent
    repo, sub = CHECKPOINTS[args.model]
    npu = NPUAgent(repo, args.binary, subfolder=sub)
    text = open(args.text).read()
    qs = [{k: v} for k, v in QUESTIONS.items()]
    npu.predict(text, qs[0], max_len=args.L)                              # warm-up

    windows, i = [], 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        w0, n, graph = time.time(), 0, []
        while time.time() - w0 < args.window and time.time() < t_end:
            npu.predict(text, qs[i % len(qs)], max_len=args.L)
            graph.append(npu.session.last_ms)
            n, i = n + 1, i + 1
        dt = time.time() - w0
        w = {"t_s": round(time.time() - (t_end - args.seconds)), "decisions_per_s": round(n / dt, 1),
             "graph_ms_p50": round(float(np.median(graph)), 1)}
        windows.append(w)
        print(json.dumps(w), flush=True)

    rates = [w["decisions_per_s"] for w in windows]
    summary = {"model": args.model, "L": args.L, "seconds": args.seconds, "decisions": i,
               "rate_first": rates[0], "rate_last": rates[-1], "rate_min": min(rates)}
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "windows": windows}, f, indent=1)


if __name__ == "__main__":
    main()
