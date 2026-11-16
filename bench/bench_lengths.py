"""Laya on the HTP vs stock Laya on the CPU, per input length.

For every sequence bucket L in the context binary, the same text is cut by Laya itself to L tokens
(`max_len=L`), so each question is exactly one L-token row on both sides. Per question it records:

  npu_graph_ms : the NPU call for the row (qnnrt graphExecute and its buffers; resident context)
  npu_ms       : NPUAgent.predict() end to end (tokenization, host tail, decoding included)
  cpu_ms       : stock laya.Agent.predict() on the CPU (torch fp32)
  same answer / max probability difference between the two

    python bench_lengths.py --model english --binary ../models/english.bin \
        --out ../results/english.json
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime"))
import numpy as np  # noqa: E402

CHECKPOINTS = {"english": ("convaiinnovations/laya", None),
               "multilingual": ("convaiinnovations/laya", "multilingual"),
               "typed-decisions": ("convaiinnovations/laya", "typed-decisions")}

# One question of each Laya type, so all three decoding paths are timed and compared.
QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is this document mainly about?",
              "criteria": {"software": "software, systems, outages, deployments",
                           "finance": "money, invoices, refunds, pricing",
                           "health": "medicine, patients, treatment",
                           "sport": "games, teams, athletes"}},
    "escalate": {"type": "noul", "instructions": "Should this be escalated to a senior engineer?"},
    "urgency": {"type": "score", "instructions": "How urgent is this for the customer?",
                "criteria": ["not urgent", "low", "medium", "high", "critical"]},
}


def probs(a):
    return np.array([1 - a["noul"], a["noul"]]) if a["type"] == "noul" else np.array(list(a["probabilities"].values()))


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="english", choices=list(CHECKPOINTS))
    ap.add_argument("--binary", required=True, help="linked context binary for this checkpoint")
    ap.add_argument("--text", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_en.txt"))
    ap.add_argument("--lengths", type=int, nargs="*", help="default: every bucket in the binary")
    ap.add_argument("--npu-runs", type=int, default=20, help="timed NPU runs per question and length")
    ap.add_argument("--cpu-runs", type=int, default=3, help="timed CPU runs per question and length")
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args()

    from laya import Agent
    from laya_npu import NPUAgent
    repo, sub = CHECKPOINTS[args.model]
    text = open(args.text).read()

    t = time.time()
    npu = NPUAgent(repo, args.binary, subfolder=sub)
    npu_load = time.time() - t
    t = time.time()
    cpu = Agent(repo, subfolder=sub, device="cpu")
    cpu_load = time.time() - t
    lengths = args.lengths or npu.session.lengths
    print("%s: NPU load %.1f s (buckets %s), CPU load %.1f s" % (args.model, npu_load, npu.session.lengths, cpu_load))

    rows = []
    for L in lengths:
        for qid, q in QUESTIONS.items():
            one = {qid: q}
            rn = npu.predict(text, one, max_len=L)                       # warm-up
            tokens = rn["usage"]["input_tokens"]
            graph, e2e = [], []
            for _ in range(args.npu_runs):
                t = time.perf_counter()
                rn = npu.predict(text, one, max_len=L)
                e2e.append((time.perf_counter() - t) * 1000)
                graph.append(npu.session.last_ms)
            cpu.predict(text, one, max_len=L)                            # warm-up
            ce = []
            for _ in range(args.cpu_runs):
                t = time.perf_counter()
                rc = cpu.predict(text, one, max_len=L)
                ce.append((time.perf_counter() - t) * 1000)
            pc, pn = probs(rc["answers"][qid]), probs(rn["answers"][qid])
            row = {"L": L, "question": qid, "type": q["type"], "input_tokens": tokens,
                   "npu_graph_ms_p50": pct(graph, 50), "npu_graph_ms_p90": pct(graph, 90),
                   "npu_ms_p50": pct(e2e, 50), "npu_ms_p90": pct(e2e, 90),
                   "cpu_ms_p50": pct(ce, 50), "same_answer": bool(pc.argmax() == pn.argmax()),
                   "max_prob_diff": float(np.abs(pc - pn).max()),
                   "cpu_probs": np.round(pc, 4).tolist(), "npu_probs": np.round(pn, 4).tolist()}
            row["speedup"] = row["cpu_ms_p50"] / row["npu_ms_p50"]
            rows.append(row)
            print("L=%-5d %-9s graph %6.1f ms | npu %6.1f ms | cpu %7.0f ms | %5.1fx | %s, diff %.4f" % (
                L, qid, row["npu_graph_ms_p50"], row["npu_ms_p50"], row["cpu_ms_p50"], row["speedup"],
                "same" if row["same_answer"] else "DIFFERENT", row["max_prob_diff"]))

    if args.out:
        meta = {"model": args.model, "binary": os.path.basename(args.binary),
                "binary_mb": round(os.path.getsize(args.binary) / 2**20), "npu_load_s": round(npu_load, 1),
                "cpu_load_s": round(cpu_load, 1), "npu_runs": args.npu_runs, "cpu_runs": args.cpu_runs,
                "host": platform.node(), "date": time.strftime("%Y-%m-%d"),
                "qnn_dir": os.environ.get("QNN_DIR", "")}
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
