"""Laya on the Hexagon NPU (QCS8550 / HTP V73).

NPUAgent is Laya's ONNXAgent (tokenizer, sequence building, calibration, decoding: all stock)
with the ONNX Runtime session swapped for a QNN context binary held resident on the HTP:

    NPU  : ModernBERT/mmBERT encoder + decision head + scorer, fp16, static L buckets
    CPU  : marker masking, softmax/entropy features, act head (a 1028x256x2 MLP)

The context binary holds one graph per sequence bucket (L=128/256/512...) with shared weights;
each question row runs on the smallest bucket that fits it.

    from laya_npu import NPURouter
    router = NPURouter(binaries={"english": "models/english.bin"})
    router.predict(text, questions)
"""
import os
import sys
import threading
import time
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qnnrt"))
import qnnrt  # noqa: E402

from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import Router, _split  # noqa: E402

MODELS_DIR = os.environ.get("LAYA_NPU_MODELS",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))


class _ActTail(nn.Module):
    """DecisionModel.forward after the scorer, verbatim maths."""

    def __init__(self, sd, d):
        super().__init__()
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, sd["act_head.2.weight"].shape[0]))
        self.act_head.load_state_dict({k[len("act_head."):]: v.float() for k, v in sd.items()
                                       if k.startswith("act_head.")})

    @torch.no_grad()
    def forward(self, logits, marker_mask, pooled):
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        return logits, self.act_head(torch.cat([pooled, feats], -1))


class NPUSession:
    """Quacks like the onnxruntime.InferenceSession that ONNXAgent calls."""

    def __init__(self, binary, tail):
        self.ctx = qnnrt.Context(binary)
        self.tail = tail
        self.buckets = {}
        for g in self.ctx.graphs:
            ins = {t.name: t for t in g.inputs}
            self.buckets[ins["input_ids"].shape[1]] = g
            self.M = ins["marker_pos"].shape[1]
            outs = sorted(g.outputs, key=lambda t: t.name)       # output_0 logits, output_1 pooled
            g.logits_name, g.pooled_name = outs[0].name, outs[1].name
        self.lengths = sorted(self.buckets)
        self.lock = threading.Lock()
        self.last_ms = 0.0

    def run(self, names, feeds):
        ids, att = feeds["input_ids"], feeds["attention_mask"]
        mpos, mmask, qtype = feeds["marker_pos"], feeds["marker_mask"], feeds["qtype"]
        n, k = mpos.shape
        if k > self.M:
            raise ValueError("NPU graph takes at most %d options per question, got %d" % (self.M, k))
        logits = np.empty((n, self.M), np.float32)
        pooled = None
        t0 = time.time()
        with self.lock:
            for r in range(n):
                length = int(att[r].sum())
                L = next((b for b in self.lengths if b >= length), None)
                if L is None:
                    raise ValueError("row of %d tokens exceeds the largest NPU bucket (%d)" % (length, self.lengths[-1]))
                g = self.buckets[L]
                row_ids = np.zeros((1, L), np.int32)
                row_att = np.zeros((1, L), np.int32)
                w = min(L, ids.shape[1])
                row_ids[0, :w] = ids[r, :w]
                row_att[0, :w] = att[r, :w]
                row_mp = np.zeros((1, self.M), np.int32)
                row_mp[0, :k] = mpos[r]
                out = g({"input_ids": row_ids, "attention_mask": row_att, "marker_pos": row_mp,
                         "qtype": np.asarray([qtype[r]], np.int32)})
                logits[r] = out[g.logits_name][0]
                pv = out[g.pooled_name][0]
                if pooled is None:
                    pooled = np.empty((n, pv.shape[0]), np.float32)
                pooled[r] = pv
        self.last_ms = (time.time() - t0) * 1000
        lg, act = self.tail(torch.from_numpy(logits[:, :k]), torch.from_numpy(mmask.astype(bool)),
                            torch.from_numpy(pooled))
        return [lg.numpy(), act.numpy()]


class NPUAgent(ONNXAgent):
    """ONNXAgent whose graph runs on the HTP. Same predict/predict_batch/decide API."""

    def __init__(self, model_id_or_path, binary, subfolder=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        for k in ("device", "compile", "fast", "backend", "dtype"):
            kwargs.pop(k, None)
        weights = hf_hub_download(model_id_or_path, "model.safetensors", subfolder=subfolder,
                                  revision=kwargs.get("revision"), token=kwargs.get("token"))
        sd = {k: v for k, v in load_file(weights).items() if k.startswith("act_head.")}
        d = sd["act_head.0.weight"].shape[1] - 4
        session = NPUSession(binary, _ActTail(sd, d))
        with mock.patch("onnxruntime.InferenceSession", lambda *a, **kw: session):
            super().__init__(model_id_or_path, onnx_path=binary, subfolder=subfolder, **kwargs)
        self.device = "npu"
        self.backend = "qnn-htp"


class NPURouter(Router):
    """Router that builds an NPUAgent for every checkpoint with a compiled binary, CPU otherwise."""

    def __init__(self, *args, binaries=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binaries = dict(binaries) if binaries is not None else {
            name: os.path.join(MODELS_DIR, name + ".bin") for name in self.models
            if os.path.exists(os.path.join(MODELS_DIR, name + ".bin"))}

    def _build(self, key):
        if key not in self.binaries:
            return super()._build(key)
        repo, sub = _split(self.models[key])
        return NPUAgent(repo, self.binaries[key], subfolder=sub, token=self.token)
