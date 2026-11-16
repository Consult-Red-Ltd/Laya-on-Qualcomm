"""Export Laya's DecisionModel as a static-shape, HTP-friendly graph.

On the NPU: ModernBERT encoder + qtype embedding + 2-layer decision head + per-marker scorer.
On the CPU (laya_npu.py): marker masking, softmax/entropy features and the tiny act head.

HTP-specific changes vs the stock forward (same maths):
  * int32 inputs (HTP has no int64), static [1, L] / [1, M] shapes;
  * attention masks are additive -1e4 (not bool / -inf): on fp16 HTP a fully masked row
    (padding query outside every sliding window) would otherwise softmax to NaN and poison
    the real tokens through 0 * NaN in the next layer;
  * eager attention (matmul-softmax-matmul) in the encoder and in the decision head.

Usage: python laya_npu_export.py --model english --L 128 256 512 --out onnx/
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4
M = 32   # option slots per question (static output shape)
SUBFOLDERS = {"english": ("convaiinnovations/laya", None),
              "multilingual": ("convaiinnovations/laya", "multilingual"),
              "typed-decisions": ("convaiinnovations/laya", "typed-decisions")}


def load_agent(name):
    from laya import Agent
    repo, sub = SUBFOLDERS[name]
    return Agent(repo, subfolder=sub, device="cpu") if sub else Agent(repo, device="cpu")


def _full_attn(q, k, v, key_add, scaling):
    """Plain attention. q/k/v [B,H,L,D], key_add [B,1,1,L] -> [B,H,L,D]."""
    att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scaling + key_add, dim=-1)
    return torch.matmul(att, v)


def _head_layer(layer, x, key_add):
    """nn.TransformerEncoderLayer (norm_first, relu) with eager additive-mask attention."""
    sa = layer.self_attn
    B, L, D = x.shape
    H = sa.num_heads
    hd = D // H
    y = layer.norm1(x)
    qkv = F.linear(y, sa.in_proj_weight, sa.in_proj_bias)
    q, k, v = (t.reshape(B, L, H, hd).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    o = _full_attn(q, k, v, key_add, hd ** -0.5).transpose(1, 2).reshape(B, L, D)
    x = x + sa.out_proj(o)
    y = layer.norm2(x)
    x = x + layer.linear2(layer.activation(layer.linear1(y)))
    return x


class NpuCore(nn.Module):
    def __init__(self, dm, L):
        super().__init__()
        from transformers.masking_utils import create_bidirectional_sliding_window_mask
        self.enc = dm.encoder
        self.head = dm.head
        self.type_emb = dm.type_emb
        self.scorer = dm.scorer
        assert all(getattr(l, "activation_relu_or_gelu", 1) == 1 for l in self.head.layers)
        self.enc.config._attn_implementation = "eager"
        # The exact band HF uses for sliding layers, taken from HF itself with no padding.
        emb = torch.zeros(1, L, self.enc.config.hidden_size)
        m = create_bidirectional_sliding_window_mask(config=self.enc.config, inputs_embeds=emb,
                                                     attention_mask=torch.ones(1, L, dtype=torch.long))
        if m is None:  # window covers everything
            band = torch.ones(L, L, dtype=torch.bool)
        elif m.dtype == torch.bool:
            band = m[0, 0]
        else:
            band = m[0, 0] == 0
        self.register_buffer("band_add", torch.where(band, 0.0, NEG)[None, None], persistent=False)

    def forward(self, input_ids, attention_mask, marker_pos, qtype):
        key_add = torch.where(attention_mask[:, None, None, :] > 0, 0.0, NEG)       # [B,1,1,L]
        masks = {"full_attention": key_add,
                 "sliding_attention": torch.clamp(key_add + self.band_add, min=NEG)}   # [B,1,L,L]
        h = self.enc(input_ids=input_ids.long(), attention_mask=masks).last_hidden_state
        h = h + self.type_emb(qtype.long())[:, None, :]
        for layer in self.head.layers:
            h = _head_layer(layer, h, key_add)
        idx = marker_pos.long()[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1)                  # [B,M]
        return logits, h[:, 0]                                                       # pooled [B,D]


def check(agent, core, L):
    """Stock DecisionModel (SDPA, HF masks) vs NpuCore on a real input, padded to L."""
    from laya.onnx_agent import ONNXAgent
    from laya.common import collate_items
    state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    qs = {"department": {"type": "choice", "instructions": "Which department should handle this?",
                         "criteria": {"billing": "invoices, payments, refunds",
                                      "technical": "bugs, outages, system errors", "other": "everything else"}},
          "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}
    internal = {k: ONNXAgent._to_internal(v) for k, v in qs.items()}
    items = ONNXAgent._encode_state(agent, state, list(qs), internal, L, 192)
    b = collate_items([items], agent.tok.pad_token_id)
    cfg = agent.model.encoder.config
    impl, cfg._attn_implementation = cfg._attn_implementation, "sdpa"   # the encoder is shared with core
    with torch.no_grad():
        ref_logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    cfg._attn_implementation = impl
    with torch.no_grad():
        n, l = b["input_ids"].shape
        ids = torch.full((n, L), agent.tok.pad_token_id, dtype=torch.int32)
        att = torch.zeros((n, L), dtype=torch.int32)
        ids[:, :l] = b["input_ids"]
        att[:, :l] = b["attention_mask"]
        mp = torch.zeros((n, M), dtype=torch.int32)
        mp[:, :b["marker_pos"].shape[1]] = b["marker_pos"]
        qt = b["qtype"].int()
        lg = torch.cat([core(ids[r:r + 1], att[r:r + 1], mp[r:r + 1], qt[r:r + 1])[0] for r in range(n)])  # B=1, as on the NPU
    k = b["marker_pos"].shape[1]
    got = lg[:, :k].masked_fill(~b["marker_mask"], -1e4)
    print("L=%d (%d real tokens) max |logit diff| = %.5f" % (L, l, (got - ref_logits).abs().max().item()))
    print(" ref", ref_logits.softmax(-1).numpy().round(4).tolist())
    print(" npu", got.softmax(-1).numpy().round(4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="english", choices=list(SUBFOLDERS))
    ap.add_argument("--L", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out", default="onnx")
    args = ap.parse_args()

    agent = load_agent(args.model)
    agent.model.eval().float()
    for L in args.L:
        core = NpuCore(agent.model, L).eval()
        check(agent, core, L)
        dummy = (torch.full((1, L), 5, dtype=torch.int32), torch.ones(1, L, dtype=torch.int32),
                 torch.arange(M, dtype=torch.int32)[None], torch.zeros(1, dtype=torch.int32))
        d = os.path.join(args.out, "%s_b1_l%d_m%d" % (args.model, L, M))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(core, dummy, path, opset_version=17, dynamo=False, do_constant_folding=True,
                              input_names=["input_ids", "attention_mask", "marker_pos", "qtype"],
                              output_names=["logits", "pooled"])
        print("wrote", path)


if __name__ == "__main__":
    main()
