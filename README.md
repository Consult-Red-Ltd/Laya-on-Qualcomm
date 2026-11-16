# Laya on the Qualcomm Hexagon NPU (QCS8550)

Run the [Laya](https://github.com/NandhaKishorM/laya) decision model on the Hexagon NPU (HTP V73) of
the Qualcomm QCS8550, with answers that match stock Laya on the CPU. This repository contains the
scripts and a step-by-step recipe; it does not contain any models, SDK files or compiled binaries.

Laya reads a text and answers typed questions (`choice`, `score`, `noul` = yes/no) with
probabilities, in one forward pass of a ModernBERT / mmBERT encoder, without generating text. Upstream
Laya runs on CUDA, Apple MPS and the CPU. This recipe adds the NPU:

- the encoder, question-type embedding, decision head and option scorer run on the HTP in **fp16**;
- tokenization, the small "act" head, calibration and answer decoding stay on the CPU and are
  **unchanged upstream Laya code** (`ONNXAgent` with its ONNX Runtime session swapped for the NPU);
- there is **no quantization and no calibration data**: the compiler converts the model to fp16.

Results on the board are right below; the steps to reproduce them are in [Reproduction](#reproduction).

## Benchmarks (RB5 Gen2, QCS8550)

Measured on a Thundercomm RB5 Gen2 with QAIRT runtime 2.51, after a clean run of the
[Reproduction](#reproduction) steps (binaries compiled locally, Laya 0.4.7). Only one NPU workload
was running. CPU numbers are stock `laya.Agent` with torch 2.14.1 fp32 on the board's Kryo CPU.
Raw data: [results/](results/).

### Latency per input length

[`bench/bench_lengths.py`](bench/bench_lengths.py): one question per call, the same text cut to
exactly L tokens on both sides, median of 20 NPU runs and 3 CPU runs, averaged over three questions
(one `choice`, one `noul`, one `score`). *NPU graph* is the NPU call alone (`graphExecute` and its
buffers); *end to end* is the whole `predict()` call, including tokenization and decoding on the
CPU. The speed-up compares end to end with the CPU.

| Checkpoint | L (tokens) | NPU graph | NPU end to end | CPU | Speed-up | Same answer |
|---|---|---|---|---|---|---|
| english | 128 | 25 ms | 49 ms | 1.29 s | **27×** | 3/3 |
| english | 256 | 49 ms | 66 ms | 2.57 s | **39×** | 3/3 |
| english | 512 | 127 ms | 148 ms | 4.62 s | **31×** | 3/3 |
| multilingual | 128 | 12 ms | 21 ms | 0.54 s | **25×** | 3/3 |
| multilingual | 256 | 25 ms | 45 ms | 0.89 s | **20×** | 3/3 |
| multilingual | 512 | 62 ms | 84 ms | 1.66 s | **20×** | 3/3 |
| multilingual | 1024 | 221 ms | 245 ms | 3.59 s | **15×** | 3/3 |
| typed-decisions | 128 | 25 ms | 49 ms | 1.25 s | **26×** | 3/3 |
| typed-decisions | 256 | 49 ms | 68 ms | 2.39 s | **35×** | 3/3 |
| typed-decisions | 512 | 126 ms | 146 ms | 4.30 s | **29×** | 3/3 |
| typed-decisions | 1024 | 416 ms | 435 ms | 8.00 s | **18×** | 3/3 |

- Doubling the input from 512 to 1024 tokens makes the NPU 3.3–3.6× slower, because the
  global attention layers grow with the square of the length.
- At short inputs about half of the end-to-end time is host work (tokenization, Python, the small
  head on the CPU), not the NPU.

### Answer agreement with the CPU

[`bench/parity.py`](bench/parity.py): 12 short texts (support tickets, spam, a prompt injection, one
in Polish to exercise the multilingual checkpoint) × 4 questions, every answer compared with stock
Laya on the CPU.

| Checkpoint | Decisions | Same answer | Probability difference, mean / max | Speed-up (whole run) |
|---|---|---|---|---|
| english | 48 | **48/48** | 0.0007 / 0.0048 | 22× |
| multilingual | 48 | **48/48** | 0.0005 / 0.0036 | 16× |
| typed-decisions | 48 | **48/48** | 0.0006 / 0.0021 | 22× |

The NPU computes in fp16 and the CPU in fp32, so the probabilities differ in the third or fourth
decimal place; no decision changed.

### Sustained load: 5 minutes without a pause

[`bench/sustained.py`](bench/sustained.py): `english`, 128 tokens, one question after another for
300 s, NPU only.

| Time | 30 s | 60 s | 120 s | 180 s | 240 s | 300 s |
|---|---|---|---|---|---|---|
| Decisions per second | 20.1 | 20.3 | 20.0 | 19.9 | 20.1 | 19.9 |
| NPU graph, median | 25.1 ms | 25.1 ms | 25.1 ms | 25.1 ms | 25.1 ms | 25.0 ms |

6,017 decisions in 5 minutes with no slowdown: the rate and the graph latency stay flat. The same
decisions would take about 2 hours on the CPU (1.29 s each). The rate is set by the host loop: the
graph alone (25.1 ms) would allow about 40 per second.

## Limits

- Batch 1: questions run one row at a time on the HTP.
- At most 32 options per question.
- Inputs up to 512 (`english`) or 1024 tokens (`multilingual`, `typed-decisions`).
- Accuracy is Laya's own: the NPU reproduces the CPU, it does not improve on it.

## Reproduction

Everything needed to repeat the build and the measurements above.

### Tested setup

| | Version |
|---|---|
| Board | Thundercomm RB5 Gen2 (Qualcomm QCS8550, Hexagon HTP V73), Ubuntu 22.04, Python 3.10 |
| Host for the build | x86_64 Linux: Ubuntu 24.04 under WSL2 on Windows 11 (Core i7-10750H, 15 GB RAM for WSL), Python 3.12 |
| Qualcomm AI Runtime (QAIRT) SDK | 2.51.200.261021, on the host and on the board |
| Laya | 0.4.7 (`convaiinnovations/laya`: `english`, `multilingual`, `typed-decisions`) |
| Python packages | torch 2.14.1 (CPU), transformers 5.19.0; converter: onnx 1.19.1, onnxruntime 1.23.2, onnxsim 0.6.2 |

Other QCS8550 boards should work as they are, and other Hexagon V73+ SoCs with their own
`soc_model` and `dsp_arch` ([step 3](#3-compile-onnx--fp16-dlc--one-context-binary)); neither was
tested.

### Repository layout

```
export/laya_npu_export.py     Laya -> static-shape ONNX, one file per sequence length, with a numerical check
compile/compile_local.sh      ONNX -> fp16 DLC -> one weight-shared context binary (QAIRT, offline)
runtime/qnnrt/                libqnnrt.so: resident QNN runner with a C API, plus its ctypes binding
runtime/laya_npu.py           NPUAgent / NPURouter: Laya's own agent running on the NPU
runtime/requirements-board.txt
bench/bench_lengths.py        NPU vs CPU per input length (latency, answer agreement)
bench/parity.py               NPU vs CPU answers on 12 short texts x 4 questions
bench/sustained.py            NPU only, one question after another for a fixed time
bench/text_en.txt             benchmark text (a fictional support ticket, written for this repo)
results/                      raw benchmark output from the board (JSON)
```

### How it works

```
 x86_64 Linux host                                              QCS8550 board
 ─────────────────                                              ─────────────
 Laya checkpoint (Hugging Face)
   │ export/laya_npu_export.py: static shapes, int32 inputs,
   │ additive -1e4 masks, eager attention, check vs stock
   ▼
 model.onnx per sequence length L (128, 256, 512[, 1024])
   │ qairt-converter --float_bitwidth 16
   ▼
 one fp16 .dlc per L
   │ qnn-context-binary-generator (soc_model 66, weight sharing)
   ▼
 <model>.bin: one graph per L, weights stored once  ───scp──►  models/<model>.bin
                                                                  │ libqnnrt.so keeps it resident on the HTP
                                                                  ▼
                                                                NPUAgent.predict(text, questions)
```

Each question becomes one row. The row runs on the smallest graph whose length fits it, padded to
that length; inputs longer than the largest graph are rejected rather than silently truncated.

### Prerequisites

- **QAIRT SDK 2.51** (free, requires a Qualcomm account):
  <https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.51.200.261021/v2.51.200.261021.zip>
  (~3 GB zip, ~7.5 GB unpacked). The same zip provides the host tools and the board runtime.
- **An x86_64 Linux host**: Ubuntu 22.04 / 24.04, native or WSL2, ~16 GB RAM, ~45 GB free disk
  (SDK 7.5 GB, ONNX files 16 GB, DLCs 8.5 GB and binaries 2.4 GB for all three checkpoints).
  It has to be Linux: weight sharing between graphs is only supported for offline preparation on
  x86_64 Linux, not on Windows or on the board.
- **The board**: Ubuntu 22.04 with Python 3.10 (`python3-venv`), `g++`, SSH access, ~10 GB free
  disk (runtime, Python environment, Laya weights and the three binaries).
- Internet access on the host for Hugging Face (Laya weights) and PyPI.

The commands below use `$SDK` for the unpacked SDK root (the directory that contains `bin/`, `lib/`,
`include/` and `examples/`) and `rb5` as the board's SSH host name.

### 1. Host: SDK and Python environments

```bash
sudo apt-get install -y unzip python3-venv
unzip v2.51.200.261021.zip -d ~/qairt
export SDK=~/qairt/qairt/2.51.200.261021

# system libraries for the converter (libc++ and friends)
sudo bash $SDK/bin/check-linux-dependency.sh

# converter environment
python3 -m venv ~/qairt-venv && source ~/qairt-venv/bin/activate
pip install --upgrade pip
source $SDK/bin/envsetup.sh
python3 $SDK/bin/check-python-dependency
pip install onnx==1.19.1 onnxruntime==1.23.2 onnxsim==0.6.2
deactivate

# export environment (separate: the converter pins its own onnx version)
python3 -m venv ~/laya-export-venv
~/laya-export-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
~/laya-export-venv/bin/pip install transformers==5.19.0 laya==0.4.7 onnx onnxscript numpy
```

### 2. Export Laya to ONNX

```bash
git clone <this repository> laya-qualcomm-npu && cd laya-qualcomm-npu
~/laya-export-venv/bin/python export/laya_npu_export.py --model english --L 128 256 512 --out onnx/
~/laya-export-venv/bin/python export/laya_npu_export.py --model multilingual --L 128 256 512 1024 --out onnx/
~/laya-export-venv/bin/python export/laya_npu_export.py --model typed-decisions --L 128 256 512 1024 --out onnx/
```

Every bucket is checked against the stock Laya forward pass before it is written; expect
`max |logit diff| = 0.00000`. Each `model.onnx` is ~1.6 GB (`english`, `typed-decisions`) or
~1.2 GB (`multilingual`). The longest bucket matches each checkpoint's `max_len` (512 for
`english`, 1024 for the other two).

What the export changes, without changing the maths:

| Change | Why |
|---|---|
| static `[1, L]` inputs, one graph per L | the HTP needs static shapes |
| 32 fixed option slots (`marker_pos`) | static output shape; more than 32 options raise an error |
| `int32` instead of `int64` inputs | the HTP has no int64 |
| additive `-1e4` attention masks instead of bool / `-inf` | in fp16 a fully masked padding row (a pad query outside every sliding window) turns into NaN, which reaches real tokens through `0 × NaN` in the next layer |
| eager attention (matmul, softmax, matmul) | SDPA and HF's dynamic mask paths do not convert |
| sliding-window band taken from HF's own mask function | the mask is identical to upstream |

### 3. Compile: ONNX → fp16 DLC → one context binary

```bash
source ~/qairt-venv/bin/activate && source $SDK/bin/envsetup.sh
bash compile/compile_local.sh english         onnx out/english         128 256 512
bash compile/compile_local.sh multilingual    onnx out/multilingual    128 256 512 1024
bash compile/compile_local.sh typed-decisions onnx out/typed-decisions 128 256 512 1024
```

[`compile_local.sh`](compile/compile_local.sh) runs two SDK tools:

1. `qairt-converter --float_bitwidth 16` turns each ONNX file into an fp16 DLC.
2. `qnn-context-binary-generator` with `libQnnModelDlc.so` prepares all DLCs for the HTP and writes
   **one context binary** with one graph per length. The HTP config sets `soc_model` 66
   (QCS8550; the phone SM8550 is 43, see "Supported Snapdragon devices" in the SDK docs),
   `dsp_arch` v73, optimization level 3, the full VTCM, and `weight_sharing_enabled`, so the weights
   are stored once for all lengths.

Measured on a laptop (Intel Core i7-10750H, 6 cores, WSL2 with 15 GB RAM) in a clean run of this
README:

| Checkpoint | Lengths | Export (step 2) | Compile (step 3) | `.bin` size |
|---|---|---|---|---|
| `english` | 128, 256, 512 | 2.6 min | 6.9 min | 857 MB |
| `multilingual` | 128, 256, 512, 1024 | 2.3 min | 7.3 min | 664 MB |
| `typed-decisions` | 128, 256, 512, 1024 | 3.2 min | 8.6 min | 876 MB |

Step 1 took 7 minutes and steps 4–5 on the board under 2 minutes, so the whole route from the SDK
zip to a first answer on the NPU takes about 40 minutes (plus downloads). Two clean runs a day apart
gave byte-identical context binaries.

Each DLC is about the size of the full fp16 weights, while the context binary with 3–4 graphs is
barely larger than one, which shows the weights are shared. The converter peaks above 9 GB of RAM
on the 512-token graph of a large checkpoint.

The generator prints `<E> unexpected GraphHtpSettings option 66` once per graph. It appears even
with no config file at all, the binary is still written, and its answers match the CPU (see the
benchmarks), so it can be ignored.

### 4. Board: QAIRT runtime

Copy the runtime for the board from the same SDK. The libraries come from the
`aarch64-oe-linux-gcc11.2` build, which the SDK lists for the QCS8550 and which runs on the
board's Ubuntu 22.04 (glibc 2.35):

```bash
ssh rb5 'mkdir -p /opt/qnn2.51/lib /opt/qnn2.51/skel'
scp $SDK/lib/aarch64-oe-linux-gcc11.2/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV73Stub.so,libQnnHtpPrepare.so} \
    rb5:/opt/qnn2.51/lib/
scp $SDK/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so rb5:/opt/qnn2.51/skel/
```

The runtime version on the board must be the same as, or newer than, the SDK that built the binary.
An older runtime refuses it with `Using newer context binary on old SDK` (error 5000).

### 5. Board: runner and Python environment

`libqnnrt.so` is built on the board from [qnnrt.cpp](runtime/qnnrt/qnnrt.cpp) and the QNN SampleApp
utilities from the SDK (Qualcomm code, so it is copied from your SDK rather than included here):

```bash
# from the repository root on the host (only the code: onnx/ and out/ hold ~25 GB of build files)
ssh rb5 'mkdir -p /opt/laya-qnpu/models /opt/laya-qnpu/results /opt/qnn-sampleapp/qnn_include'
scp -r runtime bench rb5:/opt/laya-qnpu/
scp -r $SDK/examples/QNN/SampleApp/SampleApp/src rb5:/opt/qnn-sampleapp/
scp -r $SDK/include/QNN rb5:/opt/qnn-sampleapp/qnn_include/

ssh rb5
cd /opt/laya-qnpu
bash runtime/qnnrt/build.sh /opt/qnn-sampleapp
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
venv/bin/pip install -r runtime/requirements-board.txt
```

torch on the board is only needed for the CPU baseline and Laya's own tokenizer and head code; the
NPU path does not use it for the encoder.

### 6. Run

```bash
# from the repository root on the host
scp out/english/english.bin out/multilingual/multilingual.bin \
    out/typed-decisions/typed-decisions.bin rb5:/opt/laya-qnpu/models/
```

```bash
# on the board
cd /opt/laya-qnpu && export QNN_DIR=/opt/qnn2.51
venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "runtime")
from laya_npu import NPURouter                 # one NPUAgent per models/<name>.bin, CPU otherwise
router = NPURouter()
print(router.predict("We were billed twice for March. Refund it today or we cancel.", {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, errors", "other": "everything else"}},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to leave?"}}))
PY
```

The first call loads the context binary onto the HTP (a few seconds, plus the one-time download of
the Laya weights); later calls reuse it.
`NPUAgent(repo, binary, subfolder=...)` can also be used directly for one checkpoint.

### 7. Benchmark

```bash
cd /opt/laya-qnpu/bench && export QNN_DIR=/opt/qnn2.51
for m in english multilingual typed-decisions; do
    ../venv/bin/python bench_lengths.py --model $m --binary ../models/$m.bin --out ../results/$m.json
    ../venv/bin/python parity.py --model $m --binary ../models/$m.bin --out ../results/parity_$m.json
done
../venv/bin/python sustained.py --model english --binary ../models/english.bin --L 128 --seconds 300 \
    --out ../results/sustained_english_l128.json
```

## Licenses

Laya is by its authors ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), Apache-2.0).
The QAIRT SDK and its SampleApp are Qualcomm software under Qualcomm's license and are downloaded
separately.
