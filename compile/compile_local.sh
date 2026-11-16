#!/usr/bin/env bash
# ONNX buckets -> fp16 DLCs -> one weight-shared HTP context binary for the QCS8550 (Hexagon V73),
# with the QAIRT SDK on an x86_64 Linux host (native Ubuntu or WSL2). No cloud service involved.
#
#   source $QAIRT_SDK_ROOT/bin/envsetup.sh        # and activate the venv that has the SDK's Python deps
#   ./compile_local.sh english ../onnx out 128 256 512
#
# Weight sharing (one copy of the weights for all buckets) is only supported for offline prepare
# on x86_64 Linux, which is why this does not run on Windows or on the board itself.
set -euo pipefail
: "${QAIRT_SDK_ROOT:?source \$QAIRT_SDK_ROOT/bin/envsetup.sh first}"

MODEL=$1 ONNX_DIR=$2 OUT=$3
shift 3
[ $# -gt 0 ] || { echo "usage: $0 <model> <onnx_dir> <out_dir> L [L ...]" >&2; exit 1; }
BIN=$QAIRT_SDK_ROOT/bin/x86_64-linux-clang
LIB=$QAIRT_SDK_ROOT/lib/x86_64-linux-clang
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# 1. ONNX -> DLC, fp16 (no quantization, no calibration data)
dlcs=() graphs=()
for L in "$@"; do
    name=${MODEL}_l${L}
    if [ ! -f "$OUT/$name.dlc" ]; then
        "$BIN/qairt-converter" --input_network "$ONNX_DIR/${MODEL}_b1_l${L}_m32/model.onnx" \
            --float_bitwidth 16 --output_path "$OUT/$name.dlc"
    fi
    dlcs+=("$OUT/$name.dlc")
    graphs+=("\"$name\"")
done

# 2. HTP backend options: QCS8550 (soc_model 66, V73), optimization level 3, full VTCM, weight sharing
IFS=,
cat > "$OUT/htp_backend.json" <<JSON
{
  "graphs": [{"graph_names": [${graphs[*]}], "vtcm_mb": 0, "O": 3}],
  "devices": [{"soc_model": 66, "dsp_arch": "v73"}],
  "context": {"weight_sharing_enabled": true}
}
JSON
cat > "$OUT/htp_config.json" <<JSON
{"backend_extensions": {"shared_library_path": "$LIB/libQnnHtpNetRunExtensions.so",
                        "config_file_path": "$OUT/htp_backend.json"}}
JSON

# 3. All DLCs -> one context binary with one graph per bucket
"$BIN/qnn-context-binary-generator" \
    --backend "$LIB/libQnnHtp.so" --model "$LIB/libQnnModelDlc.so" --dlc_path "${dlcs[*]}" \
    --config_file "$OUT/htp_config.json" --binary_file "$MODEL" --output_dir "$OUT"
unset IFS
ls -l "$OUT/$MODEL.bin"
