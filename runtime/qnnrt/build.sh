#!/usr/bin/env bash
# Build libqnnrt.so natively on the board against the QNN SampleApp sources of the SDK.
# The SampleApp is Qualcomm code and is not part of this repository: copy it from your SDK.
#
#   ./build.sh /path/to/SampleApp     # contains src/ (examples/QNN/SampleApp/SampleApp/src)
#                                     # and qnn_include/QNN (include/QNN)
set -euo pipefail
SA=${1:?usage: $0 <SampleApp dir with src/ and qnn_include/QNN>}
SA=$(cd "$SA" && pwd)                 # absolute, so it survives the cd below
cd "$(dirname "$0")"
g++ -std=gnu++17 -O2 -fPIC -shared -w \
    -I"$SA/src" -I"$SA/src/Log" -I"$SA/src/Utils" -I"$SA/src/WrapperUtils" -I"$SA/src/PAL/include" \
    -I"$SA/qnn_include/QNN" \
    qnnrt.cpp "$SA"/src/Utils/*.cpp "$SA"/src/WrapperUtils/*.cpp "$SA"/src/Log/*.cpp \
    "$SA"/src/PAL/src/linux/*.cpp "$SA"/src/PAL/src/common/*.cpp -ldl -o libqnnrt.so
ls -l libqnnrt.so
