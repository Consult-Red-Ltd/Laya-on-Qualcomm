// qnnrt: tiny resident QNN runner exposed as a C API (for Python ctypes).
//
// Loads a context binary once (all graphs, shared weights), keeps it on the HTP and executes
// graphs in-process with plain client buffers -- no per-call context reload (qnn-net-run) and
// no file I/O (patched qnn-sample-app server). Built on the QNN SampleApp utilities.
//
// Build (natively on the device, next to the SampleApp sources):
//   g++ -std=gnu++17 -O2 -fPIC -shared -w -I$SA/src -I$SA/src/Log -I$SA/src/Utils \
//       -I$SA/src/WrapperUtils -I$SA/src/PAL/include -I$SA/qnn_include/QNN \
//       qnnrt.cpp $SA/src/Utils/*.cpp $SA/src/WrapperUtils/*.cpp $SA/src/Log/*.cpp \
//       $SA/src/PAL/src/linux/*.cpp $SA/src/PAL/src/common/*.cpp -ldl -o libqnnrt.so
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "DynamicLoadUtil.hpp"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnSampleAppUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"

using namespace qnn::tools;
using qnn_wrapper_api::GraphInfo_t;

namespace {

struct Runner {
  sample_app::QnnFunctionPointers fp{};
  void* backendLib = nullptr;
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  GraphInfo_t** graphs = nullptr;
  uint32_t numGraphs = 0;
  uint32_t powerConfigId = 0;
  QnnHtpDevice_PerfInfrastructure_t* perf = nullptr;
  std::mutex mu;
};

void logCb(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  fprintf(stderr, "[qnnrt %d] ", (int)level);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

size_t dtypeSize(Qnn_DataType_t t) {
  switch (t) {
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 4;
  }
}

size_t tensorBytes(const Qnn_Tensor_t& t) {
  size_t n = dtypeSize(QNN_TENSOR_GET_DATA_TYPE(t));
  for (uint32_t i = 0; i < QNN_TENSOR_GET_RANK(t); i++) n *= QNN_TENSOR_GET_DIMENSIONS(t)[i];
  return n;
}

// Max-performance vote: DCVS off, performance mode, no HTP sleep, low RPC latency.
void setBurst(Runner* r) {
  QnnDevice_Infrastructure_t infra = nullptr;
  if (!r->fp.qnnInterface.deviceGetInfrastructure ||
      r->fp.qnnInterface.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
    fprintf(stderr, "[qnnrt] no device infrastructure, skipping burst\n");
    return;
  }
  auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  r->perf = &htp->perfInfra;
  if (r->perf->createPowerConfigId(0, 0, &r->powerConfigId) != QNN_SUCCESS) {
    fprintf(stderr, "[qnnrt] createPowerConfigId failed\n");
    r->perf = nullptr;
    return;
  }
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& d = dcvs.dcvsV3Config;
  d.contextId = r->powerConfigId;
  d.setDcvsEnable = 1; d.dcvsEnable = 0;
  d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  d.setSleepLatency = 1; d.sleepLatency = 40;
  d.setSleepDisable = 1; d.sleepDisable = 1;
  d.setBusParams = 1;
  d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.setCoreParams = 1;
  d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  QnnHtpPerfInfrastructure_PowerConfig_t lat = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  lat.rpcControlLatencyConfig = 100;
  QnnHtpPerfInfrastructure_PowerConfig_t poll = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  poll.rpcPollingTimeConfig = 9999;
  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &lat, &poll, nullptr};
  if (r->perf->setPowerConfig(r->powerConfigId, cfgs) != QNN_SUCCESS)
    fprintf(stderr, "[qnnrt] setPowerConfig failed\n");
}

}  // namespace

extern "C" {

void qnnrt_close(void* h);

void* qnnrt_open(const char* backendPath, const char* systemPath, const char* ctxPath, int logLevel,
                 int burst) {
  auto* r = new Runner();
  auto fail = [&](const char* msg) -> void* {
    fprintf(stderr, "[qnnrt] %s\n", msg);
    qnnrt_close(r);
    return nullptr;
  };
  if (dynamicloadutil::getQnnFunctionPointers(backendPath, "", &r->fp, &r->backendLib, false, nullptr) !=
      dynamicloadutil::StatusCode::SUCCESS)
    return fail("cannot load backend");
  if (dynamicloadutil::getQnnSystemFunctionPointers(systemPath, &r->fp) !=
      dynamicloadutil::StatusCode::SUCCESS)
    return fail("cannot load system library");
  if (logLevel > 0) r->fp.qnnInterface.logCreate(logCb, (QnnLog_Level_t)logLevel, &r->log);
  if (r->fp.qnnInterface.backendCreate(r->log, nullptr, &r->backend) != QNN_SUCCESS)
    return fail("backendCreate failed");
  if (r->fp.qnnInterface.deviceCreate &&
      r->fp.qnnInterface.deviceCreate(r->log, nullptr, &r->device) != QNN_SUCCESS)
    return fail("deviceCreate failed");

  std::ifstream f(ctxPath, std::ios::binary | std::ios::ate);
  if (!f) return fail("cannot open context binary");
  size_t size = f.tellg();
  std::unique_ptr<uint8_t[]> buf(new uint8_t[size]);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(buf.get()), size);

  QnnSystemContext_Handle_t sys = nullptr;
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (r->fp.qnnSystemInterface.systemContextCreate(&sys) != QNN_SUCCESS)
    return fail("systemContextCreate failed");
  bool ok = r->fp.qnnSystemInterface.systemContextGetBinaryInfo(sys, buf.get(), size, &info, &infoSize) ==
                QNN_SUCCESS &&
            sample_app::copyMetadataToGraphsInfo(info, r->graphs, r->numGraphs);
  r->fp.qnnSystemInterface.systemContextFree(sys);
  if (!ok) return fail("cannot read context binary metadata");

  if (r->fp.qnnInterface.contextCreateFromBinary(r->backend, r->device, nullptr, buf.get(), size,
                                                 &r->context, nullptr) != QNN_SUCCESS)
    return fail("contextCreateFromBinary failed");
  for (uint32_t g = 0; g < r->numGraphs; g++)
    if (r->fp.qnnInterface.graphRetrieve(r->context, r->graphs[g]->graphName, &r->graphs[g]->graph) !=
        QNN_SUCCESS)
      return fail("graphRetrieve failed");
  if (burst) setBurst(r);
  return r;
}

int qnnrt_num_graphs(void* h) { return h ? (int)static_cast<Runner*>(h)->numGraphs : 0; }

const char* qnnrt_graph_name(void* h, int g) { return static_cast<Runner*>(h)->graphs[g]->graphName; }

int qnnrt_num_io(void* h, int g, int isOut) {
  auto* gi = static_cast<Runner*>(h)->graphs[g];
  return isOut ? (int)gi->numOutputTensors : (int)gi->numInputTensors;
}

// dtype: raw Qnn_DataType_t; dims: up to 8; scale/offset valid for scale-offset quantization.
int qnnrt_io_info(void* h, int g, int isOut, int i, const char** name, int* dtype, int* rank,
                  uint32_t* dims, float* scale, int* offset) {
  auto* gi = static_cast<Runner*>(h)->graphs[g];
  const Qnn_Tensor_t& t = isOut ? gi->outputTensors[i] : gi->inputTensors[i];
  *name = QNN_TENSOR_GET_NAME(t);
  *dtype = (int)QNN_TENSOR_GET_DATA_TYPE(t);
  *rank = (int)QNN_TENSOR_GET_RANK(t);
  for (int k = 0; k < *rank && k < 8; k++) dims[k] = QNN_TENSOR_GET_DIMENSIONS(t)[k];
  Qnn_QuantizeParams_t q = QNN_TENSOR_GET_QUANT_PARAMS(t);
  *scale = 0.f;
  *offset = 0;
  if (q.encodingDefinition == QNN_DEFINITION_DEFINED &&
      q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    *scale = q.scaleOffsetEncoding.scale;
    *offset = q.scaleOffsetEncoding.offset;
  }
  return 0;
}

// Buffers must hold exactly the tensor's native dtype and shape (see qnnrt_io_info).
int qnnrt_execute(void* h, int g, void** inBufs, void** outBufs) {
  auto* r = static_cast<Runner*>(h);
  std::lock_guard<std::mutex> lock(r->mu);
  auto* gi = r->graphs[g];
  std::vector<Qnn_Tensor_t> in(gi->inputTensors, gi->inputTensors + gi->numInputTensors);
  std::vector<Qnn_Tensor_t> out(gi->outputTensors, gi->outputTensors + gi->numOutputTensors);
  for (size_t i = 0; i < in.size(); i++) {
    QNN_TENSOR_SET_MEM_TYPE(in[i], QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(in[i], (Qnn_ClientBuffer_t{inBufs[i], (uint32_t)tensorBytes(in[i])}));
  }
  for (size_t i = 0; i < out.size(); i++) {
    QNN_TENSOR_SET_MEM_TYPE(out[i], QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(out[i], (Qnn_ClientBuffer_t{outBufs[i], (uint32_t)tensorBytes(out[i])}));
  }
  Qnn_ErrorHandle_t e = r->fp.qnnInterface.graphExecute(gi->graph, in.data(), in.size(), out.data(),
                                                        out.size(), nullptr, nullptr);
  return e == QNN_SUCCESS ? 0 : (int)e;
}

void qnnrt_close(void* h) {
  auto* r = static_cast<Runner*>(h);
  if (!r) return;
  if (r->perf) r->perf->destroyPowerConfigId(r->powerConfigId);
  if (r->context) r->fp.qnnInterface.contextFree(r->context, nullptr);
  if (r->device && r->fp.qnnInterface.deviceFree) r->fp.qnnInterface.deviceFree(r->device);
  if (r->backend) r->fp.qnnInterface.backendFree(r->backend);
  if (r->log) r->fp.qnnInterface.logFree(r->log);
  delete r;
}

}  // extern "C"
