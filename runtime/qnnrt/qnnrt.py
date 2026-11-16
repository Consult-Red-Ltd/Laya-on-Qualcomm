"""ctypes binding for libqnnrt.so: a QNN context binary kept resident on the HTP."""
import ctypes
import os

import numpy as np

QNN_DIR = os.environ.get("QNN_DIR", "/opt/qnn")   # lib/ (libQnnHtp.so, ...) and skel/ (libQnnHtpV73Skel.so)
_DTYPES = {  # Qnn_DataType_t -> numpy
    0x0008: np.int8, 0x0016: np.int16, 0x0032: np.int32, 0x0064: np.int64,
    0x0108: np.uint8, 0x0116: np.uint16, 0x0132: np.uint32, 0x0164: np.uint64,
    0x0216: np.float16, 0x0232: np.float32, 0x0264: np.float64,
    0x0308: np.int8, 0x0316: np.int16, 0x0332: np.int32,      # SFIXED
    0x0408: np.uint8, 0x0416: np.uint16, 0x0432: np.uint32,   # UFIXED
    0x0508: np.uint8,                                         # BOOL_8
}


class Tensor:
    def __init__(self, name, dtype_code, dims, scale, offset):
        self.name, self.code, self.shape = name, dtype_code, tuple(dims)
        self.dtype = np.dtype(_DTYPES[dtype_code])
        self.scale, self.offset = scale, offset
        self.quantized = (dtype_code >> 8) in (3, 4) and scale != 0.0

    def to_native(self, x):
        x = np.asarray(x)
        if self.quantized:
            info = np.iinfo(self.dtype)
            x = np.clip(np.round(x / self.scale) - self.offset, info.min, info.max)
        return np.ascontiguousarray(x.astype(self.dtype, copy=False).reshape(self.shape))

    def from_native(self, x):
        if self.quantized:
            return (x.astype(np.float32) + self.offset) * self.scale
        return x


class Graph:
    def __init__(self, ctx, index):
        self.ctx, self.index = ctx, index
        self.name = ctx.lib.qnnrt_graph_name(ctx.h, index).decode()
        self.inputs = [ctx._info(index, 0, i) for i in range(ctx.lib.qnnrt_num_io(ctx.h, index, 0))]
        self.outputs = [ctx._info(index, 1, i) for i in range(ctx.lib.qnnrt_num_io(ctx.h, index, 1))]
        self._out = [np.empty(t.shape, t.dtype) for t in self.outputs]

    def __call__(self, feeds):
        """feeds: {input name: array} (or a list in input order). Returns {output name: array}."""
        if isinstance(feeds, dict):
            feeds = [feeds[t.name] for t in self.inputs]
        ins = [t.to_native(x) for t, x in zip(self.inputs, feeds)]
        in_p = (ctypes.c_void_p * len(ins))(*[a.ctypes.data for a in ins])
        out_p = (ctypes.c_void_p * len(self._out))(*[a.ctypes.data for a in self._out])
        rc = self.ctx.lib.qnnrt_execute(self.ctx.h, self.index, in_p, out_p)
        if rc:
            raise RuntimeError("graphExecute(%s) failed: %d" % (self.name, rc))
        return {t.name: t.from_native(a).copy() for t, a in zip(self.outputs, self._out)}


class Context:
    def __init__(self, path, qnn_dir=QNN_DIR, log_level=0, burst=True,
                 lib=os.path.join(os.path.dirname(os.path.abspath(__file__)), "libqnnrt.so")):
        os.environ.setdefault("ADSP_LIBRARY_PATH", os.path.join(qnn_dir, "skel"))
        self.lib = L = ctypes.CDLL(lib)
        L.qnnrt_open.restype = ctypes.c_void_p
        L.qnnrt_open.argtypes = [ctypes.c_char_p] * 3 + [ctypes.c_int, ctypes.c_int]
        L.qnnrt_graph_name.restype = ctypes.c_char_p
        for f in ("qnnrt_num_graphs", "qnnrt_graph_name", "qnnrt_num_io", "qnnrt_io_info",
                  "qnnrt_execute", "qnnrt_close"):
            getattr(L, f).argtypes = [ctypes.c_void_p] + {
                "qnnrt_num_graphs": [], "qnnrt_graph_name": [ctypes.c_int],
                "qnnrt_num_io": [ctypes.c_int, ctypes.c_int],
                "qnnrt_io_info": [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)],
                "qnnrt_execute": [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
                "qnnrt_close": []}[f]
        self.h = L.qnnrt_open(os.path.join(qnn_dir, "lib", "libQnnHtp.so").encode(),
                              os.path.join(qnn_dir, "lib", "libQnnSystem.so").encode(),
                              path.encode(), log_level, int(burst))
        if not self.h:
            raise RuntimeError("qnnrt_open failed for %s (see stderr)" % path)
        self.graphs = [Graph(self, g) for g in range(L.qnnrt_num_graphs(self.h))]

    def _info(self, g, is_out, i):
        name, dt, rank = ctypes.c_char_p(), ctypes.c_int(), ctypes.c_int()
        dims, scale, off = (ctypes.c_uint32 * 8)(), ctypes.c_float(), ctypes.c_int()
        self.lib.qnnrt_io_info(self.h, g, is_out, i, ctypes.byref(name), ctypes.byref(dt),
                               ctypes.byref(rank), dims, ctypes.byref(scale), ctypes.byref(off))
        return Tensor(name.value.decode(), dt.value, dims[:rank.value], scale.value, off.value)

    def close(self):
        if self.h:
            self.lib.qnnrt_close(self.h)
            self.h = None

    def __del__(self):
        self.close()
