#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL CCONV_ARRAY_API
#include <numpy/arrayobject.h>

#include "cconv.h"

static PyMethodDef Methods[] = {
    {"conv5x5", conv5x5, METH_VARARGS, "Apply edge-replicated 5x5 convolution."},
    {"nms_xyxy", nms_xyxy, METH_VARARGS, "Greedy NMS for xyxy boxes. Returns kept indices."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cconv",
    "C extension for 5x5 convolution and NMS.",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__cconv(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
