#ifndef CCONV_H
#define CCONV_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject *conv5x5(PyObject *self, PyObject *args);
PyObject *nms_xyxy(PyObject *self, PyObject *args);

#endif
