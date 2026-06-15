#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL CCONV_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "cconv.h"

static inline npy_intp clamp_index(npy_intp value, npy_intp low, npy_intp high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline float sample_replicate(
    const float *src,
    npy_intp height,
    npy_intp width,
    npy_intp y,
    npy_intp x
) {
    npy_intp yy = clamp_index(y, 0, height - 1);
    npy_intp xx = clamp_index(x, 0, width - 1);
    return src[yy * width + xx];
}

PyObject *conv5x5(PyObject *self, PyObject *args) {
    PyObject *image_obj = NULL;
    PyObject *kernel_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &image_obj, &kernel_obj)) {
        return NULL;
    }

    PyArrayObject *image = (PyArrayObject *)PyArray_FROM_OTF(
        image_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *kernel = (PyArrayObject *)PyArray_FROM_OTF(
        kernel_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );

    if (image == NULL || kernel == NULL) {
        Py_XDECREF(image);
        Py_XDECREF(kernel);
        return NULL;
    }

    if (PyArray_NDIM(image) != 2) {
        PyErr_SetString(PyExc_ValueError, "image must be a 2D grayscale array");
        Py_DECREF(image);
        Py_DECREF(kernel);
        return NULL;
    }

    if (PyArray_NDIM(kernel) != 2 || PyArray_DIM(kernel, 0) != 5 || PyArray_DIM(kernel, 1) != 5) {
        PyErr_SetString(PyExc_ValueError, "kernel must have shape (5, 5)");
        Py_DECREF(image);
        Py_DECREF(kernel);
        return NULL;
    }

    npy_intp height = PyArray_DIM(image, 0);
    npy_intp width = PyArray_DIM(image, 1);
    npy_intp dims[2] = {height, width};

    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (output == NULL) {
        Py_DECREF(image);
        Py_DECREF(kernel);
        return NULL;
    }

    const float *src = (const float *)PyArray_DATA(image);
    const float *k = (const float *)PyArray_DATA(kernel);
    float *dst = (float *)PyArray_DATA(output);

    if (height < 5 || width < 5) {
        for (npy_intp y = 0; y < height; ++y) {
            for (npy_intp x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (npy_intp ky = 0; ky < 5; ++ky) {
                    for (npy_intp kx = 0; kx < 5; ++kx) {
                        acc += sample_replicate(src, height, width, y + ky - 2, x + kx - 2) * k[ky * 5 + kx];
                    }
                }
                dst[y * width + x] = acc;
            }
        }
    } else {
        for (npy_intp y = 0; y < height; ++y) {
            for (npy_intp x = 0; x < width; ++x) {
                if (y >= 2 && y < height - 2 && x >= 2 && x < width - 2) {
                    const float *p = src + (y - 2) * width + (x - 2);
                    float acc =
                        p[0] * k[0] + p[1] * k[1] + p[2] * k[2] + p[3] * k[3] + p[4] * k[4] +
                        p[width] * k[5] + p[width + 1] * k[6] + p[width + 2] * k[7] +
                        p[width + 3] * k[8] + p[width + 4] * k[9] +
                        p[2 * width] * k[10] + p[2 * width + 1] * k[11] +
                        p[2 * width + 2] * k[12] + p[2 * width + 3] * k[13] +
                        p[2 * width + 4] * k[14] +
                        p[3 * width] * k[15] + p[3 * width + 1] * k[16] +
                        p[3 * width + 2] * k[17] + p[3 * width + 3] * k[18] +
                        p[3 * width + 4] * k[19] +
                        p[4 * width] * k[20] + p[4 * width + 1] * k[21] +
                        p[4 * width + 2] * k[22] + p[4 * width + 3] * k[23] +
                        p[4 * width + 4] * k[24];
                    dst[y * width + x] = acc;
                } else {
                    float acc = 0.0f;
                    for (npy_intp ky = 0; ky < 5; ++ky) {
                        for (npy_intp kx = 0; kx < 5; ++kx) {
                            acc += sample_replicate(src, height, width, y + ky - 2, x + kx - 2) * k[ky * 5 + kx];
                        }
                    }
                    dst[y * width + x] = acc;
                }
            }
        }
    }

    Py_DECREF(image);
    Py_DECREF(kernel);
    return (PyObject *)output;
}
