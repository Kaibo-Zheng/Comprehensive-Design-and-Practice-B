#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

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

static PyObject *conv5x5(PyObject *self, PyObject *args) {
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

typedef struct {
    double score;
    npy_intp index;
} ScoreIndex;

static int compare_score_desc(const void *left, const void *right) {
    const ScoreIndex *a = (const ScoreIndex *)left;
    const ScoreIndex *b = (const ScoreIndex *)right;
    if (a->score < b->score) {
        return 1;
    }
    if (a->score > b->score) {
        return -1;
    }
    if (a->index < b->index) {
        return 1;
    }
    if (a->index > b->index) {
        return -1;
    }
    return 0;
}

static inline double box_iou_xyxy(const double *boxes, npy_intp i, npy_intp j) {
    const double ax1 = boxes[i * 4 + 0];
    const double ay1 = boxes[i * 4 + 1];
    const double ax2 = boxes[i * 4 + 2];
    const double ay2 = boxes[i * 4 + 3];
    const double bx1 = boxes[j * 4 + 0];
    const double by1 = boxes[j * 4 + 1];
    const double bx2 = boxes[j * 4 + 2];
    const double by2 = boxes[j * 4 + 3];

    const double inter_x1 = ax1 > bx1 ? ax1 : bx1;
    const double inter_y1 = ay1 > by1 ? ay1 : by1;
    const double inter_x2 = ax2 < bx2 ? ax2 : bx2;
    const double inter_y2 = ay2 < by2 ? ay2 : by2;
    const double inter_w = inter_x2 > inter_x1 ? inter_x2 - inter_x1 : 0.0;
    const double inter_h = inter_y2 > inter_y1 ? inter_y2 - inter_y1 : 0.0;
    const double inter = inter_w * inter_h;

    const double aw = ax2 > ax1 ? ax2 - ax1 : 0.0;
    const double ah = ay2 > ay1 ? ay2 - ay1 : 0.0;
    const double bw = bx2 > bx1 ? bx2 - bx1 : 0.0;
    const double bh = by2 > by1 ? by2 - by1 : 0.0;
    const double union_area = aw * ah + bw * bh - inter;
    return union_area > 0.0 ? inter / union_area : 0.0;
}

static PyObject *nms_xyxy(PyObject *self, PyObject *args) {
    PyObject *boxes_obj = NULL;
    PyObject *scores_obj = NULL;
    double iou_threshold = 0.45;

    if (!PyArg_ParseTuple(args, "OO|d", &boxes_obj, &scores_obj, &iou_threshold)) {
        return NULL;
    }

    PyArrayObject *boxes = (PyArrayObject *)PyArray_FROM_OTF(
        boxes_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *scores = (PyArrayObject *)PyArray_FROM_OTF(
        scores_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY
    );

    if (boxes == NULL || scores == NULL) {
        Py_XDECREF(boxes);
        Py_XDECREF(scores);
        return NULL;
    }

    if (PyArray_NDIM(boxes) != 2 || PyArray_DIM(boxes, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "boxes must have shape (N, 4)");
        Py_DECREF(boxes);
        Py_DECREF(scores);
        return NULL;
    }
    if (PyArray_NDIM(scores) != 1 || PyArray_DIM(scores, 0) != PyArray_DIM(boxes, 0)) {
        PyErr_SetString(PyExc_ValueError, "scores must have shape (N,)");
        Py_DECREF(boxes);
        Py_DECREF(scores);
        return NULL;
    }

    const npy_intp count = PyArray_DIM(boxes, 0);
    npy_intp dims[1] = {count};
    PyArrayObject *keep = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (keep == NULL) {
        Py_DECREF(boxes);
        Py_DECREF(scores);
        return NULL;
    }

    ScoreIndex *order = PyMem_New(ScoreIndex, count > 0 ? count : 1);
    unsigned char *suppressed = PyMem_Calloc(count > 0 ? count : 1, sizeof(unsigned char));
    if (order == NULL || suppressed == NULL) {
        PyMem_Free(order);
        PyMem_Free(suppressed);
        Py_DECREF(boxes);
        Py_DECREF(scores);
        Py_DECREF(keep);
        return PyErr_NoMemory();
    }

    const double *box_data = (const double *)PyArray_DATA(boxes);
    const double *score_data = (const double *)PyArray_DATA(scores);
    for (npy_intp i = 0; i < count; ++i) {
        order[i].score = score_data[i];
        order[i].index = i;
    }
    qsort(order, (size_t)count, sizeof(ScoreIndex), compare_score_desc);

    npy_intp keep_count = 0;
    npy_intp *keep_data = (npy_intp *)PyArray_DATA(keep);
    for (npy_intp oi = 0; oi < count; ++oi) {
        const npy_intp current = order[oi].index;
        if (suppressed[current]) {
            continue;
        }
        keep_data[keep_count++] = current;
        for (npy_intp oj = oi + 1; oj < count; ++oj) {
            const npy_intp candidate = order[oj].index;
            if (!suppressed[candidate] && box_iou_xyxy(box_data, current, candidate) > iou_threshold) {
                suppressed[candidate] = 1;
            }
        }
    }

    PyMem_Free(order);
    PyMem_Free(suppressed);
    Py_DECREF(boxes);
    Py_DECREF(scores);

    PyArray_Dims new_dims = {.ptr = &keep_count, .len = 1};
    if (PyArray_Resize(keep, &new_dims, 0, NPY_ANYORDER) == NULL) {
        Py_DECREF(keep);
        return NULL;
    }
    return (PyObject *)keep;
}

static PyMethodDef Methods[] = {
    {"conv5x5", conv5x5, METH_VARARGS, "Apply edge-replicated 5x5 convolution."},
    {"nms_xyxy", nms_xyxy, METH_VARARGS, "Greedy NMS for xyxy boxes. Returns kept indices."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cconv",
    "C extension for 5x5 convolution.",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__cconv(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
