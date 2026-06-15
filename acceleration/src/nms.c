#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL CCONV_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "cconv.h"

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

PyObject *nms_xyxy(PyObject *self, PyObject *args) {
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
