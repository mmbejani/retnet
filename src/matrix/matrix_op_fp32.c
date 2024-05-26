#include "matrix_op_fp32.h"

#ifdef USE_MKL
#include "mkl.h"

inline void matmul_mkl(float32 *w, float32 *x, float32 *y, size_t *dims)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dims[0], dims[1], dims[2], 1.0f, w, dims[1], x, dims[2], 0.f, y, dims[2]);
}
#else
#include "vector/vector_op_fp32.h"
#ifdef USE_OMP
#include "initialization.h"
#include <omp.h>
#endif
inline void matmul_plain(const float32 *w, const float32 *x, float32 *__restrict__ y, size_t *dims)
{
#ifdef USE_OMP
    SET_OMP_THREADS();
#pragma omp parallel for
#endif
    for (size_t i = 0; i < dims[0]; i++)
    {
        for (size_t j = 0; j < dims[1]; j++)
            vec_dot_prod_f32(dims[2], w + (i * dims[0]), x + (j * dims[1]), y + (i * dims[0] + j));
    }
}
#endif

void matmul(const float32 *w, const float32 *x, float32 *y, size_t *dims)
{
#ifdef USE_MKL
    matmul_mkl(w, x, y, dims);
#else
    matmul_plain(w, x, y, dims);
#endif
}