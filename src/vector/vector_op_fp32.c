#include <stdio.h>
#include <sys/sysinfo.h>

#define USE_MKL
#ifdef USE_MKL
#include "mkl.h"
#else
#ifdef USE_OMP
#include <omp.h>
#endif
#ifdef USE_AVX
#include <immintrin.h>
#endif
#endif

#include "logging.h"
#include "vector/vector_op_fp32.h"

void vec_dot_prod_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ y, float32 *__restrict__ result)
{
    void *vz;
    if (posix_memalign(&vz, F32x8_AVX_ALGIN, sizeof(float32) * n) != 0)
        logger(FATAL, "There is a error to allocate align memory");
    result = (float32 *)vz;
#ifdef USE_SIMD
    const int n_partial = (n & ~(F32_AVX_NREG - 1));
#define F32_VEC_4 4

    __m128 vec_sum = _mm_setzero_ps();

    __m128 x_partial;
    __m128 y_partial;

    for (int i = 0; i < n_partial; i += F32_AVX_NREG)
    {
        x_partial = _mm_loadu_ps(x + i);
        y_partial = _mm_loadu_ps(y + i);

        vec_sum = _mm_fmadd_ps(x_partial, y_partial, vec_sum);
    }

    vec_reduce_sum_f32(F32_AVX_NREG, &vec_sum, result);

    float32 rediual = 0.0f;
    for (int i = n_partial; i < n; ++i)
        rediual += x[i] * y[i];
    *result += rediual;
    return;
#else
#ifdef USE_OMP
    // use it with caution, this may make it slower! (the spwanning thread have overhead)
    SET_OMP_THREADS();

#pragma omp parallel for
#endif
    float32 *z = (float32 *)vz;
    for (size_t i = 0; i < n; i++)
        z[i] = x[i] * y[i];

    vec_reduce_sum_f32(n, z, result);
#endif

    return;
}

void vec_reduce_sum_f32(const size_t n, void *__restrict__ vx, float32 *__restrict__ result)
{
    float32 r = 0;
#ifdef USE_OMP
    float32 *x = (float32 *)vx;
    SET_OMP_THREADS();
    register size_t reduce_size = n;
    while (reduce_size % 2 == 0)
    {
        reduce_size >>= 1;
#pragma omp parallel for
        for (size_t i = 0; i < reduce_size; i++)
            x[i] += x[i + reduce_size];
    }

    for (size_t i = 0; i < reduce_size; i++)
        r += x[i];
#elif defined(USE_SIMD)
    __m128 x = *(__m128 *)vx;
    __m128 reduced_x = _mm_hadd_ps(x, x);
    *result = _mm_cvtss_f32(_mm_hadd_ps(reduced_x, reduced_x));
#else
    float32 *x = (float32 *)vx;
    for (size_t i = 0; i < n; i++)
        r += x[i];
#endif
    *result = r;
}