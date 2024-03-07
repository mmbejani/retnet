#include "macro/device.h"
#include "macro/constant.h"
#include "macro/initialization.h"

#include "operation/vector_op.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

#ifdef OMP
#include <omp.h>
#elif SIMD
#include <immintrin.h>
#endif

void vec_dot_prod_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ y, float32 *__restrict__ result)
{
    void *vz;
    if (posix_memalign(&vz, F32x8_AVX_ALGIN, sizeof(float32) * n) != 0)
    {
        perror("The allocation cannot be done currectly");
        exit(EXIT_FAILURE);
    }
    float32 *z = (float32 *)vz;
#ifdef AVX
    const int n_partial = (n & ~(F32_AVX_NREG - 1));
#define F32_VEC_4 4

    __m128 vec_sum = _mm_setzero_ps();

    __m128 x_partial;
    __m128 y_partial;

    for (int i = 0; i < n_partial; i += F32_AVX_NREG)
    {
        ax = _mm_loadu_ps(x + i);
        ay = _mm_loadu_ps(y + i);

        vec_sum = _mm_fmadd_ps(ax, ay, sums);
    }

    vec_reduce_sum_f32(F32_AVX_NREG, &vec_sum, result);

    float32 rediual = 0.0f;
    for (int i = n_partial; i < n; ++i)
        rediual += x[i] * y[i];
    *result += rediual;
    return;
#else
#ifdef OMP
    SET_OMP_THREADS();

#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
        z[i] = x[i] * y[i];

    vec_reduce_sum_f32(n, z, result);
#endif

    return;
}

void vec_dot_prod_f16(const size_t n, float16 *__restrict__ x, float16 *__restrict__ y, float16 *__restrict__ result)
{
    void *vz;
    if (posix_memalign(&vz, F32x8_AVX_ALGIN, sizeof(float32) * n) != 0)
    {
        perror("The allocation cannot be done currectly");
        exit(EXIT_FAILURE);
    }
    float32 *z = (float32 *)vz;
#ifdef AVX
    const int n_partial = (n & ~(F16_AVX_NREG - 1));
#define F16_VEC_8 8

    __m128h vec_sum = _mm_setzero_ph();

    __m128h x_partial;
    __m128h y_partial;

    for (int i = 0; i < n_partial; i += F16_AVX_NREG)
    {
        ax = _mm_loadu_ph(x + i);
        ay = _mm_loadu_ph(y + i);

        vec_sum = _mm_fmadd_ph(ax, ay, sums);
    }

    vec_reduce_sum_f16(F32_AVX_NREG, &vec_sum, result);

    //[TODO] the remain values should be convert to fp-32 and normally computed
#else
    perror("The fp-16 is not supported on this device");
    exit(EXIT_FAILURE);
#endif
}

void vec_reduce_sum_f32(const size_t n, void *__restrict__ vx, float32 *__restrict__ result)
{
    float32 r = 0;
#ifdef OMP
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
#elif AVX
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

void vec_reduce_sum_f16(const size_t n, void *__restrict__ vx, float16 *__restruct__)
{
#ifdef AVX
    __m128h x = *(__m128h *)vx;
    __m128h reduced_x = _mm_hadd_ph(x, x);
#else
    perror("FP16 is not support by this device");
    exit(EXIT_FAILURE)
#endif
}