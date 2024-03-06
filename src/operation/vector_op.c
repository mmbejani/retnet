#include "macro/device.h"
#include "macro/constant.h"
#include "macro/initialization.h"

#include "operation/vector_op.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

void vec_dot_prod_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ y, float32 *__restrict__ result)
{
    void *vz;
    if (posix_memalign(&vz, F32x8_ALGIN, sizeof(float32) * n) != 0)
    {
        perror("The allocation cannot be done currectly");
        exit(EXIT_FAILURE);
    }
    float32 *z = (float32 *)vz;
#ifdef OMP
    SET_OMP_THREADS();

#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
        z[i] = x[i] * y[i];

    vec_reduce_sum_f32(n, z, result);

    return;
}

void vec_reduce_sum_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ result)
{
    float32 r = 0;
#ifdef OMP
    SET_OMP_THREADS();
    register size_t reduce_size = n >> 1;
    printf("The reduce size is %li\n", reduce_size & 0x00000001);
    while (reduce_size & 0x00000001 != 1UL)
    {
        printf("The reduce size is %li\n", reduce_size);
#pragma omp parallel for
        for (size_t i = 0; i < reduce_size; i++)
            x[i] += x[i + reduce_size];
        reduce_size >>= 1;
    }

    //for (size_t i = 0; i < reduce_size; i++)
    //    r += x[i];
#else
    for (size_t i = 0; i < n; i++)
        r += x[i];
#endif
    *result = r;
}