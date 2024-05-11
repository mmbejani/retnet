#include <assert.h>
#include <immintrin.h>

#include "vector_op/vector_op_q8.h"
#include "macro/constant.h"

inline float avx_vec_dot_q8(const __m256i x, const __m256i y, float scale, float bias)
{
    __m256i z = _mm256_maddubs_epi16(x, y);
    z = _mm256_hadd_epi16(z, z);
    z = _mm256_hadd_epi16(z, z);
    z = _mm256_hadd_epi16(z, z);
    int sum = _mm256_extract_epi16(z, 0) + _mm256_extract_epi16(z, 8);
    return scale * sum + bias;
}

void vec_dot_prod_q8(const tensor8q *__restrict__ x, const tensor8q *__restrict__ y, float32 *__restrict__ z)
{
    assert(x->block_size == y->block_size);
    assert(x->dims[1] == y->dims[1]);

#define AVX
#ifdef AVX
    for (uint16_t i = 0; i < x->dims[0] * x->dims[1]; i += x->block_size)
    {
        register partial_sum = .0f;
        float32 scale, bias;
        if (i % x->block_size == 0)
        {
            scale = x->scales[i / x->block_size] * y->scales[i / x->block_size];
            bias = x->bias[i / x->block_size] * y->bias[i / x->block_size];
        }
        for (uint16_t j = 0; j < x->block_size; j += I8_AVX_NREG)
        {
            __m256i axv_x = _mm256_loadu_si256((const __m256i_u *)(x + j));
            __m256i avx_y = _mm256_loadu_si256((const __m256i_u *)(y + j));
            partial_sum += avx_vec_dot_q8(axv_x, avx_y, scale, bias);
        }
    }

#else
    for (uint16_t i = 0; i < x->dims[1]; i++)
    {
        float32 partial_sum = .0f;
        const uint16_t offset = i * x->block_size;
        for (uint16_t j = 0; j < x->block_size; j++)
        {
            partial_sum += x->data[offset + j] * y->data[offset + j];
        }
        *z += partial_sum * x->scales[i] * y->scales[i];
    }
#endif
}