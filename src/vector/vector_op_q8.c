#include <assert.h>
#include <immintrin.h>

#include "vector/vector_op_q8.h"
#include "macro/constant.h"

#if __cplusplus
extern "C"
{
#endif

    inline float avx_vec_dot_q8(const __m256i x, const __m256i y, float scale, float bias)
    {
        __m256i z = _mm256_maddubs_epi16(x, y);
        z = _mm256_hadd_epi16(z, z);
        z = _mm256_hadd_epi16(z, z);
        z = _mm256_hadd_epi16(z, z);
        int sum = _mm256_extract_epi16(z, 0) + _mm256_extract_epi16(z, 8);
        return scale * sum + bias;
    }

    void vec_dot_prod_q8(const block_q8 *__restrict__ x, const block_q8 *__restrict__ y, const uint16_t num_blocks, float32 *__restrict__ z)
    {
        assert(x->block_size == y->block_size);
        assert(x->block_size % I8_AVX_NREG == 0);
#define AVX
#ifdef AVX
        for (uint16_t i = 0; i < num_blocks; i++)
        {
            register float32 block_sum = .0f;
            for (uint16_t j = 0; j < x->block_size; j += I8_AVX_NREG)
            {
                __m256i axv_x = _mm256_loadu_si256((const __m256i_u *)(x[i].data + j));
                __m256i avx_y = _mm256_loadu_si256((const __m256i_u *)(y[j].data + j));
                block_sum += avx_vec_dot_q8(axv_x, avx_y, x->scale * y->scale, x->bias + y->bias);
            }
            *z += block_sum;
        }

#else
    for (uint16_t i = 0; i < num_blocks; i++)
    {
        float32 partial_sum = .0f;
        for (uint16_t j = 0; j < x->block_size; j++)
        {
            partial_sum += x[i].data[j] * y[i].data[j];
        }
        *z += partial_sum * x[i].scale * y[i].scale + (x[i].bias + y[i].bias);
    }
#endif
    }

#if __cplusplus
}
#endif