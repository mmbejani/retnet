#include "dtype.h"
#include "vector/vector_op_fp16.h"

float32 *convert_fp16_to_fp32(const size_t n, float16 *__restrict__ x)
{
    return NULL;
}

float16 *convert_fp32_to_fp16(const size_t n, float32 *__restrict__ x)
{
    return NULL;
}

void vec_dot_prod_f16(const size_t n, float16 *__restrict__ x, float16 *__restrict__ y, float16 *__restrict__ result)
{
    float32 *x_fp32 = convert_fp16_to_fp32(n, x);
    float32 *y_fp32 = convert_fp16_to_fp32(n, y);

    vec_dot_prod_f32(n, x, y, result);
}