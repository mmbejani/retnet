
#include "dtype.h"
#include <stddef.h>

void vec_dot_prod_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ y, float32 *__restrict__ result);
void vec_dot_prod_f16(const size_t n, float16 *__restrict__ x, float16 *__restrict__ y, float16 *__restrict__ result);

void vec_reduce_sum_f32(const size_t n, float32 *__restrict__ x, float32 *__restrict__ result);