#include "dtype.h"
#include <stdlib.h>

#include "macro/device.h"
#include "macro/constant.h"
#include "macro/initialization.h"

#include "vector_op/vector_op_fp32.h"

float32 *convert_fp16_to_fp32(const size_t n, float16 *__restrict__ x);
float16 *convert_fp32_to_fp16(const size_t n, float32 *__restrict__ x);

void vec_dot_prod_f16(const size_t n, float16 *__restrict__ x, float16 *__restrict__ y, float16 *__restrict__ result);