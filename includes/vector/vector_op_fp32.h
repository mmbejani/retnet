#include <stddef.h>

#include "dtype.h"

#include "macro/device.h"
#include "macro/constant.h"
#include "macro/initialization.h"

void vec_dot_prod_f32(const size_t n, const float32 *x, const float32 *y, float32 *__restrict__ result);

void vec_reduce_sum_f32(const size_t n, void *__restrict__ x, float32 *__restrict__ result);