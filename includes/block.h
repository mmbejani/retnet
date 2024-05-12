#pragma once

#include <stdint.h>
#include "dtype.h"

typedef struct {
    int8_t *data;
    uint16_t block_size;
    float32 scale;
    float32 bias;
} block_q8;