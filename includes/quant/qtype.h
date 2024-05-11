#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "dtype.h"
#include "macro/constant.h"

typedef struct
{
    float scale;
    char *data;
    uint8_t block_size;
} block_8bit;
struct block_8bit_mean;
struct block_8bit_max;
struct block_8bit_gptq;