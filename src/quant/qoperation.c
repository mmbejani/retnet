#include "tensor.h"
#include "logging.h"
#include "macro/initialization.h"

#include <math.h>

#define ABS(x) x > 0 ? x : -x
/**
 * This is absolute max of an array
 */
inline float32 amax(const size_t n, float32 *data)
{
    float32 max_value = ABS(data[0]);

    for (size_t i = 0; i < n; i++)
    {
        if (ABS(data[i]) > max_value)
            max_value = ABS(data[i]);
    }
    return max_value;
}

/**
 * Creat a fake input based on
 */
tensor *create_fake_input(uint8_t *dims)
{
    tensor *x = (tensor *)malloc(sizeof(tensor));
}

void quantize_based_awq(tensor *inputs, tensor *weight, tensor8q *qweight, uint8_t block_size)
{
    assert(weight->dims == 2);
    assert(inputs->dims == 3);
    const unsigned short row_num_blocks = weight->dims[1] / block_size;
    qweight = (tensor8q *)malloc(sizeof(tensor8q));
    float32 *data = (float32 *)weight->data;

    const float32 max_bits = powf(2, 7) - 1;
    const float32 min_bits = -powf(2, 7);

    // initialize blocks of quantized weights
    qweight->block_size = block_size;
    qweight->block_dims = (uint16_t *)malloc(sizeof(uint16_t) * 2);
    qweight->block_dims[0] = weight->dims[0];
    qweight->block_dims[1] = row_num_blocks;
    qweight->data = malloc(sizeof(block_8bit) * qweight->block_dims[0] * qweight->block_dims[1]);

    for (unsigned short block_counter = 0; block_counter < row_num_blocks; block_counter++)
    {
        for (unsigned short i = 0; i < qweight->block_dims[1]; i++)
        {
            block_8bit *block = (block_8bit *)malloc(sizeof(block_8bit));
            block->block_size = block_size;
            block->scale = .0f;
            block->data = (char *)malloc(sizeof(uint8_t) * block_size);
            qweight->data[block_counter + qweight->block_dims[0] * i] = block;
        }
    }

    logger(INFO, "AWQ quantization just started");

#ifdef OMP
    logger(INFO, "OpenMP is activated")
        SET_OMP_THREADS();

#pragma omp parallel for
#endif
    for (unsigned short block_counter = 0; block_counter < row_num_blocks; block_counter++)
    {
        for (unsigned short row = 0; row < weight->dims[0]; row++)
        {
            float32 max_val = amax(block_size, data + (block_size * block_counter + weight->dims[1] * row));
        }
    }   
}