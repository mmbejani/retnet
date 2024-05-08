#include "8bit.h"
#include "logging.h"
#include "macro/initialization.h"

#include <math.h>

inline float32 max(const unsigned short n, float32 *data)
{
    float32 max_value = data[0];

    for (unsigned short i = 0; i < n; i++)
    {
        if (data[i] > max_value)
            max_value = data[i];
    }
    return max_value;
}

inline float32 min(const unsigned short n, float32 *data)
{
    float32 min_value = data[0];

    for (unsigned short i = 0; i < n; i++)
    {
        if (data[i] < min_value)
            min_value = data[i];
    }
    return min_value;
}

void quantize_based_awq(tensor *inputs, tensor *weight, qtensor *quantized_weight, uint8_t block_size)
{
    assert(weight->dims == 2);
    assert(inputs->dims == 3);
    quantized_weight = (qtensor *)malloc(sizeof(qtensor));
    float32 *data = (float32 *)weight->data;

    const unsigned short num_blocks = weight->dims[1] / block_size;
    const float32 max_bits = powf(2, 7) - 1;
    const float32 min_bits = -powf(2, 7);

    logger(INFO, "AWQ quantization just started");

#ifdef OMP
    logger(INFO, "OpenMP is activated")
        SET_OMP_THREADS();

#pragma omp parallel for
#endif
    for (unsigned short block_counter = 0; block_counter < num_blocks; block_counter++)
    {
        for (unsigned short row = 0; row < weight->dims[0]; row++)
        {
            float32 max_val = max(block_size, data + (block_size * block_counter + weight->dims[1] * row));
            float32 min_val = min(block_size, data + (block_size * block_counter + weight->dims[1] * row));
        }
    }
}