#include "8bit.h"
#include "logging.h"
#include "macro/initialization.h"

void quantize_based_awq(tensor *inputs, tensor *weight, qtensor *quantized_weight, uint8_t block_size)
{
    assert(weight->dims == 2);
    assert(inputs->dims == 3);
    quantized_weight = (qtensor *)malloc(sizeof(qtensor));

    const unsigned short num_blocks = weight->dims[1] / block_size;

    logger(INFO, "AWQ quantization just started");

#ifdef OMP
    logger(INFO, "OpenMP is activated")
    SET_OMP_THREADS();

#pragma omp parallel for
#endif
    for (unsigned short block_counter = 0; block_counter < num_blocks; block_counter)
    {
        
    }
}