#include "dtype.h"

struct block_8bit_awq;
struct block_8bit_mean;
struct block_8bit_max;
struct block_8bit_gptq;

void quantize_based_awq(struct tensor3d *inputs, struct tensor2d *weight, struct tensor2d *quantized_weight, uint8_t block_size);