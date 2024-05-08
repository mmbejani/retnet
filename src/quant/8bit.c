#include "quant/8bit.h"
#include "tensor.h"
#include "macro/constant.h"

struct block_8bit_awq {
    const uint8_t block_size;
    float scale;
};

void quantize_based_awq(struct tensor3d *inputs, struct tensor2d *weight, struct tensor2d *quantized_weight, uint8_t block_size) {
    
}