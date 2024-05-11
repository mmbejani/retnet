#include "tensor.h"

/**
 * This function convert a weight in single precision (float32) to 8bit integer based on AWQ schema.
 * @param inputs: a sample of input where assume that the shape of the tensor is [ sequence_length x batch-size x embedding-dim ].
 * @param weight: this the original weight of the network (float32).
 * @param quantized_weight: the result of quantization will be saved in this variable.
 * @param block_size: the size of block that considering for quatntization.
 */
void quantize_based_awq(tensor *inputs, tensor *weight, tensor8q *quantized_weight, uint8_t block_size);