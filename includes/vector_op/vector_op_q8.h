#include <stddef.h>

#include "tensor.h"

/**
 * This function is consider that you give it row of a tensor (in shape of blocks) then, it will compute the
 * inner product of two row
 * @param x: the first blocks in 8-bit relam
 * @param y: the second blocks in 8-bit realm
 * @param num_blocks: number of blocks `x` and `y`
 * @param z: a float number that the results is saved in this variable
 */
void vec_dot_prod_q8(const block_q8 *__restrict__ x, const block_q8 *__restrict__ y, const uint16_t num_blocks, float32 *__restrict__ z);
