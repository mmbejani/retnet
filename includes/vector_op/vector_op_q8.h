#include <stddef.h>

#include "tensor.h"

/**
 * This function is consider that you give it row of a tensor (in shape of blocks) then, it will compute the
 * inner product of two row
 * @param x: the first vector in 8-bit relam
 * @param y: the second vector in 8-bit realm
 * @param z: a float number that the results is saved in this variable
*/
void vec_dot_prod_q8(const tensor8q *__restrict__ x, const tensor8q *__restrict__ y, float32 *__restrict__ z);
