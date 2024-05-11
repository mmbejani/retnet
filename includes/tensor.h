/**
 * This header file contain tensor
 *
 */
#include <stdint.h>
#include "dtype.h"

typedef char bool;
const char true = 1;
const char false = 0;

/**
 * This is a common tensor with precision float32
 * @param data: the pointer to `float32` array that represent a tensor (2d and 3d)
 * @param dims: the dimensions size of the tensor in each axis
 * @param num_dims: it mentions that the tensor is 1d, 2d, or 3d
 * @param iweight: it mentions that the weight is important or not. If it is, it will not
 * enroll in quantization procedure.
 */
typedef struct
{
    float32 *data;
    uint16_t *dims;
    uint16_t num_dims;
    bool iweight;
} tensor;

/**
 * This is an abstract struct of quantized tensor which can be weight of data (input or output of a layer)
 * @param scale: this array with size of `block_size` shows the scale of each block
 * @param bias: this array (most of the time is negative) where should be add to each block for dequantization
 * @param data: the array that represent each block after quantization (warn: all of data should be positive)
 * @param block_size: this variable shows the size of each block that is saved in `data`
 * @param dims: memorize the size of each dims (e.g. for weight [ rows x cols ])
 * @param num_dims: how many dims has this tensor
 */
typedef struct
{
    float32 *scales;
    float32 *bias;
    int8_t *data;
    uint16_t *dims;
    uint16_t block_size;
    uint16_t num_dims;
} tensor8q;

/**
 * see `tensor8q`
 */
typedef struct
{
    float32 *scales;
    uint8_t *data;
    uint16_t *dims;
    uint16_t block_size;
    uint16_t num_dims;
} tensor4q;
