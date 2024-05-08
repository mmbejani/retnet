#include <stdint.h>
#include <dtype.h>

typedef enum
{
    INT8,
    INT6,
    INT4,
    INT3,
    INT2,
} type;

/**
 *
 */
typedef struct
{
    float32 *data;
    uint8_t *dims;
    uint8_t num_dims;
} tensor;

/**
 * This is an abstract struct of quantized tensor which can be weight of data (input or output of a layer)
 * @param scale: if the weight has been quantized, this value is
 * @param data: a pointer to `data` without considering data type for abstraction
 */
typedef struct
{
    float32 *scales;
    void *data;
    uint16_t *block_dims;
    uint16_t num_block_dims;
    uint16_t num_blocks;
    uint16_t block_size;
    type dtype;
} qtensor;
