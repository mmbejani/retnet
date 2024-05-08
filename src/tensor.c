#include "tensor.h"

struct tensor3d
{
    // the tensor data where is saved as void ptr because it can take any type (abstraction)
    void *data;

    // the dims is length of [ sequence x batch-size x embedding-size ]
    uint8_t dims[3];

    // scaling of quantization (if exists)
    float32 scale;
};

struct tensor2d {
    // the tensor data where is saved as void ptr because it can take any type (abstraction)
    void *data;

    // the dims is length of [ width x height ]
    uint8_t dims[2];

    // scaling of quantization (if exists)
    float32 scale;   
};