#include "tensor.h"


struct GroupNorm
{
    short num_groups, num_channels;
    const float32 eps = 0.00001f;
    bool affine = true;

    tensor *weight;
    tensor *bias;

    tensor forward(tensor *x) noexcept;
};