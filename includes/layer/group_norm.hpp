
#include "module.hpp"

struct GroupNorm : Module
{
    // Network parameters
    short num_groups, num_channels;
    const float32 eps = 0.00001f;
    bool affine = true;

    tensor *weight;
    tensor *bias;

    // Network meta data for computation graph

    tensor *call(tensor *x) const override;
};