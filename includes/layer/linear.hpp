#include "tensor.h"

struct Linear
{
    tensor *weight, *bias;

    tensor forward(tensor *x) noexcept;
};