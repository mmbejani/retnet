#pragma once
#include "tensor.h"

struct Module
{
    virtual tensor *call(tensor *x) const;
};
