#pragma once
#include "tensor.h"

struct Module
{
    virtual tensor *call(tensor *x) const;
};

struct Module8Q
{
    virtual tensor8q *call(tensor8q *x) const;
};
