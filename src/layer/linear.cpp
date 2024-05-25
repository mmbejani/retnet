#include <stdlib.h>

#include "tensor.h"
#include "linear.hpp"

tensor *Linear::call(tensor *x) const
{
    // allocating the output
    tensor *y = (tensor *)malloc(sizeof(tensor));

    return nullptr;
}

tensor8q *Linear8Q::call(tensor8q *x) const
{
    tensor8q *y = (tensor8q *)malloc(sizeof(tensor8q));
    return nullptr;
}
