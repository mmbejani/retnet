#include <stdlib.h>
#include "mkl.h"

#include "tensor.h"
#include "linear.hpp"

tensor *Linear::call(tensor *x) const
{
    // [#over-optimization!]
    // TODO: every time that the linear module call it will create a new intermediate tensor.
    // By creating a single intermediate tensor may decrease the allocation time. However,
    // I didn't see that the allocation decrease the performance. You can say the same to 
    //other version of `Linear`
    tensor *y = (tensor *)malloc(sizeof(tensor));
    #ifdef MKL
    #else

    #endif

    return nullptr;
}

tensor8q *Linear8Q::call(tensor8q *x) const
{
    tensor8q *y = (tensor8q *)malloc(sizeof(tensor8q));
    return nullptr;
}
