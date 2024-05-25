#include "module.hpp"

struct Linear : Module
{
    // The pointer of parameters
    tensor *weight, *bias;

    tensor* call(tensor *x) const override;    
};

struct Linear8Q : Module8Q {
    tensor8q *weight, *bias;

    tensor8q* call(tensor8q *x) const override;
};