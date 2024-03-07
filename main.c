#include "operation/vector_op.h"
#include <stdio.h>

int main()
{
    float32 a[100];
    float32 b[100];

    for (size_t i = 0; i < 100; i++)
    {
        a[i] = 1.0f;
        b[i] = 1.0f;
    }

    float32 c = 0;

    vec_dot_prod_f32(100UL, a, b, &c);

    printf("the final value is %f", c);
    return 0;
}