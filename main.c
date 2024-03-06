#include "operation/vector_op.h"
#include <stdio.h>

int main(){
    float32 a[1024];
    float32 b[1024];

    for (size_t i = 0; i < 1024; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }    

    float32 c = 0;

    vec_dot_prod_f32(1024UL, a, b, &c);

    printf("the final value is %f", c);
    return 0;
}