#ifdef CUDA
    #define CU_GLOBAL __global__
    #define CU_DEVICE __device__
#else
    #define CU_GLOBAL
    #define CU_DEVICE
#endif