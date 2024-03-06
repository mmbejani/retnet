#define SET_OMP_THREADS()                     \
    do {                                      \
        const int num_threads = get_nprocs(); \
        omp_set_num_threads(num_threads);     \
    } while(0)
