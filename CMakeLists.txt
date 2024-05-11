cmake_minimum_required(VERSION 3.15)

project(retnet LANGUAGES C )

include(cmake/function.cmake)
include(cmake/intrin.cmake)

set(CUDA 0 CACHE STRING "If you set this flag to '1', the code base is compiled in CUDA, otherwise it will be compiled for CPU")

if (${CUDA})
    message(STATUS "The code is going to compile for CUDA")
    
else()
    set(USE_OMP 0 CACHE STRING "If you set this flag '1', in some part of code, OpenMP is used to make performance better (default value is '0')")

    if(USE_OMP)
        message(STATUS "Using OpenMP in some situations to improve performance")
        find_package(OpenMP REQUIRED)
    endif()

    add_header_files()
    check_if_sse_available()
    check_if_avx_available()


    if(COMPILER_SUPPORTS_AVX)
        message(STATUS "Adding AVX flag")
        add_definitions(-DAVX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mfma -mavx512f")
    endif()

    add_executable(${PROJECT_NAME} main.c src/operation/vector_op_fp16.c)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)

    if(OPENMP_FOUND)
        message(STATUS "Adding OpenMP flag")
        add_definitions(-DOMP)
        target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
    endif()
endif()
