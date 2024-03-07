cmake_minimum_required(VERSION 3.15)

project(retnet LANGUAGES C)

find_package(OpenMP REQUIRED)

include_directories(header)
include_directories(header/macro)
include_directories(header/operation)

add_executable(${PROJECT_NAME} main.c src/operation/vector_op.c)
target_compile_options(${PROJECT_NAME} PRIVATE -O3)

if(OPENMP_FOUND)
    add_definitions(-DOMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif()