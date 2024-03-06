cmake_minimum_required(VERSION 3.15)

project(retnet LANGUAGES C)

find_package(OpenMP)

include_directories(header)
include_directories(header/macro)
include_directories(header/operation)

add_executable(${PROJECT_NAME} main.c src/operation/vector_op.c)

if(OPENMP_FOUND)
    add_definitions(-DOMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif()