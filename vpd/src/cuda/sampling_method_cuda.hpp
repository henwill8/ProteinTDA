#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>

#define CUDA_CHECK(expr_to_check) do {                  \
    cudaError_t result = expr_to_check;                 \
    if (result != cudaSuccess) {                        \
        fprintf(stderr,                                 \
                "CUDA Runtime Error: %s:%i:%d = %s\n",  \
                __FILE__,                               \
                __LINE__,                               \
                result,                                 \
                cudaGetErrorString(result));            \
        exit (EXIT_FAILURE);                            \
    }                                                   \
} while(0)
