#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime_api.h"

#include "driver_types.h"
#include "driver_functions.h"
#include "channel_descriptor.h"
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"

static void __cuda_safe_call(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
		fprintf(stderr, "CUDA fatal error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

static void __cuda_safe_check(const char *file, int line ) {
	cudaError_t err = cudaGetLastError();
	__cuda_safe_call(err, file, line);
}

#define cuda_safe_call(err) (__cuda_safe_call( err, __FILE__, __LINE__ ))
#define cuda_safe_check() (__cuda_safe_check(__FILE__, __LINE__ ))

#endif