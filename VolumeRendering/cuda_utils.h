/****************************************/
// CUDA errors handling
/****************************************/

#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <stdlib.h>

#include "Logger.h"

#include "cuda_runtime_api.h"

#include "driver_types.h"
#include "driver_functions.h"
#include "channel_descriptor.h"
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"

extern bool NO_SAFE;

// generic error responses

static inline void __cuda_safe_call(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
		Logger::log("CUDA fatal error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        if (!NO_SAFE) 
			exit(EXIT_FAILURE);
    }
}

static inline void __cuda_safe_check(const char *file, int line ) {
	cudaError_t err = cudaGetLastError();
	__cuda_safe_call(err, file, line);
}

#define cuda_safe_call(err) (__cuda_safe_call( err, __FILE__, __LINE__ ))
#define cuda_safe_check() (__cuda_safe_check(__FILE__, __LINE__ ))

// memory errors

static inline cudaError_t cuda_safe_malloc(cudaError_t err) {
    if (err != cudaSuccess) {
		Logger::log("CUDA memory allocation error: %s\n", cudaGetErrorString(err));
		Logger::log("Data is probably too large for video memory.\n");
    }
	return err;
}

#endif