#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_utils.h"

struct Volume_model {
	unsigned char *data;
	unsigned int size;
	int3 dims;		
	float3 min_bound;
	float3 max_bound;

	__host__ __device__ float sample_data(float3 pos) {
		unsigned char sample = data[
			map_float_int((pos.z + 1)*0.5f, dims.z) * dims.x * dims.y +
			map_float_int((pos.y + 1)*0.5f, dims.y) * dims.x +
			map_float_int((pos.x + 1)*0.5f, dims.x)
		];
		return (sample / 255.0f); 
	}
};

int load_model(const char* file_name);

Volume_model get_model();

#endif