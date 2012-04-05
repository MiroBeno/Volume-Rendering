#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_utils.h"
#include "ddsbase.h"

struct Model {
	unsigned int size;
	uint3 dims;		
	float3 bound;

	__host__ __device__ unsigned char sample_data(unsigned char volume_data[], float3 pos) {
		return volume_data[
			map_float_int((pos.z + 1)*0.5f, dims.z) * dims.x * dims.y +
			map_float_int((pos.y + 1)*0.5f, dims.y) * dims.x +
			map_float_int((pos.x + 1)*0.5f, dims.x)
		];
	}
};

class ModelBase {
	public:
		static Model volume;
		static unsigned char *data;
		static float histogram[256];
		static int load_model(const char* file_name);
};

#endif