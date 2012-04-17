#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_utils.h"
#include "ddsbase.h"

struct Model {
	unsigned char *data;
	unsigned int size;
	ushort3 dims;		
	float3 min_bound;

	__host__ __device__ unsigned char sample_data(float3 pos) {
		return data[
			map_float_int((pos.z + 1)*0.5f, dims.z) * dims.x * dims.y +
			map_float_int((pos.y + 1)*0.5f, dims.y) * dims.x +
			map_float_int((pos.x + 1)*0.5f, dims.x)
		];
	}
};

class ModelBase {
	public:
		static Model volume;
		static float histogram[256];
		static int load_model(const char* file_name);
	private:
		static void compute_histogram();
};

#endif