#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_utils.h"

//const float POS_INF = FLT_MAX, NEG_INF = FLT_MIN;
//CUDART_MAX_NORMAL_F, CUDART_MIN_DENORM_F
#define POS_INF 100
#define NEG_INF -100

struct Volume_model {
	int3 dims;
	float3 min_bound;
	float3 max_bound;
	float ray_step;											

	__host__ __device__ unsigned char sample_data(float3 pos, unsigned char *volume_data) {
		return volume_data[
			map_float_int((pos.z + 1) / 2, dims.z) * dims.x * dims.y +
			map_float_int((pos.y + 1) / 2, dims.y) * dims.x +
			map_float_int((pos.x + 1) / 2, dims.x)
		];
	}

	__host__ __device__ float4 transfer_function(unsigned char sample) {
		return make_float4(sample / 255.0f, sample / 255.0f, sample / 255.0f, sample / 255.0f);
	}

	__host__ __device__ float4 sample_color(float3 point, unsigned char *volume_data) {
	#if 1
		float4 intensity = transfer_function(sample_data(point, volume_data));
		float4 color = {(point.x+1)*0.5f, (point.y+1)*0.5f, (point.z+1)*0.5f, 1};
		return intensity * color;
	#else
		float4 color = {(point.x+1)*0.5f, (point.y+1)*0.5f, (point.z+1)*0.5f, 0.1f};  // prepocitanie polohy bodu <-1;1>(x,y,z) na float vyjadrenie farby <0;1>(r,g,b,1)
		return color;	
		//point = (point + make_float3(1,1,1)) * 0.5f;	
		//return make_float4(point.x, point.y, point.z, 0.1f);	
	#endif
	}

};

int load_model(const char* file_name);

size_t get_volume_data(unsigned char **volume_data);
Volume_model get_model();

#endif