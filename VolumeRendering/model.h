#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_utils.h"

//const float POS_INF = FLT_MAX, NEG_INF = FLT_MIN;
//CUDART_MAX_NORMAL_F, CUDART_MIN_DENORM_F
#define POS_INF 10000
#define NEG_INF -10000

struct Volume_model {
	unsigned char *data;
	unsigned int size;
	cudaExtent dims;		
	float3 min_bound;
	float3 max_bound;
	float ray_step;											

	__host__ __device__ float sample_data(float3 pos) {
		unsigned char sample = data[
			map_float_int((pos.z + 1)*0.5f, dims.depth) * dims.width * dims.height +
			map_float_int((pos.y + 1)*0.5f, dims.height) * dims.width +
			map_float_int((pos.x + 1)*0.5f, dims.width)
		];
		return (sample / 255.0f); 
	}

	__host__ __device__ float4 transfer_function(float sample, float3 pos) {
		float4 intensity = {sample, sample, sample, sample};
		float4 color = {(pos.x+1)*0.5f, (pos.y+1)*0.5f, (pos.z+1)*0.5f, 1};
		return intensity * color;	
	}

	__host__ __device__ float4 sample_color(float3 pos) {
	#if 1
		return transfer_function(sample_data(pos), pos);
	#else
		float4 color = {(pos.x+1)*0.5f, (pos.y+1)*0.5f, (pos.z+1)*0.5f, 0.1f};  // prepocitanie polohy bodu <-1;1>(x,y,z) na float vyjadrenie farby <0;1>(r,g,b,1)
		return color;	
		//point = (pos + make_float3(1,1,1)) * 0.5f;	
		//return make_float4(pos.x, pos.y, pos.z, 0.1f);	
	#endif
	}

};

int load_model(const char* file_name);

Volume_model get_model();

#endif