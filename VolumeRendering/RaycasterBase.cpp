#include <stdlib.h>

#include "RaycasterBase.h"

Raycaster RaycasterBase::raycaster = {	
							Model(),
							View(),
							(float4 *) malloc(TF_SIZE * sizeof(float4)),
							0.06f,
							0.95f,
							true,
							(esl_type*) malloc(ESL_VOLUME_SIZE * sizeof(esl_type)),
							(uchar2*) malloc(ESL_VOLUME_DIMS * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS * sizeof(uchar2)),
							ESL_MIN_BLOCK_SIZE,
							{0, 0, 0},
							0.6f
						};

float4 RaycasterBase::base_transfer_fn[TF_SIZE];
float2 RaycasterBase::ray_step_limits;

void RaycasterBase::change_ray_step(float step, bool reset) {
	raycaster.ray_step = CLAMP(reset ? step : raycaster.ray_step + step, ray_step_limits.x, ray_step_limits.y);
	Logger::log("Ray sampling step: %.4f (approx %.1f sampling points)\n", raycaster.ray_step, 2/raycaster.ray_step);
}

void RaycasterBase::change_ray_threshold(float threshold, bool reset) {
	raycaster.ray_threshold = CLAMP(reset ? threshold : raycaster.ray_threshold + threshold, 0.5f, 1);
	Logger::log("Ray accumulation threshold: %.3f\n", raycaster.ray_threshold);
}

void RaycasterBase::change_light_intensity(float intensity, bool reset) {
	raycaster.light_kd = CLAMP(reset ? intensity : raycaster.light_kd + intensity, 0, 2);
	Logger::log("Light intensity: %.3f\n", raycaster.light_kd);
}

void RaycasterBase::toggle_esl() {
	raycaster.esl = !raycaster.esl;
	Logger::log("Empty space leaping: %s\n", raycaster.esl ? "on" : "off");
}

void RaycasterBase::update_transfer_fn() {
	for (int i = 0; i < TF_SIZE; i++) {					// pre-multiplikacia tf
		raycaster.transfer_fn[i].x = base_transfer_fn[i].x * base_transfer_fn[i].w;
		raycaster.transfer_fn[i].y = base_transfer_fn[i].y * base_transfer_fn[i].w;
		raycaster.transfer_fn[i].z = base_transfer_fn[i].z * base_transfer_fn[i].w;
		raycaster.transfer_fn[i].w = base_transfer_fn[i].w;
	}
	unsigned short esl_temp_tf[TF_SIZE];		// pomocne pole rozsahov indexov transfer_fn, ktore maju 0 opacity
	int x, y;
	for(x = 0; x < TF_SIZE; x++) {				
		for(y = x; y < TF_SIZE; y++) {
			if (raycaster.transfer_fn[y].w != 0)
				break;
		}
		esl_temp_tf[x] = y;
	}
	for(unsigned int i = 0; i < ESL_VOLUME_DIMS * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS; i++) {
		if (esl_temp_tf[raycaster.esl_min_max[i].x / TF_RATIO] > raycaster.esl_min_max[i].y / TF_RATIO)
			raycaster.esl_volume[i/32] |= 1 << (i % 32);
		else 
			raycaster.esl_volume[i/32]  &= ~(1 << (i % 32));
	}
	/*for(unsigned int i = 0; i < ESL_VOLUME_DIMS * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS; i++) {
		if (esl_temp_tf[raycaster.esl_min_max[i].x / TF_RATIO] > raycaster.esl_min_max[i].y / TF_RATIO)
			raycaster.esl_volume[i] = 1;
		else 
			raycaster.esl_volume[i] = 0;
	}*/
}

void RaycasterBase::reset_transfer_fn() {
	for (int i =0; i < TF_SIZE; i++) {
		RaycasterBase::base_transfer_fn[i] = make_float4(i <= TF_SIZE/3 ? (i*3)/(float)(TF_SIZE) : 0.0f, 
										(i > TF_SIZE/3) && (i <= TF_SIZE/3*2) ? ((i-TF_SIZE/3)*3)/(float)(TF_SIZE) : 0.0f, 
										i > TF_SIZE/3*2 ? ((i-TF_SIZE/3*2)*3)/(float)(TF_SIZE) : 0.0f, 
										i > ((255.0f * 0.1f)/TF_RATIO) ? i/(float)(TF_SIZE) : 0.0f);
	}
	//RaycasterBase::base_transfer_fn[30] = make_float4(1,1,1,1);
	/*for (int i =0; i < TF_SIZE; i++) {
		RaycasterBase::base_transfer_fn[i] = make_float4(0.23f, 0.23f, 0.0f, i/(float)TF_SIZE);
	}*/
	update_transfer_fn();
}

void RaycasterBase::reset_ray_step() {
	int max_dim = MAXIMUM(raycaster.volume.dims.x, MAXIMUM(raycaster.volume.dims.y, raycaster.volume.dims.z));
	raycaster.ray_step = 2.0f / max_dim;		// dlzka najvacsej hrany je 2 
	raycaster.ray_step -= raycaster.ray_step / max_dim;
	ray_step_limits.x = raycaster.ray_step / 3;
	ray_step_limits.y = raycaster.ray_step * 1.666f;
}

void RaycasterBase::set_volume(Model volume) {
	raycaster.volume = volume;

	int max_dim = MAXIMUM(volume.dims.x, MAXIMUM(volume.dims.y, volume.dims.z));
	raycaster.esl_block_dims = (max_dim + ESL_VOLUME_DIMS - 1) / ESL_VOLUME_DIMS;
	raycaster.esl_block_dims = MAXIMUM(ESL_MIN_BLOCK_SIZE, raycaster.esl_block_dims);

	for(unsigned int i = 0; i < ESL_VOLUME_DIMS * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS; i++) {
		raycaster.esl_min_max[i].x = 255;
		raycaster.esl_min_max[i].y = 0;
	}
	for(unsigned int z = 0; z < volume.dims.z; z++)
		for(unsigned int y = 0; y < volume.dims.y; y++)
			for(unsigned int x = 0; x < volume.dims.x; x++) {
				unsigned char sample = volume.data[z * volume.dims.x * volume.dims.y + y * volume.dims.x + x];
				unsigned int esl_index = 
					(z / raycaster.esl_block_dims) * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS + 
					(y / raycaster.esl_block_dims) * ESL_VOLUME_DIMS + 
					(x / raycaster.esl_block_dims);
				if (raycaster.esl_min_max[esl_index].x > sample)
					raycaster.esl_min_max[esl_index].x = sample;
				if (raycaster.esl_min_max[esl_index].y < sample)
					raycaster.esl_min_max[esl_index].y = sample;
			}
	raycaster.esl_block_size = make_float3(
		2.0f * raycaster.esl_block_dims / volume.dims.x, //maxbound - minbound
		2.0f * raycaster.esl_block_dims / volume.dims.y,
		2.0f * raycaster.esl_block_dims / volume.dims.z
	);
	update_transfer_fn();
	reset_ray_step();
}

void RaycasterBase::set_view(View view) {
	raycaster.view = view;
}
