#include <stdio.h>
#include "Raycaster.h"

Raycaster RaycasterBase::raycaster = {	
							Model(),
							View(),
							(float4 *) malloc(TF_SIZE * sizeof(float4)),
							0.06f,
							0.95f,
							false,
							(unsigned char*) malloc(ESL_VOLUME_SIZE * sizeof(unsigned char)),
							(uchar2*) malloc(ESL_VOLUME_SIZE * 8 * sizeof(uchar2)),
							ESL_MIN_BLOCK_SIZE,
							{0, 0, 0},
							0.6f,
							0.2f
						};

float4 RaycasterBase::base_transfer_fn[TF_SIZE];

void RaycasterBase::change_ray_step(float step, bool reset) {
	raycaster.ray_step = CLAMP(reset ? step : raycaster.ray_step + step, 0.001f, 1);
	printf("Ray sampling step: %.4f (approx %.1f sampling points)\n", raycaster.ray_step, 2/raycaster.ray_step);
}

void RaycasterBase::change_ray_threshold(float threshold, bool reset) {
	raycaster.ray_threshold = CLAMP(reset ? threshold : raycaster.ray_threshold + threshold, 0.25f, 1);
	printf("Ray accumulation threshold: %.3f\n", raycaster.ray_threshold);
}

void RaycasterBase::toggle_esl() {
	raycaster.esl = !raycaster.esl;
	printf("Empty space leaping: %s\n", raycaster.esl ? "on" : "off");
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
	/*for(unsigned int i = 0; i < ESL_VOLUME_SIZE * 8; i++) {
		if (esl_temp_tf[raycaster.esl_min_max[i].x / TF_RATIO] > raycaster.esl_min_max[i].y / TF_RATIO)
			raycaster.esl_volume[i/8] |= 1 << (i % 8);
		else 
			raycaster.esl_volume[i/8]  &= ~(1 << (i % 8));
	}*/
	for(unsigned int i = 0; i < ESL_VOLUME_SIZE; i++) {
		if (esl_temp_tf[raycaster.esl_min_max[i].x / TF_RATIO] > raycaster.esl_min_max[i].y / TF_RATIO)
			raycaster.esl_volume[i] = 1;
		else 
			raycaster.esl_volume[i] = 0;
	}
}

void RaycasterBase::set_volume(Model volume) {
	raycaster.volume = volume;
	int max_size = MAXIMUM(volume.dims.x, MAXIMUM(volume.dims.y, volume.dims.z));
	raycaster.ray_step = 2.0f / max_size;		// dlzka najvacsej hrany je 2 
	raycaster.ray_step -= raycaster.ray_step / max_size;

	int max_dim = MAXIMUM(volume.dims.x, MAXIMUM(volume.dims.y, volume.dims.z));
	raycaster.esl_block_dims = (max_dim + ESL_VOLUME_DIMS - 1) / ESL_VOLUME_DIMS;
	raycaster.esl_block_dims = MAXIMUM(ESL_MIN_BLOCK_SIZE, raycaster.esl_block_dims);

	for(unsigned int i = 0; i < ESL_VOLUME_SIZE * 8; i++) {
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
}

