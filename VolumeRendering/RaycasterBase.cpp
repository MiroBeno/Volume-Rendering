#include <stdio.h>
#include "Raycaster.h"

Raycaster RaycasterBase::raycaster = {	
							Model(),
							View(),
							(float4 *) malloc(256 * sizeof(float4)),
							0.06f,
							0.95f,
							NULL,
							NULL,
							{8, 8, 8},
							{33, 33, 33},
						};

void RaycasterBase::change_ray_step(float step, bool reset) {
	raycaster.ray_step = CLAMP(reset ? step : raycaster.ray_step + step, 0.001f, 1);
	printf("Ray sampling step: %.4f (approx %.1f sampling points)\n", raycaster.ray_step, 2/raycaster.ray_step);
}

void RaycasterBase::change_ray_threshold(float threshold, bool reset) {
	raycaster.ray_threshold = CLAMP(reset ? threshold : raycaster.ray_threshold + threshold, 0.25f, 1);
	printf("Ray accumulation threshold: %.3f\n", raycaster.ray_threshold);
}

void RaycasterBase::update_esl_volume() {
	unsigned char esl_temp_tf[256];
	for(int x = 0; x < 256; x++) {
		esl_temp_tf[x] = 255;
		for(int y = x; y < 256; y++)
			if (raycaster.transfer_fn[y].w != 0) {
				esl_temp_tf[x] = y;
				break;
			}
	}
	for(unsigned int z = 0; z < raycaster.esl_volume_dims.z; z++)
		for(unsigned int y = 0; y < raycaster.esl_volume_dims.y; y++)
			for(unsigned int x = 0; x < raycaster.esl_volume_dims.x; x++) {
				unsigned int esl_index = z * raycaster.esl_volume_dims.x * raycaster.esl_volume_dims.y +
					y * raycaster.esl_volume_dims.x +
					x;
				raycaster.esl_volume[esl_index] = 
					(esl_temp_tf[raycaster.esl_min_max[esl_index].x] > raycaster.esl_min_max[esl_index].y) ? 1 : 0;
			}
}

void RaycasterBase::set_volume(Model volume) {
	raycaster.volume = volume;
	int max_size = MAXIMUM(volume.dims.x, MAXIMUM(volume.dims.y, volume.dims.z));
	raycaster.ray_step = 2.0f / max_size;  // dlzka najvacsej hrany je 2 
	raycaster.ray_step -= raycaster.ray_step / max_size;

	if (raycaster.esl_min_max != NULL) {
		free(raycaster.esl_min_max);
		free(raycaster.esl_volume);
	}
	ushort3 esl_block_dims = make_ushort3(8, 8, 8);
	ushort3 esl_volume_dims = make_ushort3(33, 33, 33);
#if 1				//fixna velkost bloku
	esl_volume_dims = make_ushort3(volume.dims.x / esl_block_dims.x + 1, volume.dims.y / esl_block_dims.y + 1, volume.dims.z / esl_block_dims.z + 1);
#else				//fixna velkost mriezky
	esl_block_dims.x = volume.dims.x / (esl_volume_dims.x - 1);
	esl_block_dims.y = volume.dims.y / (esl_volume_dims.y - 1);
	esl_block_dims.z = volume.dims.z / (esl_volume_dims.z - 1);
#endif
	unsigned int esl_volume_size = esl_volume_dims.x * esl_volume_dims.y * esl_volume_dims.z;
	uchar2 *esl_min_max = (uchar2*) malloc(esl_volume_size * sizeof(uchar2));
	unsigned char *esl_volume = (unsigned char*) malloc(esl_volume_size * sizeof(unsigned char));
	for(unsigned int i = 0; i < esl_volume_size; i++) {
		esl_min_max[i].x = 255;
		esl_min_max[i].y = 0;
	}
	for(unsigned int z = 0; z < volume.dims.z; z++)
		for(unsigned int y = 0; y < volume.dims.y; y++)
			for(unsigned int x = 0; x < volume.dims.x; x++) {
				unsigned char sample = volume.data[z * volume.dims.x * volume.dims.y + y * volume.dims.x + x];
				unsigned int esl_index = (z / esl_block_dims.z) * esl_volume_dims.x * esl_volume_dims.y + (y / esl_block_dims.y) * esl_volume_dims.x + (x / esl_block_dims.x);
				if (esl_min_max[esl_index].x > sample)
					esl_min_max[esl_index].x = sample;
				if (esl_min_max[esl_index].y < sample)
					esl_min_max[esl_index].y = sample;
			}

	float3 esl_block_size = make_float3(
		2.0f * esl_block_dims.x / volume.dims.x,			//maxbound - minbound
		2.0f * esl_block_dims.y / volume.dims.y,
		2.0f * esl_block_dims.z / volume.dims.z
	);
	raycaster.esl_volume = esl_volume;
	raycaster.esl_min_max = esl_min_max;
	raycaster.esl_block_dims = esl_block_dims;
	raycaster.esl_volume_dims = esl_volume_dims;
	raycaster.esl_block_size = esl_block_size;
	update_esl_volume();
}

