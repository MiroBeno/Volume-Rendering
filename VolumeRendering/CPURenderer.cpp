#include <string.h>
#include "Renderer.h"		

static uchar2 *esl_volume;
static unsigned char *esl_volume_bool;
static int esl_block_size = 8;
static uint4 esl_size;

CPURenderer::CPURenderer(Raycaster r) {
	set_window_buffer(r.view);
	set_volume(r.volume);
	set_transfer_fn(r);
}

	inline uchar2 sample_data_esl(float3 pos, Model volume) {
		return esl_volume[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_size) * esl_size.x * esl_size.y +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_size) * esl_size.x  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_size)
		];
	}

	inline unsigned char sample_data_esl_bool(float3 pos, Model volume) {
		return esl_volume_bool[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_size) * esl_size.x * esl_size.y +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_size) * esl_size.x  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_size)
		];
	}

	inline float4 sample_color_esl(float3 pos, Model volume, float4 transfer_fn[]) {
		unsigned char sample = sample_data_esl(pos, volume).y;
		float4 color = transfer_fn[sample];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

inline void render_ray(Raycaster raycaster, uchar4 buffer[], int2 pos) {
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (raycaster.intersect(origin, direction, &k_range))	{	
		float4 color_acc = {0, 0, 0, 0};
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(raycaster.transfer_fn, pt);
			//float4 color_cur = sample_color_esl(pt, raycaster.volume, raycaster.transfer_fn);
			if (color_cur.w == 0) {
				if (sample_data_esl_bool(pt, raycaster.volume) == 1) {
					int max_size = MAXIMUM(raycaster.volume.dims.x, MAXIMUM(raycaster.volume.dims.y, raycaster.volume.dims.z));
					k +=  (2.0f * esl_block_size) / max_size;
					//color_cur.x = 1; color_cur.y = 1; color_cur.z = 0; color_cur.w = 0.02f;
				}
				continue;
			}
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
		raycaster.write_color(color_acc, pos, buffer);
	}
}

void CPURenderer::set_transfer_fn(Raycaster r) {
	//minmax:
	unsigned char esl_min_max[256];
	for(int x = 0; x < 256; x++) {
		esl_min_max[x] = 255;
		for(int y = x; y < 256; y++)
			if (r.transfer_fn[y].w != 0) {
				esl_min_max[x] = y;
				break;
			}
	}
	//bool:
	for(unsigned int z = 0; z < esl_size.z; z++)
		for(unsigned int y = 0; y < esl_size.y; y++)
			for(unsigned int x = 0; x < esl_size.x; x++) {
				unsigned int esl_index = z * esl_size.x * esl_size.y + y * esl_size.x + x;
				esl_volume_bool[esl_index] = (esl_min_max[esl_volume[esl_index].x] > esl_volume[esl_index].y) ? 1 : 0;
			}
}

void CPURenderer::set_volume(Model volume) {

	if (esl_volume != NULL) {
		free(esl_volume);
		free(esl_volume_bool);
	}
	esl_size = make_uint4(volume.dims.x / esl_block_size + 1, volume.dims.y / esl_block_size + 1, volume.dims.z / esl_block_size + 1, 0);
	esl_size.w = esl_size.x * esl_size.y * esl_size.z;
	esl_volume = (uchar2*) malloc(esl_size.w * sizeof(uchar2));
	esl_volume_bool = (unsigned char*) malloc(esl_size.w * sizeof(unsigned char));
	for(unsigned int i = 0; i < esl_size.w; i++) {
		esl_volume[i].x = 255;
		esl_volume[i].y = 0;
	}
	for(unsigned int z = 0; z < volume.dims.z; z++)
		for(unsigned int y = 0; y < volume.dims.y; y++)
			for(unsigned int x = 0; x < volume.dims.x; x++) {
				unsigned char sample = volume.data[z * volume.dims.x * volume.dims.y + y * volume.dims.x + x];
				unsigned int esl_index = (z / esl_block_size) * esl_size.x * esl_size.y + (y / esl_block_size) * esl_size.x + (x / esl_block_size);
				if (esl_volume[esl_index].x > sample)
					esl_volume[esl_index].x = sample;
				if (esl_volume[esl_index].y < sample)
					esl_volume[esl_index].y = sample;
			}
}

int CPURenderer::render_volume(uchar4 *buffer, Raycaster r) {
	memset(buffer, 0, r.view.size_px.x * r.view.size_px.y * 4);
	for(int row = 0; row < r.view.size_px.y; row++)
		for(int col = 0; col < r.view.size_px.x; col++)	{
			render_ray(r, buffer, make_int2(col, row));
		}
	return 0;
}
