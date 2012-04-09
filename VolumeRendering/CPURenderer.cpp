#include <string.h>
#include "Renderer.h"		

static float4 *transfer_fn;
static unsigned char *volume_data;

static uchar2 *esl_volume;
static int esl_block_size = 8;
static uint4 esl_size;
//static bool esl_transfer_fn[256][256];
static unsigned char esl_transfer_fn[256];

CPURenderer::CPURenderer(int2 size, float4 *tf, Model volume, unsigned char *d) {
	set_window_buffer(size);
	set_transfer_fn(tf);
	set_volume(volume, d);
}

	inline uchar2 sample_data_esl(float3 pos, Model volume) {
		return esl_volume[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_size) * esl_size.x * esl_size.y +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_size) * esl_size.x  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_size)
		];
	}

	inline float4 sample_color_esl(float3 pos, Model volume) {
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
			float4 color_cur = raycaster.sample_color(volume_data, transfer_fn, pt);
			//float4 color_cur = sample_color_esl(pt, raycaster.model);
			
			if (color_cur.w == 0) {
				uchar2 esl_sample = sample_data_esl(pt, raycaster.model);
				//if (esl_transfer_fn[esl_sample.x][esl_sample.y]) {
				if (esl_transfer_fn[esl_sample.x] > esl_sample.y) {
					int max_size = MAXIMUM(raycaster.model.dims.x, MAXIMUM(raycaster.model.dims.y, raycaster.model.dims.z));
					k +=  (2.0f * esl_block_size) / max_size;
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

void CPURenderer::set_transfer_fn(float4 *tf) {
	transfer_fn = tf;

	for(int x = 0; x < 256; x++) {
		esl_transfer_fn[x] = 255;
		for(int y = x; y < 256; y++)
			if (transfer_fn[y].w != 0) {
				esl_transfer_fn[x] = y;
				break;
			}
	}
/*
	for(int x = 0; x < 256; x++) {
		esl_transfer_fn[x][x] = (transfer_fn[x].w == 0);
		for(int y = x + 1; y < 256; y++)
			esl_transfer_fn[x][y] = (esl_transfer_fn[x][y-1]) ? (transfer_fn[y].w == 0) : false;
	}
*/
}

void CPURenderer::set_volume(Model volume, unsigned char *d) {
	volume_data = d;

	if (esl_volume != NULL)
		free(esl_volume);
	esl_size = make_uint4(volume.dims.x / esl_block_size + 1, volume.dims.y / esl_block_size + 1, volume.dims.z / esl_block_size + 1, 0);
	esl_size.w = esl_size.x * esl_size.y * esl_size.z;
	esl_volume = (uchar2*) malloc(esl_size.w * sizeof(uchar2));
	for(unsigned int i = 0; i < esl_size.w; i++) {
		esl_volume[i].x = 255;
		esl_volume[i].y = 0;
	}
	for(unsigned int z = 0; z < volume.dims.z; z++)
		for(unsigned int y = 0; y < volume.dims.y; y++)
			for(unsigned int x = 0; x < volume.dims.x; x++) {
				unsigned char sample = volume_data[z * volume.dims.x * volume.dims.y + y * volume.dims.x + x];
				unsigned int esl_index = (z / esl_block_size) * esl_size.x * esl_size.y + (y / esl_block_size) * esl_size.x + (x / esl_block_size);
				if (esl_volume[esl_index].x > sample)
					esl_volume[esl_index].x = sample;
				if (esl_volume[esl_index].y < sample)
					esl_volume[esl_index].y = sample;
			}
}

int CPURenderer::render_volume(uchar4 *buffer, Raycaster *r) {
	memset(buffer, 0, r->view.size_px.x * r->view.size_px.y * 4);
	for(int row = 0; row < r->view.size_px.y; row++)
		for(int col = 0; col < r->view.size_px.x; col++)	{
			render_ray(*r, buffer, make_int2(col, row));
		}
	return 0;
}
