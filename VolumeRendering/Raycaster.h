#ifndef _RAYCASTER_H_
#define _RAYCASTER_H_

#include "data_utils.h"
#include "Model.h"
#include "View.h"

#define ESL_MIN_BLOCK_SIZE 8
#define ESL_VOLUME_DIMS 32
#define ESL_VOLUME_SIZE 32*32*32		//32*32*32 bitov 
//#define ESL_VOLUME_SIZE 4096			//32*32*32 bitov / 8
#define TF_SIZE 128
#define TF_RATIO 2						// 256 / TF_SIZE

struct Raycaster {
	Model volume;
	View view;
	float4 *transfer_fn;
	float ray_step;
	float ray_threshold;
	bool esl;
	unsigned char *esl_volume;
	uchar2 *esl_min_max;
	unsigned short esl_block_dims;
	float3 esl_block_size;
	float light_kd;
	float light_ks;

	__host__ __device__ float4 sample_color(float4 transfer_fn[], float3 pos) {
		unsigned char sample = volume.sample_data(pos);
		float4 color = transfer_fn[sample / TF_RATIO];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

	__host__ __device__ bool intersect(float3 pt, float3 dir, float2 *k) {  // mozne odchylky pri vypocte => hodnoty k mimo volume; riesi sa clampovanim vysledku na stenu
		float3 k1 = (volume.min_bound - pt) / dir;			// ak je zlozka vektora rovnobezna s osou a teda so stenou kocky (dir == 0), tak
		float3 k2 = (-volume.min_bound - pt) / dir;				// ak lezi bod v romedzi kocky v danej osi je vysledok (-oo; +oo), inak (-oo;-oo) alebo (+oo;+oo) 
		k->x = MAXIMUM(MAXIMUM(MINIMUM(k1.x, k2.x), MINIMUM(k1.y, k2.y)),MINIMUM(k1.z, k2.z)); 
		k->y = MINIMUM(MINIMUM(MAXIMUM(k1.x, k2.x), MAXIMUM(k1.y, k2.y)),MAXIMUM(k1.z, k2.z));
		k->x = MAXIMUM(k->x, 0);							// ak x < 0 bod vzniku luca je vnutri kocky - zacneme nie vstupnym priesecnikom, ale bodom vzniku (k = 0)
		return ((k->x < k->y) && (k->y > 0));				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci	 
	}	

	__host__ __device__ void write_color(float4 color, short2 pos, uchar4 buffer[]) {
		buffer[pos.y * view.size_px.x + pos.x] = 
			make_uchar4( map_float_int(color.x, 256), 
						map_float_int(color.y, 256), 
						map_float_int(color.z, 256), 
						map_float_int(color.w, 256));
	}

	inline uchar2 sample_data_esl_min_max(float3 pos) {
		return esl_min_max[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims) * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims) * ESL_VOLUME_DIMS  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims)
		];
	}

	inline float4 sample_color_min_max(float3 pos) {
		unsigned char sample = sample_data_esl_min_max(pos).y;
		float4 color = transfer_fn[sample];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

	__forceinline __host__ __device__  bool sample_data_esl(unsigned char esl_volume[], float3 pos) {
		unsigned int index = ((map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims) * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims) * ESL_VOLUME_DIMS +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims));
		//unsigned char sample = esl_volume[index / 8];
		//return ((sample & (1 << (index % 8))) == 0) ? false : true;
		unsigned char sample = esl_volume[index];
		return (sample == 0) ? false : true;
	}

	__forceinline __host__ __device__ void leap_empty_space(float3 pt, float3 dir, float2 *k) {
		ushort3 index = make_ushort3(
			map_float_int((pt.x + 1)*0.5f, volume.dims.x) / esl_block_dims,
			map_float_int((pt.y + 1)*0.5f, volume.dims.y) / esl_block_dims,
			map_float_int((pt.z + 1)*0.5f, volume.dims.z) / esl_block_dims
		);
		if (dir.x > 0) index.x++;
		if (dir.y > 0) index.y++;
		if (dir.z > 0) index.z++;
		float3 kp = (volume.min_bound + (esl_block_size * index) - pt) / dir;
		if (dir.x == 0) kp.x = 100;
		if (dir.y == 0) kp.y = 100;
		if (dir.z == 0) kp.z = 100;	
		float dk = MINIMUM(kp.x, kp.y);
		dk = MINIMUM(dk, kp.z);
		dk = MAXIMUM(dk, 0);
		dk = floor(dk / ray_step) * ray_step;
		k->x += dk;
	}

	__host__ __device__ float3 shade(float3 pt, float3 dir) {
		float3 light_dir = vector_normalize(view.light_pos - pt);
		float sample = volume.sample_data(pt) / 255.0f;
		float sample_l = volume.sample_data(pt + light_dir * 0.01f) / 255.0f;
		float diffuse_light = (sample_l - sample);
		return make_float3(diffuse_light, diffuse_light, diffuse_light) * 0.6f;
	}

};

class RaycasterBase {
	public:
		static Raycaster raycaster;
		static void change_ray_step(float step, bool reset);
		static void change_ray_threshold(float threshold, bool reset);
		static void toggle_esl();
		static void set_volume(Model volume);
		static void update_esl_volume();
};

#endif