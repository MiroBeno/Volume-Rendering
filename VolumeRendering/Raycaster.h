#ifndef _RAYCASTER_H_
#define _RAYCASTER_H_

#include "data_utils.h"
#include "Model.h"
#include "View.h"

struct Raycaster {
	Model volume;
	View view;
	float4 *transfer_fn;
	float ray_step;
	float ray_threshold;
	unsigned char *esl_volume;
	uchar2 *esl_min_max;
	ushort3 esl_block_dims;
	ushort3 esl_volume_dims;
	float3 esl_block_size;

	__host__ __device__ float4 sample_color(float4 transfer_fn[], float3 pos) {
		unsigned char sample = volume.sample_data(pos);
		float4 color = transfer_fn[sample];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

	__host__ __device__ bool intersect(float3 pt, float3 dir, float2 *k) {  // mozne odchylky pri vypocte => hodnoty k mimo volume; riesi sa clampovanim vysledku na stenu
		float3 k1 = (-volume.bound - pt) / dir;			// ak je zlozka vektora rovnobezna s osou a teda so stenou kocky (dir == 0), tak
		float3 k2 = (volume.bound - pt) / dir;				// ak lezi bod v romedzi kocky v danej osi je vysledok (-oo; +oo), inak (-oo;-oo) alebo (+oo;+oo) 
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

	__host__ __device__ uchar2 sample_data_esl_min_max(float3 pos) {
		return esl_min_max[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims.z) * esl_volume_dims.x * esl_volume_dims.y +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims.y) * esl_volume_dims.x  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims.x)
		];
	}

	__host__ __device__ float4 sample_color_min_max(float3 pos) {
		unsigned char sample = sample_data_esl_min_max(pos).x;
		float4 color = transfer_fn[sample];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

	__forceinline __host__ __device__  unsigned char sample_data_esl(float3 pos) {
		return esl_volume[
			(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims.z) * esl_volume_dims.x * esl_volume_dims.y +
			(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims.y) * esl_volume_dims.x  +
			(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims.x)
		];
	}

	__forceinline __host__ __device__ bool leap_empty_space(float3 pt, float3 origin, float3 direction, float2 *k) {
		bool skipped = false;
		while (sample_data_esl(pt) == 1) {
			skipped = true;
			int3 index = make_int3(
				map_float_int((pt.x + 1)*0.5f, volume.dims.x) / esl_block_dims.x,
				map_float_int((pt.y + 1)*0.5f, volume.dims.y) / esl_block_dims.y,
				map_float_int((pt.z + 1)*0.5f, volume.dims.z) / esl_block_dims.z
			);
			if (direction.x > 0) index.x++;
			if (direction.y > 0) index.y++;
			if (direction.z > 0) index.z++;
			float3 esl_bound = make_float3(-1, -1, -1);				//min_bound;
			esl_bound = esl_bound + (esl_block_size * index);
			float3 kp = (esl_bound - pt) / direction;
			if (direction.x == 0) kp.x = 100;
			if (direction.y == 0) kp.y = 100;
			if (direction.z == 0) kp.z = 100;	
			float dk = MINIMUM(kp.x, kp.y);
			dk = MINIMUM(dk, kp.z);
			dk = MAXIMUM(dk, 0);
			dk = floor(dk / ray_step) * ray_step;
			k->x += dk;
			k->x += ray_step;  
			pt = origin + (direction * k->x);
			if (k->x > k->y) 
				break;
		}
		if (skipped)
			k->x -= ray_step;
		return skipped;
	}
};

class RaycasterBase {
	public:
		static Raycaster raycaster;
		static void change_ray_step(float step, bool reset);
		static void change_ray_threshold(float threshold, bool reset);
		static void set_volume(Model volume);
		static void update_esl_volume();
};

#endif