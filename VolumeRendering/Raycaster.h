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
	float tf_offset;

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

	__host__ __device__ void write_color(float4 color, int2 pos, uchar4 buffer[]) {
		buffer[pos.y * view.size_px.x + pos.x] = 
			make_uchar4( map_float_int(color.x, 256), 
						map_float_int(color.y, 256), 
						map_float_int(color.z, 256), 
						map_float_int(color.w, 256));
	}
};

class RaycasterBase {
	public:
		static Raycaster raycaster;
		static void change_tf_offset(float offset, bool reset);
		static void change_ray_step(float step, bool reset);
		static void change_ray_threshold(float threshold, bool reset);
		static void set_volume(Model volume);
};

#endif