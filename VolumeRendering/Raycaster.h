#ifndef _RAYCASTER_H_
#define _RAYCASTER_H_

#include "data_utils.h"
#include "model.h"
#include "projection.h"

//const float POS_INF = FLT_MAX, NEG_INF = FLT_MIN;
//CUDART_MAX_NORMAL_F, CUDART_MIN_NORMAL_F
#define POS_INF 10000
#define NEG_INF -10000

struct Raycaster {
	Volume_model model;
	View view;
	float ray_step;
	float ray_threshold;
	float tf_offset;
	float3 bg_color;

	__host__ __device__ float4 sample_color(unsigned char volume_data[], float4 transfer_fn[], float3 pos) {
		unsigned char sample = model.sample_data(volume_data, pos);
		float4 color = transfer_fn[sample];  // (int)sample
		color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
		color.y *= color.w;
		color.z *= color.w;
		return color;
	}

	__host__ __device__ bool intersect(float3 pt, float3 dir, float2 *k) {  // mozne odchylky pri vypocte => hodnoty k mimo volume; riesi sa clampovanim vysledku na stenu
		float3 k1 = (model.min_bound - pt) / dir;			// ak je zlozka vektora rovnobezna s osou a teda so stenou kocky (dir == 0), tak
		float3 k2 = (model.max_bound - pt) / dir;				// ak lezi bod v romedzi kocky v danej osi je vysledok (-oo; +oo), inak (-oo;-oo) alebo (+oo;+oo) 
		k->x = MAXIMUM(MAXIMUM(MINIMUM(k1.x, k2.x), MINIMUM(k1.y, k2.y)),MINIMUM(k1.z, k2.z)); 
		k->y = MINIMUM(MINIMUM(MAXIMUM(k1.x, k2.x), MAXIMUM(k1.y, k2.y)),MAXIMUM(k1.z, k2.z));
		k->x = MAXIMUM(k->x, 0);							// ak x < 0 bod vzniku luca je vnutri kocky - zacneme nie vstupnym priesecnikom, ale bodom vzniku (k = 0)
		return ((k->x < k->y) && (k->y > 0));				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci	 
	}	

	__host__ __device__ void write_color(float4 color, int2 pos, uchar4 buffer[]) {
		if (color.w <= ray_threshold)
			color = color + (bg_color * (1 - color.w));
		buffer[pos.y * view.size_px.x + pos.x] = 
			make_uchar4( map_float_int(color.x,256), 
						map_float_int(color.y,256), 
						map_float_int(color.z,256), 
						255);
	}
};

void change_tf_offset(float offset, bool reset);
void change_ray_step(float step, bool reset);
void change_ray_threshold(float threshold, bool reset);

void set_raycaster_model(Volume_model model);
void set_raycaster_view(View view);

Raycaster *get_raycaster();
float4 *get_transfer_fn();

#endif