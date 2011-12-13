#ifndef _RAYCASTER_H_
#define _RAYCASTER_H_

#include "data_utils.h"
#include "model.h"

//const float POS_INF = FLT_MAX, NEG_INF = FLT_MIN;
//CUDART_MAX_NORMAL_F, CUDART_MIN_NORMAL_F
#define POS_INF 10000
#define NEG_INF -10000

struct Raycaster {
	Volume_model model;
	float ray_step;
	float ray_thershold;
	float tf_offset;

	__host__ __device__ float4 transfer_function(float sample, float3 pos) {
		float4 intensity = {sample, sample, sample, sample >= tf_offset ? sample : 0};
		float4 color = {(pos.x+1)*0.5f, (pos.y+1)*0.5f, (pos.z+1)*0.5f, 1};
		return intensity * color;	
	}

	__host__ __device__ float4 sample_color(float3 pos) {
	#if 1
		return transfer_function(model.sample_data(pos), pos);
	#else
		float4 color = {(pos.x+1)*0.5f, (pos.y+1)*0.5f, (pos.z+1)*0.5f, 0.1f};  // prepocitanie polohy bodu <-1;1>(x,y,z) na float vyjadrenie farby <0;1>(r,g,b,1)
		return color;	
	#endif
	}

	__host__ __device__ float2 intersect_1D(float pt, float dir, float min_bound, float max_bound) {
		if (dir == 0) {											// ak je zlozka vektora rovnobezna so stenou kocky
			if ((pt < min_bound) || (pt > max_bound))			// ak nelezi bod v romedzi kocky v danej osi
				return make_float2(POS_INF, NEG_INF);			// interval bude nulovy
			else
				return make_float2(NEG_INF, POS_INF);			// inak interval bude nekonecny
		}
		float k1 = (min_bound - pt) / dir;
		float k2 = (max_bound - pt) / dir;
		return k1 <= k2 ? make_float2(k1, k2) : make_float2(k2, k1); // skontroluj opacny vektor
	}

	__host__ __device__ float2 intersect(float3 pt, float3 dir) {
		float2 xRange = intersect_1D(pt.x, dir.x, model.min_bound.x, model.max_bound.x);
		float2 yRange = intersect_1D(pt.y, dir.y, model.min_bound.y, model.max_bound.y);
		float2 zRange = intersect_1D(pt.z, dir.z, model.min_bound.z, model.max_bound.z);
		float k1 = xRange.x, k2 = xRange.y;
		if (yRange.x > k1) k1 = yRange.x;
		if (zRange.x > k1) k1 = zRange.x;
		if (yRange.y < k2) k2 = yRange.y;
		if (zRange.y < k2) k2 = zRange.y;
		return make_float2(k1, k2);					// pri vypocte k mozu vzniknut artefakty, a hodnoty mozu byt mimo volume, mozno riesit k +-= 0.00001f; alebo clampovanim vysledku na stenu
	}

};

void set_tf_offset(float offset);
void set_ray_step(float step);
void set_ray_threshold(float threshold);

void set_raycaster_model(Volume_model model);

Raycaster get_raycaster();

#endif