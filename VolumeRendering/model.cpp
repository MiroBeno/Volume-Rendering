#include <stdio.h>
#include <stdlib.h>
#include "model.h"

/*static*/ int volume_size_x, volume_size_y, volume_size_z;	
/*static*/ size_t volume_size_bytes = 0;
/*static*/ unsigned char *volume = NULL;
/*static*/ float3 min_bound, max_bound;
/*static*/ float step = 0.06f;											// upravit podla poctu voxelov 
/*static*/ const float4 bg_color = {0.5,0.5,0.5,1};							// opacity backgroundu je 1

int load_file(const char *file_name, unsigned char **result, size_t *file_size) {
	*file_size = 0;
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) 
	{ 
		*result = NULL;
		return -1;				//opening fail
	} 
	fseek(f, 0, SEEK_END);
	*file_size = (size_t) ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (unsigned char *)malloc(*file_size);
	if (*file_size != fread(*result, sizeof(unsigned char), *file_size, f)) 
	{ 
		free(*result);
		return -2;				//reading fail
	} 
	fclose(f);
	return 0;
}

int load_model(const char* file_name) {
	int result = load_file(file_name, &volume, &volume_size_bytes);
	if (volume_size_bytes == 0 || result != 0) 
		return -1;
	printf("File loaded: %s. Size: %u B.\n", file_name, volume_size_bytes);
	volume_size_x = volume_size_y = volume_size_z = 32;								// nacita sa z hlavicky, zatial explicitne
	int max_size = MAXIMUM(volume_size_x, MAXIMUM(volume_size_y, volume_size_z));	// dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
	max_bound = make_float3(volume_size_x / (float) max_size, volume_size_y / (float) max_size, volume_size_z / (float) max_size);
	min_bound = -max_bound;
	return 0;
}

unsigned char sample_data(float3 pos) {
   return volume[
		map_float_int((pos.z + 1) / 2, volume_size_z) * volume_size_x * volume_size_y +
		map_float_int((pos.y + 1) / 2, volume_size_y) * volume_size_x +
		map_float_int((pos.x + 1) / 2, volume_size_x)
	];
}

float4 transfer_function(unsigned char sample) {
	return make_float4(sample / 255.0f, sample / 255.0f, sample / 255.0f, sample / 255.0f);
}

float4 sample_color(float3 point) {
	#if 0
		return transfer_function(sample_data(point));
	#else
		point = (point + make_float3(1,1,1)) * 0.5f;		// prepocitanie polohy bodu <-1;1>(x,y,z) na float vyjadrenie farby <0;1>(r,g,b,1)
		return make_float4(point.x, point.y, point.z, 0.05f);			
	#endif
}

float2 intersect_1D(float pt, float dir, float min_bound, float max_bound) {
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

float2 intersect_3D(float3 pt, float3 dir) {
	float2 xRange = intersect_1D(pt.x, dir.x, min_bound.x, max_bound.x);
	float2 yRange = intersect_1D(pt.y, dir.y, min_bound.y, max_bound.y);
	float2 zRange = intersect_1D(pt.z, dir.z, min_bound.z, max_bound.z);
	float k1 = xRange.x, k2 = xRange.y;
	if (yRange.x > k1) k1 = yRange.x;
	if (zRange.x > k1) k1 = zRange.x;
	if (yRange.y < k2) k2 = yRange.y;
	if (zRange.y < k2) k2 = zRange.y;
	return make_float2(k1, k2);					// pri vypocte k mozu vzniknut artefakty, a hodnoty mozu byt mimo volume, mozno riesit k +-= 0.00001f; alebo clampovanim vysledku na stenu
}

float4 render_ray(float3 origin, float3 direction) {
	float2 k_range = intersect_3D(origin, direction);
	if ((k_range.x > k_range.y) || (k_range.y < 0))				// prazdny interval koeficientu k = nie je presecnik ALEBO vystupny priesecnik je za bodom vzniku luca
		return bg_color;
	if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
		k_range.x = 0;
	float4 color_acc = {0,0,0,0};
	for (float k = k_range.x; k <= k_range.y; k += step) {		
		float3 pt = origin + (direction * k);
		float4 color_cur = sample_color(pt);
		color_cur.x *= color_cur.w;								// transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
		color_cur.y *= color_cur.w;
		color_cur.z *= color_cur.w;
		color_acc = color_acc + (color_cur * (1 - color_acc.w));
		if (color_acc.w > 0.95f) 
			break;
	}
	color_acc = color_acc + (bg_color * (1 - color_acc.w));	
	return color_acc;
}

//////////////////
//OLD CODE
/////////////////

float2 k_range = {0,0};
float4 color_acc = {0,0,0,0}, color_cur = {0,0,0,0};
float k, k1, k2, pom;
float3 pnt;

void intersect1D_alt(float bound_min, float bound_max, float o, float d, float2 *ival) {
			if (d == 0)	{										// ak je zlozka vektora rovnobezna so stenou kocky
				if ((o < bound_min) || (o > bound_max)) {			// ak nelezi bod v romedzi kocky v danej osi
					ival->x = POS_INF;							// interval bude nulovy
					ival->y = NEG_INF;
				}
				return;											// inak interval neovplyvni
			}
			k1 = (bound_min - o) / d;
			k2 = (bound_max - o) / d;
			if (k1 > k2)  {										// pri opacnom vektore
				pom = k1;
				k1 = k2;
				k2 = pom;
			}
			if (k1 > ival->x)									// orezanie intervalu zlava (max z k1)
				ival->x = k1;
			if (k2 < ival->y)									// orezanie intervalu sprava (min z k2)
				ival->y = k2;
}

float4 render_ray_alt(float3 origin, float3 direction) {
	k_range.x = NEG_INF;
	k_range.y = POS_INF;
	intersect1D_alt(min_bound.x, max_bound.x, origin.x, direction.x, &k_range);
	intersect1D_alt(min_bound.y, max_bound.y, origin.y, direction.y, &k_range);
	intersect1D_alt(min_bound.z, max_bound.z, origin.z, direction.z, &k_range);
	if ((k_range.x > k_range.y) || (k_range.y <0))				
		return bg_color;
	if ((k_range.x < 0))										
		k_range.x = 0;
	color_acc.x = 0; color_acc.y = 0; color_acc.z = 0; color_acc.w = 0;
	for (k = k_range.x; k <= k_range.y; k += step) {		
		pnt.x = origin.x + direction.x * k;
		pnt.y = origin.y + direction.y * k;
		pnt.z = origin.z + direction.z * k;
		color_cur = sample_color(pnt);
		color_cur.x *= color_cur.w;								
		color_cur.y *= color_cur.w;
		color_cur.z *= color_cur.w;
		color_acc.x += color_cur.x * (1-color_acc.w);
		color_acc.y += color_cur.y * (1-color_acc.w);
		color_acc.z += color_cur.z * (1-color_acc.w);
		color_acc.w += color_cur.w * (1-color_acc.w);
		if (color_acc.w > 0.95f) 
			break;
	}
	color_acc.x += bg_color.x * (1-color_acc.w);
	color_acc.y += bg_color.y * (1-color_acc.w);
	color_acc.z += bg_color.z * (1-color_acc.w);
	color_acc.w += bg_color.w * (1-color_acc.w);
	return color_acc;
}
