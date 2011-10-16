#include <stdio.h>
#include <stdlib.h>
#include "model.h"

static int volume_size_x, volume_size_y, volume_size_z;	
static size_t volume_size_bytes = 0;
static unsigned char *volume = NULL;
static float3 min_bound, max_bound;
static float step = 0.06f;											// upravit podla poctu voxelov 
static const float4 bg_color = mk_float4(0,0,0,1);					// opacity backgroundu je 1

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
	int max_size = MAXIMUM(volume_size_x, MAXIMUM(volume_size_y,volume_size_z));	// dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
	max_bound = mk_float3(volume_size_x / (float) max_size, volume_size_y / (float) max_size, volume_size_z / (float) max_size);
	min_bound = mul(max_bound, -1);
	return 0;
}

unsigned char sample_data(float3 pos) {
   return volume[
		map_interval((pos.z + 1) / 2, volume_size_z) * volume_size_x * volume_size_y +
		map_interval((pos.y + 1) / 2, volume_size_y) * volume_size_x +
		map_interval((pos.x + 1) / 2, volume_size_x)
	];
}

float4 transfer_function(unsigned char sample) {
	return mk_float4(sample / 255.0f, sample / 255.0f, sample / 255.0f, sample / 255.0f);
}

float4 sample_color(float3 point) {
	#if 1
		return transfer_function(sample_data(point));
	#else
		point = mul(add(point, mk_float3(1,1,1)), 0.5f);		// propocitanie polohy bodu <-1;1>(x,y,z) na float vyjadrenie farby <0;1>(r,g,b,1)
		return mk_float4(point.x, point.y, point.z, 0.5f);			
	//	return mk_float4((sample * point.x)/ 255.0f, (sample * point.y) / 255.0f, (sample * point.z) / 255.0f, sample / 255.0f);
	#endif
}

float2 intersect_1D(float pt, float dir, float min_bound, float max_bound) {
	if (dir == 0) {										// ak je zlozka vektora rovnobezna so stenou kocky
		if ((pt < min_bound) || (pt > max_bound))		// ak nelezi bod v romedzi kocky v danej osi
			return mk_float2(POS_INF, NEG_INF);			// interval bude nulovy
		else
			return mk_float2(NEG_INF, POS_INF);			// inak interval bude nekonecny
	}
	float k1 = (min_bound - pt) / dir;
	float k2 = (max_bound - pt) / dir;
	return k1 <= k2 ? mk_float2(k1, k2) : mk_float2(k2, k1); // skontroluj opacny vektor
}

float2 intersect_3D(float3 pt, float3 dir) {
	float2 xRange = intersect_1D(pt.x, dir.x, min_bound.x, max_bound.x);
	float2 yRange = intersect_1D(pt.y, dir.y, min_bound.y, max_bound.y);
	float2 zRange = intersect_1D(pt.z, dir.z, min_bound.z, max_bound.z);
	float k1 = xRange.a, k2 = xRange.b;
	if (yRange.a > k1) k1 = yRange.a;
	if (zRange.a > k1) k1 = zRange.a;
	if (yRange.b < k2) k2 = yRange.b;
	if (zRange.b < k2) k2 = zRange.b;
	return mk_float2(k1, k2);					// pri vypocte k mozu vzniknut artefakty, a hodnoty mozu byt mimo volume, mozno riesit k +-= 0.00001f; alebo clampovanim vysledku na stenu
}

float4 render_ray(float3 origin, float3 direction) {
	float2 k_range = intersect_3D(origin, direction);
	if ((k_range.a > k_range.b) || (k_range.b <0))				// prazdny interval koeficientu k = nie je presecnik ALEBO vystupny priesecnik je za bodom vzniku luca
		return bg_color;
	if ((k_range.a < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
		k_range.a = 0;
	float4 color_acc = mk_float4(0,0,0,0);
	for (float k = k_range.a; k <= k_range.b; k += step) {		
		float3 pt = add(origin, mul(direction,k));
		float4 color_cur = sample_color(pt);
		color_cur.r *= color_cur.a;								// transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
		color_cur.g *= color_cur.a;
		color_cur.b *= color_cur.a;
		color_acc = add(color_acc, mul(color_cur, 1-color_acc.a));
		if (color_acc.a > 0.95f) 
			break;
	}
	color_acc = add(color_acc, mul(bg_color, 1-color_acc.a));			// opacity backgroundu je 1
	color_acc.a = 1.0f;
	return color_acc;
};

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
					ival->a = POS_INF;							// interval bude nulovy
					ival->b = NEG_INF;
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
			if (k1 > ival->a)									// orezanie intervalu zlava (max z k1)
				ival->a = k1;
			if (k2 < ival->b)									// orezanie intervalu sprava (min z k2)
				ival->b = k2;
}

float4 render_ray_alt(float3 origin, float3 direction) {
	k_range.a = NEG_INF;
	k_range.b = POS_INF;
	intersect1D_alt(min_bound.x, max_bound.x, origin.x, direction.x, &k_range);
	intersect1D_alt(min_bound.y, max_bound.y, origin.y, direction.y, &k_range);
	intersect1D_alt(min_bound.z, max_bound.z, origin.z, direction.z, &k_range);
	if ((k_range.a > k_range.b) || (k_range.b <0))				
		return bg_color;
	if ((k_range.a < 0))										
		k_range.a = 0;
	color_acc.r = 0; color_acc.g = 0; color_acc.b = 0; color_acc.a = 0;
	for (k = k_range.a; k <= k_range.b; k += step) {		
		pnt.x = origin.x + direction.x * k;
		pnt.y = origin.y + direction.y * k;
		pnt.z = origin.z + direction.z * k;
		color_cur = sample_color(pnt);
		color_cur.r *= color_cur.a;								
		color_cur.g *= color_cur.a;
		color_cur.b *= color_cur.a;
		color_acc.r += color_cur.r * (1-color_acc.a);
		color_acc.g += color_cur.g * (1-color_acc.a);
		color_acc.b += color_cur.b * (1-color_acc.a);
		color_acc.a += color_cur.a * (1-color_acc.a);
		if (color_acc.a > 0.95f) 
			break;
	}
	color_acc.r += bg_color.r * (1-color_acc.a);
	color_acc.g += bg_color.g * (1-color_acc.a);
	color_acc.b += bg_color.b * (1-color_acc.a);
	color_acc.a += bg_color.a * (1-color_acc.a);
	return color_acc;
};