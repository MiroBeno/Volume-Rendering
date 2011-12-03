#include "data_utils.h"
#include "projection.h"
#include "model.h"

#include <ctime>

static const float4 bg_color = {0.5,0.5,0.5,1};			// opacity backgroundu je 1

static Volume_model volume;
static View view;

static clock_t startTime;
static float elapsedTime;

float4 render_ray_cpu(float3 origin, float3 direction) {
	float2 k_range = volume.intersect(origin, direction);
	if ((k_range.x > k_range.y) || (k_range.y < 0))				// prazdny interval koeficientu k = nie je presecnik ALEBO vystupny priesecnik je za bodom vzniku luca
		return bg_color;
	if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
		k_range.x = 0;
	float4 color_acc = {0,0,0,0};
	for (float k = k_range.x; k <= k_range.y; k += volume.ray_step) {		
		float3 pt = origin + (direction * k);
		float4 color_cur = volume.sample_color(pt);
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

extern void init_cpu(Volume_model volume_model) {
	volume = volume_model;
}

extern float render_volume_cpu(unsigned char *buffer, View current_view) {
	view = current_view;
	float3 origin = {0,0,0}, direction = {0,0,0};
	startTime = clock();
	for(int row = 0; row < WIN_HEIGHT; row++)
		for(int col = 0; col < WIN_WIDTH; col++)
		{	
			view.get_ortho_ray(col, row, &origin, &direction);
			float4 color = render_ray_cpu(origin, direction);
			*buffer++ = (unsigned char) map_float_int(color.x,256);
			*buffer++ = (unsigned char) map_float_int(color.y,256);
			*buffer++ = (unsigned char) map_float_int(color.z,256);
			*buffer++ = 255;
		}
	elapsedTime = (clock() - startTime) / (CLOCKS_PER_SEC / 1000.0f);
	return elapsedTime;
}


//////////////////
//OLD CODE
//////////////////

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
	intersect1D_alt(volume.min_bound.x, volume.max_bound.x, origin.x, direction.x, &k_range);
	intersect1D_alt(volume.min_bound.y, volume.max_bound.y, origin.y, direction.y, &k_range);
	intersect1D_alt(volume.min_bound.z, volume.max_bound.z, origin.z, direction.z, &k_range);
	if ((k_range.x > k_range.y) || (k_range.y <0))				
		return bg_color;
	if ((k_range.x < 0))										
		k_range.x = 0;
	color_acc.x = 0; color_acc.y = 0; color_acc.z = 0; color_acc.w = 0;
	for (k = k_range.x; k <= k_range.y; k += volume.ray_step) {		
		pnt.x = origin.x + direction.x * k;
		pnt.y = origin.y + direction.y * k;
		pnt.z = origin.z + direction.z * k;
		color_cur = volume.sample_color(pnt);
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