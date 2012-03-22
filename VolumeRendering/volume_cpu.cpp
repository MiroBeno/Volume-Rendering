#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include <ctime>

static Raycaster raycaster;

static clock_t startTime;
static float elapsedTime;

void render_ray_cpu(uchar4 buffer[], int2 pos) {
	float3 origin, direction;
	raycaster.view.get_ray(pos, &origin, &direction);
	float2 k_range = raycaster.intersect(origin, direction);

	float4 color_acc = {0,0,0,0};
	if ((k_range.x < k_range.y) || (k_range.y > 0))	{			// prazdny interval koeficientu k = nie je presecnik ALEBO vystupny priesecnik je za bodom vzniku luca
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, buffer);
}

extern float render_volume_cpu(uchar4 *buffer, Raycaster current_raycaster) {
	raycaster = current_raycaster;
	startTime = clock();
	for(int row = 0; row < raycaster.view.size_px.y; row++)
		for(int col = 0; col < raycaster.view.size_px.x; col++)	{
			render_ray_cpu(buffer, make_int2(col, row));
		}
	elapsedTime = (clock() - startTime) / (CLOCKS_PER_SEC / 1000.0f);
	return elapsedTime;
}
