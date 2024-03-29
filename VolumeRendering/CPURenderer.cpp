/****************************************/
// CPU implementation
/****************************************/

#include <string.h>
#include "Renderer.h"		

CPURenderer::CPURenderer(Raycaster r) {
}

inline static void render_ray(Raycaster raycaster, uchar4 buffer[], short2 pos) {
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (!raycaster.intersect(origin, direction, &k_range)) 
		return;
	float3 pt = origin + (direction * k_range.x);
	while(k_range.x <= k_range.y) {						// empty space leaping loop
		if (raycaster.esl && raycaster.sample_data_esl(raycaster.esl_volume, pt)) 
			raycaster.leap_empty_space(pt, direction, &k_range);
		else 
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	if (k_range.x > k_range.y) 
		return;
	float4 color_acc = {0, 0, 0, 0};
	while (k_range.x <= k_range.y) {					// color accumulation loop
		unsigned char sample = raycaster.volume.sample_data(pt);
		float4 color_cur = raycaster.transfer_fn[sample / TF_RATIO];
		if (color_cur.w > 0.05f && raycaster.light_kd > 0.01f)
			raycaster.shade(&color_cur, pt, sample);	// shading
		color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
		if (color_acc.w > raycaster.ray_threshold)		// early ray termination
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	raycaster.write_color(color_acc, pos, buffer);
}

int CPURenderer::render_volume(uchar4 *buffer, Raycaster r) {
	if (r.volume.data == NULL || r.transfer_fn == NULL || r.esl_volume == NULL || buffer == NULL)
		return 1;

	memset(buffer, 0, r.view.dims.x * r.view.dims.y * 4);
	for(int row = 0; row < r.view.dims.y; row++)
		for(int col = 0; col < r.view.dims.x; col++)	{
			render_ray(r, buffer, make_short2(col, row));
		}
	return 0;
}
