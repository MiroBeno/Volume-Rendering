#include <string.h>
#include "Renderer.h"		

CPURenderer::CPURenderer(Raycaster r) {
	set_window_buffer(r.view);
	set_volume(r.volume);
	set_transfer_fn(r);
}

inline static void render_ray(Raycaster raycaster, uchar4 buffer[], short2 pos) {
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (raycaster.intersect(origin, direction, &k_range)) {	
		float3 pt = origin + (direction * k_range.x);
		for(; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {
			if (raycaster.esl && raycaster.sample_data_esl(raycaster.esl_volume, pt)) 
				raycaster.leap_empty_space(pt, direction, &k_range);
			else 
				break;
		}
		float4 color_acc = {0, 0, 0, 0};
		for (; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {		
			unsigned char sample = raycaster.volume.sample_data(pt);
			float4 color_cur = raycaster.transfer_fn[sample / TF_RATIO];
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
		raycaster.write_color(color_acc, pos, buffer);
	}
}

int CPURenderer::render_volume(uchar4 *buffer, Raycaster r) {
	memset(buffer, 0, r.view.size_px.x * r.view.size_px.y * 4);
	for(int row = 0; row < r.view.size_px.y; row++)
		for(int col = 0; col < r.view.size_px.x; col++)	{
			render_ray(r, buffer, make_short2(col, row));
		}
	return 0;
}
