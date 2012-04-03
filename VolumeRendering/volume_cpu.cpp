#include "Renderer.h"		

inline void CPURenderer::render_ray(Raycaster raycaster, uchar4 buffer[], int2 pos) {
	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 

	if (raycaster.intersect(origin, direction, &k_range))	{			
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(raycaster.model.data, transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, buffer);
}

void CPURenderer::set_transfer_fn(float4 *transfer_fn) {
	this->transfer_fn = transfer_fn;
}

float CPURenderer::render_volume(uchar4 *buffer, Raycaster *raycaster) {
	for(int row = 0; row < raycaster->view.size_px.y; row++)
		for(int col = 0; col < raycaster->view.size_px.x; col++)	{
			render_ray(*raycaster, buffer, make_int2(col, row));
		}
	return 0;
}
