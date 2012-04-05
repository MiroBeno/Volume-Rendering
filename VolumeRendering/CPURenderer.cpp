#include "Renderer.h"		

static float4 *transfer_fn;
unsigned char *data;

CPURenderer::CPURenderer(int2 size, float4 *tf, Model volume, unsigned char *d) {
	set_window_buffer(size);
	set_transfer_fn(tf);
	set_volume(volume, d);
}

inline void render_ray(Raycaster raycaster, uchar4 buffer[], int2 pos) {
	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 

	if (raycaster.intersect(origin, direction, &k_range))	{			
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(data, transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, buffer);
}

void CPURenderer::set_transfer_fn(float4 *tf) {
	transfer_fn = tf;
}

void CPURenderer::set_volume(Model volume, unsigned char *d) {
	data = d;
}

int CPURenderer::render_volume(uchar4 *buffer, Raycaster *r) {
	for(int row = 0; row < r->view.size_px.y; row++)
		for(int col = 0; col < r->view.size_px.x; col++)	{
			render_ray(*r, buffer, make_int2(col, row));
		}
	return 0;
}
