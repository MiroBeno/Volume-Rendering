#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

static float4 *my_transfer_fn;

inline void render_ray_cpu(Raycaster raycaster, uchar4 buffer[], int2 pos) {
	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	raycaster.view.get_ray(pos, &origin, &direction);
	float2 k_range = raycaster.intersect(origin, direction);

	if ((k_range.x < k_range.y) && (k_range.y > 0))	{			// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(raycaster.model.data, my_transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, buffer);
}

extern void set_transfer_fn_cpu(float4 *transfer_fn) {
	my_transfer_fn = transfer_fn;
}

extern float render_volume_cpu(uchar4 *buffer, Raycaster *current_raycaster) {
	for(int row = 0; row < current_raycaster->view.size_px.y; row++)
		for(int col = 0; col < current_raycaster->view.size_px.x; col++)	{
			render_ray_cpu(*current_raycaster, buffer, make_int2(col, row));
		}
	return 0;
}
