// CUDA implementation using constant memory / constant memory + GL interop

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"

extern dim3 THREADS_PER_BLOCK;
extern dim3 num_blocks;

static __constant__ Raycaster raycaster;
static __constant__ float4 dev_transfer_fn[256];

extern uchar4 *dev_buffer;
extern int dev_buffer_size;
extern unsigned char *dev_volume_data;

__global__ void render_ray_gpu2(uchar4 dev_buffer[], unsigned char dev_volume_data[]) {
	int2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 

	if (raycaster.intersect(origin, direction, &k_range)) {				
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(dev_volume_data, dev_transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

extern void set_transfer_fn_gpu23(float4 *transfer_fn) {
	cudaMemcpyToSymbol(dev_transfer_fn, transfer_fn, 256 * sizeof(float4));
}

extern float render_volume_gpu2(uchar4 *buffer, Raycaster *current_raycaster) {
	cudaMemcpyToSymbol(raycaster, current_raycaster, sizeof(Raycaster));
	render_ray_gpu2<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer, dev_volume_data);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	return 0;
}

extern float render_volume_gpu3(uchar4 *buffer, Raycaster *current_raycaster) {
	cudaMemcpyToSymbol(raycaster, current_raycaster, sizeof(Raycaster));
	render_ray_gpu2<<<num_blocks, THREADS_PER_BLOCK>>>(buffer, dev_volume_data);
	return 0;
}