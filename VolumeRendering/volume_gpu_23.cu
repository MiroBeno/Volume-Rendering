// CUDA implementation using constant memory / constant memory + GL interop

#include "Renderer.h"

#include "cuda_runtime_api.h"

static __constant__ Raycaster raycaster;
static __constant__ float4 transfer_fn[256];

GPURenderer2::GPURenderer2(int2 size, float4 *tf, Volume_model volume) {
	set_window_buffer(size);
	set_transfer_fn(tf);
	set_volume(volume);
}

GPURenderer3::GPURenderer3(int2 size, float4 *tf, Volume_model volume) {
	set_window_buffer(size);
	set_transfer_fn(tf);
	set_volume(volume);
}

__global__ void render_ray(uchar4 dev_buffer[], unsigned char dev_volume_data[]) {
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
			float4 color_cur = raycaster.sample_color(dev_volume_data, transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

void GPURenderer2::set_transfer_fn(float4 *tf) {
	cudaMemcpyToSymbol(transfer_fn, tf, 256 * sizeof(float4));
}

int GPURenderer2::render_volume(uchar4 *buffer, Raycaster *r) {
	cudaMemcpyToSymbol(raycaster, r, sizeof(Raycaster));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer, dev_volume_data);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	return 0;
}

int GPURenderer3::render_volume(uchar4 *buffer, Raycaster *r) {
	cudaMemcpyToSymbol(raycaster, r, sizeof(Raycaster));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(buffer, dev_volume_data);
	return 0;
}