// CUDA implementation using constant memory / constant memory + GL interop

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"

extern dim3 THREADS_PER_BLOCK;
extern dim3 num_blocks;

static __constant__ Raycaster raycaster;

extern uchar4 *dev_buffer;
extern int dev_buffer_size;
extern unsigned char *dev_volume_data;
extern cudaEvent_t start, stop; 
extern float elapsedTime;

__global__ void render_ray_gpu2(uchar4 dev_buffer[]) {
	int2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	raycaster.view.get_ray(pos, &origin, &direction);
	float2 k_range = raycaster.intersect(origin, direction);

	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

extern float render_volume_gpu2(uchar4 *buffer, Raycaster current_raycaster) {
	current_raycaster.model.data = dev_volume_data;
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(raycaster, &current_raycaster, sizeof(Raycaster));
	render_ray_gpu2<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}

extern float render_volume_gpu3(uchar4 *buffer, Raycaster current_raycaster) {
	current_raycaster.model.data = dev_volume_data;
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(raycaster, &current_raycaster, sizeof(Raycaster));
	render_ray_gpu2<<<num_blocks, THREADS_PER_BLOCK>>>(buffer);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}