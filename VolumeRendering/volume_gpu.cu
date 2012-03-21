// Standard CUDA implementation

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"

dim3 THREADS_PER_BLOCK(16, 16);				// pocet threadov v bloku - podla occupancy calculator

unsigned char *dev_volume_data;
uchar4 *dev_buffer;
int dev_buffer_size = WIN_WIDTH * WIN_HEIGHT * 4;
cudaEvent_t start, stop; 
float elapsedTime;

__global__ void render_ray_gpu(Raycaster raycaster, uchar4 dev_buffer[]) {
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
	color_acc = color_acc + (raycaster.bg_color * (1 - color_acc.w));
	raycaster.write_color(color_acc, pos, dev_buffer);
}

extern void init_gpu(Volume_model volume) {
	cudaMalloc((void **)&dev_volume_data, volume.size);
	cudaMemcpy(dev_volume_data, volume.data, volume.size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_buffer, dev_buffer_size);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

extern void free_gpu() {
	cudaFree(dev_buffer);
	cudaFree(dev_volume_data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

extern void reset_gpu_buffer(int buffer_size) {
	cudaFree(dev_buffer);
	dev_buffer_size = buffer_size;
	cudaMalloc((void **)&dev_buffer, dev_buffer_size);
}

extern float render_volume_gpu(uchar4 *buffer, Raycaster current_raycaster) {
	current_raycaster.model.data = dev_volume_data;
	dim3 num_blocks((current_raycaster.view.size_px.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					(current_raycaster.view.size_px.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	render_ray_gpu<<<num_blocks, THREADS_PER_BLOCK>>>(current_raycaster, dev_buffer);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}