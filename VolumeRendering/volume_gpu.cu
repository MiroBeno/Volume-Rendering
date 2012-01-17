// Standard CUDA implementation

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"

int BUFFER_SIZE_CUDA = WIN_WIDTH * WIN_HEIGHT * 4;
int THREADS_DIM = 16;
dim3 THREADS_PER_BLOCK(THREADS_DIM, THREADS_DIM);				// pocet threadov v bloku - podla occupancy calculator

Volume_model volume_model;

uchar4 *dev_buffer;

cudaEvent_t start, stop; 
float elapsedTime;

__global__ void render_ray_gpu(View view, Raycaster raycaster, uchar4 dev_buffer[]) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ((col >= view.size_px.x) || (row >= view.size_px.y))					// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	//float bg = (((col / 16) + (row / 16)) % 2) * 0.05f;
	//float4 bg_color = {bg, bg, bg, 1};
	float4 bg_color = {0.5,0.5,0.5,1};
	float4 color_acc;

	float3 origin = {0,0,0}, direction = {0,0,0};
	view.get_ortho_ray(col, row, &origin, &direction);
	float2 k_range = raycaster.intersect(origin, direction);

	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		color_acc = make_float4(0,0,0,0);
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_thershold) 
				break;
		}
		color_acc = color_acc + (bg_color * (1 - color_acc.w));	
	}
	else {
		color_acc = bg_color;
	}

	int offset = (row * view.size_px.x + col);
	dev_buffer[offset].x = map_float_int(color_acc.x,256);
	dev_buffer[offset].y = map_float_int(color_acc.y,256);
	dev_buffer[offset].z = map_float_int(color_acc.z,256);
	dev_buffer[offset].w = 255;
}

extern void init_gpu(Volume_model volume) {
	volume_model = volume;
	unsigned char *dev_volume_data;
	cudaMalloc((void **)&dev_volume_data, volume_model.size);
	cudaMemcpy(dev_volume_data, volume_model.data, volume_model.size, cudaMemcpyHostToDevice);
	volume_model.data = dev_volume_data;
	cudaMalloc((void **)&dev_buffer, BUFFER_SIZE_CUDA);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

extern void free_gpu() {
	cudaFree(dev_buffer);
	cudaFree(volume_model.data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

extern float render_volume_gpu(uchar4 *buffer, View current_view, Raycaster current_raycaster) {
	current_raycaster.model = volume_model;
	dim3 num_blocks((current_view.size_px.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					(current_view.size_px.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	render_ray_gpu<<<num_blocks, THREADS_PER_BLOCK>>>(current_view, current_raycaster, dev_buffer);
	cudaMemcpy(buffer, dev_buffer, BUFFER_SIZE_CUDA, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}