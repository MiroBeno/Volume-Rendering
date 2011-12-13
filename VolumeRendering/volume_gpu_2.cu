// CUDA implementation using constant memory

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"

extern int BUFFER_SIZE_CUDA;

extern Volume_model volume_model;
static __constant__ Volume_model volume;
static __constant__ View view;
static __constant__ Raycaster raycaster;

extern uchar4 *dev_buffer;

extern cudaEvent_t start, stop; 
extern float elapsedTime;

__global__ void render_ray_gpu2(uchar4 dev_buffer[]) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ((col >= view.size_px.x) || (row >= view.size_px.y))					// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	//float bg = (((col / 16) + (row / 16)) % 2) * 0.1f;
	//float4 bg_color = {bg, bg, bg, 1};
	float4 bg_color = {0.5,0.5,0.5,1};
	float4 color_acc;

	float3 origin = {0,0,0}, direction = {0,0,0};
	view.get_ortho_ray(col, row, &origin, &direction);
	float2 k_range = raycaster.intersect(origin, direction);

	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
			k_range.x = 0;
		color_acc = make_float4(0,0,0,0);
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(pt);
			color_cur.x *= color_cur.w;								// transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			color_cur.y *= color_cur.w;
			color_cur.z *= color_cur.w;
			color_acc = color_acc + (color_cur * (1 - color_acc.w));
			if (color_acc.w > raycaster.ray_thershold) 
				break;
		}
		color_acc = color_acc + (bg_color * (1 - color_acc.w));	
	}
	else {
		color_acc = bg_color;
	}

	int offset = (row * WIN_WIDTH + col);
	dev_buffer[offset].x = map_float_int(color_acc.x,256);
	dev_buffer[offset].y = map_float_int(color_acc.y,256);
	dev_buffer[offset].z = map_float_int(color_acc.z,256);
	dev_buffer[offset].w = 255;
}

extern void init_gpu2() {
	cudaMemcpyToSymbol(volume, &volume_model, sizeof(Volume_model));
}

extern float render_volume_gpu2(uchar4 *buffer, View current_view, Raycaster current_raycaster) {
	current_raycaster.model = volume_model;
	int threads_dim = 16;
	dim3 threads_per_block(threads_dim, threads_dim);				// podla occupancy calculator
	dim3 num_blocks((WIN_WIDTH + threads_dim - 1) / threads_dim, (WIN_HEIGHT + threads_dim - 1) / threads_dim);		// celociselne delenie, 
																													// ak su rozmery okna nedelitelne 16, spustaju sa bloky	s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(view, &current_view, sizeof(View));
	cudaMemcpyToSymbol(raycaster, &current_raycaster, sizeof(Raycaster));
	render_ray_gpu2<<<num_blocks, threads_per_block>>>(dev_buffer);
	cudaMemcpy(buffer, dev_buffer, BUFFER_SIZE_CUDA, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}