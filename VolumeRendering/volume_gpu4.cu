// CUDA implementation using constant + 3D texture memory + GL interop

#include "data_utils.h"
#include "projection.h"
#include "model.h"

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "driver_functions.h"
#include "channel_descriptor.h"
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"

extern int BUFFER_SIZE_CUDA;

extern Volume_model volume_model;
static __constant__ Volume_model volume;
static __constant__ Ortho_view view;

texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
extern cudaArray *volume_array;

extern cudaEvent_t start, stop; 
extern float elapsedTime;

__device__ float4 sample_color_texture4(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	return volume.transfer_function(sample, pos);
}

__global__ void render_ray_gpu4(uchar4 dev_buffer[]) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ((col >= view.size_px.x) || (row >= view.size_px.y))					// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float bg = (((col / 16) + (row / 16)) % 2) * 0.2f;
	float4 bg_color = {bg, bg, bg, 1};
	float4 color_acc;

	float3 origin = {0,0,0}, direction = {0,0,0};
	view.get_view_ray(col, row, &origin, &direction);
	float2 k_range = volume.intersect(origin, direction);

	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
			k_range.x = 0;
		color_acc = make_float4(0,0,0,0);
		for (float k = k_range.x; k <= k_range.y; k += volume.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = sample_color_texture4(pt);
			color_cur.x *= color_cur.w;								// transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			color_cur.y *= color_cur.w;
			color_cur.z *= color_cur.w;
			color_acc = color_acc + (color_cur * (1 - color_acc.w));
			if (color_acc.w > 0.95f) 
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

extern void init_gpu4() {
	cudaMemcpyToSymbol(volume, &volume_model, sizeof(Volume_model));

    volume_texture.normalized = true;                      
    volume_texture.filterMode = cudaFilterModeLinear;      
    volume_texture.addressMode[0] = cudaAddressModeClamp;  
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaBindTextureToArray(volume_texture, volume_array, channelDesc);
}

extern void free_gpu4() {
	cudaUnbindTexture(volume_texture);
}

extern float render_volume_gpu4(uchar4 *buffer, Ortho_view ortho_view) {
	int threads_dim = 16;
	dim3 threads_per_block(threads_dim, threads_dim);				// podla occupancy calculator
	dim3 num_blocks((WIN_WIDTH + threads_dim - 1) / threads_dim, (WIN_HEIGHT + threads_dim - 1) / threads_dim);		// celociselne delenie, 
																													// ak su rozmery okna nedelitelne 16, spustaju sa bloky	s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(view, &ortho_view, sizeof(Ortho_view));
	render_ray_gpu4<<<num_blocks, threads_per_block>>>(buffer);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}