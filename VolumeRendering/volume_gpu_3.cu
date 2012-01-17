// CUDA implementation using constant + 3D texture memory

#include "data_utils.h"
#include "projection.h"
#include "model.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "driver_functions.h"
#include "channel_descriptor.h"
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"

extern int BUFFER_SIZE_CUDA;
extern dim3 THREADS_PER_BLOCK;

extern Volume_model volume_model;

static __constant__ View view;
static __constant__ Raycaster raycaster;

extern uchar4 *dev_buffer;

texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
cudaArray *volume_array = 0;

extern cudaEvent_t start, stop; 
extern float elapsedTime;

__device__ float4 sample_color_texture(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	return raycaster.transfer_function(sample, pos);
}

__global__ void render_ray_gpu3(uchar4 dev_buffer[]) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ((col >= view.size_px.x) || (row >= view.size_px.y))					// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	//float bg = (((col / 16) + (row / 16)) % 2) * 0.15f;
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
			float4 color_cur = sample_color_texture(pt);
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

extern void init_gpu3() {
	cudaExtent volumeDims = {volume_model.dims.x, volume_model.dims.y, volume_model.dims.z};	

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();	
	cudaMalloc3DArray(&volume_array, &channelDesc, volumeDims);

    cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(volume_model.data, volumeDims.width*sizeof(unsigned char), volumeDims.width, volumeDims.height);
    copyParams.dstArray = volume_array;
    copyParams.extent   = volumeDims;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    volume_texture.normalized = true;                      
    volume_texture.filterMode = cudaFilterModeLinear;      
    volume_texture.addressMode[0] = cudaAddressModeClamp;  
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(volume_texture, volume_array, channelDesc);
}

extern void free_gpu3() {
	cudaUnbindTexture(volume_texture);
	cudaFreeArray(volume_array);
}

extern float render_volume_gpu3(uchar4 *buffer, View current_view, Raycaster current_raycaster) {
	current_raycaster.model = volume_model;
	dim3 num_blocks((current_view.size_px.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					(current_view.size_px.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(view, &current_view, sizeof(View));
	cudaMemcpyToSymbol(raycaster, &current_raycaster, sizeof(Raycaster));
	render_ray_gpu3<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer);
	cudaMemcpy(buffer, dev_buffer, BUFFER_SIZE_CUDA, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}