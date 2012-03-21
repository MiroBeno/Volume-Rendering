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

extern dim3 THREADS_PER_BLOCK;

static __constant__ Raycaster raycaster;

extern unsigned char *dev_volume_data;
extern uchar4 *dev_buffer;
extern int dev_buffer_size;
extern cudaEvent_t start, stop; 
extern float elapsedTime;
cudaArray *volume_array = 0;
texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;

__device__ float4 sample_color_texture3(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	return raycaster.transfer_function(sample, pos);
}

__global__ void render_ray_gpu3(uchar4 dev_buffer[]) {
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
			float4 color_cur = sample_color_texture3(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	color_acc = color_acc + (raycaster.bg_color * (1 - color_acc.w));
	raycaster.write_color(color_acc, pos, dev_buffer);
}

extern void init_gpu3(Volume_model volume) {
	cudaExtent volumeDims = {volume.dims.x, volume.dims.y, volume.dims.z};	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();	
	cudaMalloc3DArray(&volume_array, &channelDesc, volumeDims);

    cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(dev_volume_data, volumeDims.width*sizeof(unsigned char), volumeDims.width, volumeDims.height);
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

extern float render_volume_gpu3(uchar4 *buffer, Raycaster current_raycaster) {
	dim3 num_blocks((current_raycaster.view.size_px.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					(current_raycaster.view.size_px.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(raycaster, &current_raycaster, sizeof(Raycaster));
	render_ray_gpu3<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}