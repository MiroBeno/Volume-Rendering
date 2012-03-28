// CUDA implementation using constant memory + 3D texture memory + GL interop

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
extern dim3 num_blocks;

static __constant__ Raycaster raycaster;
//static __constant__ float4 dev_transfer_fn[256];

extern unsigned char *dev_volume_data;
extern cudaEvent_t start, stop; 
extern float elapsedTime;
cudaArray *volume_array = 0;
texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
cudaArray *transfer_fn_array = 0;
texture<float4, 1, cudaReadModeElementType> transfer_fn_texture;

__device__ float4 sample_color_texture(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	float4 color = tex1D(transfer_fn_texture, sample);
	//float4 color = dev_transfer_fn[int(sample*255)]; 
	color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
	color.y *= color.w;
	color.z *= color.w;
	return color;
}

__global__ void render_ray_gpu4(uchar4 dev_buffer[]) {
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
			float4 color_cur = sample_color_texture(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

extern void set_transfer_fn_gpu4(float4 *transfer_fn) {
	if (transfer_fn_array == 0) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&transfer_fn_array, &channelDesc, 256, 1); 
		cudaMemcpyToArray(transfer_fn_array, 0, 0, transfer_fn, 256 * sizeof(float4), cudaMemcpyHostToDevice);

		//transfer_fn_texture.filterMode = cudaFilterModeLinear;
		transfer_fn_texture.normalized = true;
		transfer_fn_texture.addressMode[0] = cudaAddressModeClamp; 
		cudaBindTextureToArray(transfer_fn_texture, transfer_fn_array, channelDesc);
	}
	//cudaMemcpyToArray(transfer_fn_array, 0, 0, transfer_fn, sizeof(transfer_fn), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_transfer_fn, transfer_fn, 256 * sizeof(float4));
}

extern void init_gpu4(Volume_model volume) {
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
    //volume_texture.filterMode = cudaFilterModeLinear;      
    volume_texture.addressMode[0] = cudaAddressModeClamp;  
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(volume_texture, volume_array, channelDesc);
}

extern void free_gpu4() {
	cudaUnbindTexture(volume_texture);
	cudaFreeArray(volume_array);
	cudaUnbindTexture(transfer_fn_texture);
	cudaFreeArray(transfer_fn_array);
}

extern float render_volume_gpu4(uchar4 *buffer, Raycaster *current_raycaster) {
	cudaEventRecord(start, 0);
	cudaMemcpyToSymbol(raycaster, current_raycaster, sizeof(Raycaster));
	render_ray_gpu4<<<num_blocks, THREADS_PER_BLOCK>>>(buffer);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}