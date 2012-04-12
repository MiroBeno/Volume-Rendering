// CUDA implementation using constant memory + 3D texture memory + GL interop

#include "cuda_utils.h"
#include "Renderer.h"

static __constant__ Raycaster raycaster;
//static __constant__ float4 transfer_fn[256];

cudaArray *volume_array = 0;
texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
cudaArray *transfer_fn_array = 0;
texture<float4, 1, cudaReadModeElementType> transfer_fn_texture;

GPURenderer4::GPURenderer4(Raycaster r) {
	set_window_buffer(r.view);
	set_transfer_fn(r);
	set_volume(r.volume);
}

GPURenderer4::~GPURenderer4() {
	cuda_safe_call(cudaUnbindTexture(volume_texture));
	cuda_safe_call(cudaFreeArray(volume_array));
	cuda_safe_call(cudaUnbindTexture(transfer_fn_texture));
	cuda_safe_call(cudaFreeArray(transfer_fn_array));
}

__device__ float4 sample_color_texture(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	float4 color = tex1D(transfer_fn_texture, sample);
	//float4 color = transfer_fn[int(sample*255)]; 
	color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)
	color.y *= color.w;
	color.z *= color.w;
	return color;
}

__global__ void render_ray(uchar4 dev_buffer[]) {
	short2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (raycaster.intersect(origin, direction, &k_range)) {	
		float4 color_acc = {0, 0, 0, 0};
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = sample_color_texture(pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
		raycaster.write_color(color_acc, pos, dev_buffer);
	}
}

void GPURenderer4::set_transfer_fn(Raycaster r) {
	if (transfer_fn_array == 0) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cuda_safe_call(cudaMallocArray(&transfer_fn_array, &channelDesc, 256, 1)); 
		cuda_safe_call(cudaMemcpyToArray(transfer_fn_array, 0, 0, r.transfer_fn, 256 * sizeof(float4), cudaMemcpyHostToDevice));

		transfer_fn_texture.filterMode = cudaFilterModeLinear; 
		transfer_fn_texture.normalized = true;
		transfer_fn_texture.addressMode[0] = cudaAddressModeClamp; 
		cuda_safe_call(cudaBindTextureToArray(transfer_fn_texture, transfer_fn_array, channelDesc));
	}
	else {
		cuda_safe_call(cudaMemcpyToArray(transfer_fn_array, 0, 0, r.transfer_fn, 256 * sizeof(float4), cudaMemcpyHostToDevice));
		//cuda_safe_call(cudaMemcpyToSymbol(transfer_fn, r.transfer_fn, 256 * sizeof(float4)));
	}
}

void GPURenderer4::set_volume(Model volume) {
	cudaExtent volumeDims = {volume.dims.x, volume.dims.y, volume.dims.z};	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();	
	cuda_safe_call(cudaMalloc3DArray(&volume_array, &channelDesc, volumeDims));

    cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(volume.data, volumeDims.width*sizeof(unsigned char), volumeDims.width, volumeDims.height);
    copyParams.dstArray = volume_array;
    copyParams.extent   = volumeDims;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cuda_safe_call(cudaMemcpy3D(&copyParams));

    volume_texture.normalized = true;                      
    volume_texture.filterMode = cudaFilterModeLinear; //vypnut pri cm ?   
    volume_texture.addressMode[0] = cudaAddressModeClamp;  
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;
    cuda_safe_call(cudaBindTextureToArray(volume_texture, volume_array, channelDesc));
}

int GPURenderer4::render_volume(uchar4 *buffer, Raycaster r) {
	cuda_safe_call(cudaMemset(buffer, 0, dev_buffer_size));
	cuda_safe_call(cudaMemcpyToSymbol(raycaster, &r, sizeof(Raycaster)));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(buffer);
	cuda_safe_check();
	return 0;
}