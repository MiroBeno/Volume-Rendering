// CUDA implementation using constant memory + 3D texture memory + GL interop

#include "cuda_utils.h"
#include "Renderer.h"

static __constant__ Raycaster raycaster;
//static __constant__ float4 transfer_fn[TF_SIZE];

cudaArray *volume_array = 0;
texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
cudaArray *transfer_fn_array = 0;
texture<float4, 1, cudaReadModeElementType> transfer_fn_texture;
cudaArray *esl_array = 0;
texture<esl_type, 2, cudaReadModeElementType> esl_texture;

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
	cuda_safe_call(cudaUnbindTexture(esl_texture));
	cuda_safe_call(cudaFreeArray(esl_array));
}

__device__  bool sample_data_esl_texture(float3 pos) {
		esl_type sample = tex2D(esl_texture, 
							map_float_int((pos.y + 1)*0.5f, raycaster.volume.dims.y) / raycaster.esl_block_dims,
							map_float_int((pos.z + 1)*0.5f, raycaster.volume.dims.z) / raycaster.esl_block_dims);
		unsigned short index = map_float_int((pos.x + 1)*0.5f, raycaster.volume.dims.x) / raycaster.esl_block_dims;
		return ((sample & (1 << index)) != 0);
}

__device__ void shade_texture(float4 *color, float3 pos, float sample) {
		float3 light_dir = vector_normalize(raycaster.view.light_pos - pos);
		float sample_l = tex3D(volume_texture, 
			(pos.x + light_dir.x * 0.01f + 1)*0.5f,
			(pos.y + light_dir.y * 0.01f + 1)*0.5f,
			(pos.z + light_dir.z * 0.01f + 1)*0.5f);
		float diffuse_light = (sample_l - sample) * raycaster.light_kd;
		color->x += diffuse_light;
		color->y += diffuse_light;
		color->z += diffuse_light;
}

static __global__ void render_ray(uchar4 dev_buffer[]) {
	short2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.dims.x) || (pos.y >= raycaster.view.dims.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (!raycaster.intersect(origin, direction, &k_range)) 
		return;
	float3 pt = origin + (direction * k_range.x);
	while(k_range.x <= k_range.y) { 
		if (raycaster.esl && sample_data_esl_texture(pt)) 
			raycaster.leap_empty_space(pt, direction, &k_range);
		else 
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	if (k_range.x > k_range.y) 
		return;
	float4 color_acc = {0, 0, 0, 0};
	//color_acc = color_acc + (make_float4(0.5f, 0.5f, 1, 0.5f) * (1 - color_acc.w));
	while (k_range.x <= k_range.y) {
		float sample = tex3D(volume_texture, (pt.x + 1)*0.5f, (pt.y + 1)*0.5f, (pt.z + 1)*0.5f);
		float4 color_cur = tex1D(transfer_fn_texture, sample);
		//float4 color_cur = transfer_fn[int(sample*(TF_SIZE-1))];
		if (color_cur.w > 0.05f && raycaster.light_kd > 0.01f)
			shade_texture(&color_cur, pt, sample);
		color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
		if (color_acc.w > raycaster.ray_threshold) 
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

void GPURenderer4::set_transfer_fn(Raycaster r) {
	if (transfer_fn_array == 0) {
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
		cuda_safe_malloc(cudaMallocArray(&transfer_fn_array, &channel_desc, TF_SIZE, 1)); 

		transfer_fn_texture.filterMode = cudaFilterModeLinear; 
		transfer_fn_texture.normalized = true;
		transfer_fn_texture.addressMode[0] = cudaAddressModeClamp; 
		cuda_safe_call(cudaBindTextureToArray(transfer_fn_texture, transfer_fn_array, channel_desc));
	}
	cuda_safe_call(cudaMemcpyToArray(transfer_fn_array, 0, 0, r.transfer_fn, TF_SIZE * sizeof(float4), cudaMemcpyHostToDevice));
	//cuda_safe_call(cudaMemcpyToSymbol(transfer_fn, r.transfer_fn, TF_SIZE * sizeof(float4)));

	if (esl_array == 0) {
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<esl_type>();	
		cuda_safe_malloc(cudaMallocArray(&esl_array, &channel_desc, ESL_VOLUME_DIMS, ESL_VOLUME_DIMS));

		esl_texture.normalized = false;
		esl_texture.filterMode = cudaFilterModePoint;  
		esl_texture.addressMode[0] = cudaAddressModeClamp;  
		esl_texture.addressMode[1] = cudaAddressModeClamp;
		cuda_safe_call(cudaBindTextureToArray(esl_texture, esl_array, channel_desc));
	}
	cuda_safe_call(cudaMemcpyToArray(esl_array, 0, 0, r.esl_volume, ESL_VOLUME_SIZE * sizeof(esl_type), cudaMemcpyHostToDevice));
}

int GPURenderer4::set_volume(Model volume) {
	if (volume_array != 0) {
		cuda_safe_call(cudaUnbindTexture(volume_texture));
		cuda_safe_call(cudaFreeArray(volume_array));
		volume_array = 0;
	}
	if (volume.data == NULL)
		return 1;
	cudaExtent volume_dims = {volume.dims.x, volume.dims.y, volume.dims.z};	
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();	
	cuda_safe_malloc(cudaMalloc3DArray(&volume_array, &channel_desc, volume_dims));
	if (cudaGetLastError() == cudaErrorMemoryAllocation)
		return 1;

    cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(volume.data, volume_dims.width*sizeof(unsigned char), volume_dims.width, volume_dims.height);
    copyParams.dstArray = volume_array;
    copyParams.extent   = volume_dims;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cuda_safe_call(cudaMemcpy3D(&copyParams));

    volume_texture.normalized = true;                      
    volume_texture.filterMode = cudaFilterModeLinear; //vypnut pri cm ?   
    volume_texture.addressMode[0] = cudaAddressModeClamp;  
    volume_texture.addressMode[1] = cudaAddressModeClamp;
    volume_texture.addressMode[2] = cudaAddressModeClamp;
    cuda_safe_call(cudaBindTextureToArray(volume_texture, volume_array, channel_desc));
	return 0;
}

int GPURenderer4::render_volume(uchar4 *buffer, Raycaster r) {
	if (volume_array == 0 || transfer_fn_array == 0 || esl_array == 0 || buffer == NULL)
		return 1;
	cuda_safe_call(cudaMemset(buffer, 0, dev_buffer_size));
	cuda_safe_call(cudaMemcpyToSymbol(raycaster, &r, sizeof(Raycaster)));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(buffer);
	cuda_safe_check();
	return 0;
}