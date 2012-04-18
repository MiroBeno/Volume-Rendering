// CUDA implementation using constant memory + 3D texture memory + GL interop

#include "cuda_utils.h"
#include "Renderer.h"

static __constant__ Raycaster raycaster;
static __constant__ float4 transfer_fn[TF_SIZE];
//static __constant__ unsigned char esl_volume[ESL_VOLUME_SIZE];

cudaArray *volume_array = 0;
texture<unsigned char, 3, cudaReadModeNormalizedFloat> volume_texture;
cudaArray *transfer_fn_array = 0;
texture<float4, 1, cudaReadModeElementType> transfer_fn_texture;
cudaArray *esl_array = 0;
texture<unsigned char, 3, cudaReadModeElementType> esl_texture;

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

__device__ float4 sample_color_texture(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	float4 color = tex1D(transfer_fn_texture, sample);
	//float4 color = transfer_fn[int(sample*(TF_SIZE-1))]; 
	color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)			//vyhodit z kernela, prevypocitat
	color.y *= color.w;
	color.z *= color.w;
	return color;
}

__device__ float4 sample_color_texture_interpolated(float3 pos) {
	float sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
	float index = sample * (TF_SIZE-1);
	float floor_index = floor(index);
	float ceil_index = ceil(index);
	float4 color = floor_index != ceil_index
					? transfer_fn[(int) floor_index] * abs(floor_index - index) + transfer_fn[(int) ceil_index] * abs(ceil_index - index)
					: transfer_fn[(int) index];
	color.x *= color.w;				// aplikovanie optickeho modelu pre kompoziciu (farba * alfa)			//vyhodit z kernela, prevypocitat
	color.y *= color.w;
	color.z *= color.w;
	return color;
}

__device__  bool sample_data_esl_texture(float3 pos) {
		unsigned char sample = tex3D(esl_texture, 
			map_float_int((pos.x + 1)*0.5f, raycaster.volume.dims.x) / raycaster.esl_block_dims,
			map_float_int((pos.y + 1)*0.5f, raycaster.volume.dims.y) / raycaster.esl_block_dims,
			map_float_int((pos.z + 1)*0.5f, raycaster.volume.dims.z) / raycaster.esl_block_dims);
		return (sample == 0) ? false : true;
}

__device__ float3 shade_texture(float3 pos, float3 dir, float sample) {
		float3 light_dir = vector_normalize(raycaster.view.light_pos - pos);
		sample = tex3D(volume_texture, (pos.x + 1)*0.5f, (pos.y + 1)*0.5f, (pos.z + 1)*0.5f);
		float sample_l = tex3D(volume_texture, 
			(pos.x + light_dir.x * 0.01f + 1)*0.5f,
			(pos.y + light_dir.y * 0.01f + 1)*0.5f,
			(pos.z + light_dir.z * 0.01f + 1)*0.5f);
		//float sample_l = sample_color_texture(pos + light_dir * 0.01f).w;
		float diffuse_light = (sample_l - sample) * raycaster.light_kd;
		//light_dir = vector_normalize(light_dir + dir);
		//sample_l = sample_color_texture(pos + light_dir * 0.01f).w;
		//float specular_light = pow(sample_l - sample + 1, 7) * raycaster.light_ks;
		return make_float3(diffuse_light, diffuse_light, diffuse_light);
}

static __global__ void render_ray(uchar4 dev_buffer[]) {
	short2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (raycaster.intersect(origin, direction, &k_range)) {	
		float3 pt = origin + (direction * k_range.x);
		for(; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {
			if (raycaster.esl && sample_data_esl_texture(pt)) 
				raycaster.leap_empty_space(pt, direction, &k_range);
			else 
				break;
		}
		float4 color_acc = {0, 0, 0, 0};
		/*if (k_range.x > k_range.y) return;
		color_acc = color_acc + (make_float4(0.5f, 0.5f, 1, 0.5f) * (1 - color_acc.w));*/
		for (; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {		
			float4 color_cur = sample_color_texture(pt);
			if (color_cur.w > 0.1f) color_cur = color_cur + shade_texture(pt, direction, color_cur.w);
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
		cuda_safe_call(cudaMallocArray(&transfer_fn_array, &channelDesc, TF_SIZE, 1)); 
		cuda_safe_call(cudaMemcpyToArray(transfer_fn_array, 0, 0, r.transfer_fn, TF_SIZE * sizeof(float4), cudaMemcpyHostToDevice));

		transfer_fn_texture.filterMode = cudaFilterModeLinear; 
		transfer_fn_texture.normalized = true;
		transfer_fn_texture.addressMode[0] = cudaAddressModeClamp; 
		cuda_safe_call(cudaBindTextureToArray(transfer_fn_texture, transfer_fn_array, channelDesc));
	}
	else {
		cuda_safe_call(cudaMemcpyToArray(transfer_fn_array, 0, 0, r.transfer_fn, TF_SIZE * sizeof(float4), cudaMemcpyHostToDevice));
		cuda_safe_call(cudaMemcpyToSymbol(transfer_fn, r.transfer_fn, TF_SIZE * sizeof(float4)));
	}
	if (esl_array == 0) {
		cudaExtent volumeDims = {ESL_VOLUME_DIMS, ESL_VOLUME_DIMS, ESL_VOLUME_DIMS};	
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();	
		cuda_safe_call(cudaMalloc3DArray(&esl_array, &channelDesc, volumeDims));

		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr   = make_cudaPitchedPtr(r.esl_volume, volumeDims.width*sizeof(unsigned char), volumeDims.width, volumeDims.height);
		copyParams.dstArray = esl_array;
		copyParams.extent   = volumeDims;
		copyParams.kind     = cudaMemcpyHostToDevice;
		cuda_safe_call(cudaMemcpy3D(&copyParams));

		esl_texture.normalized = false;
		esl_texture.filterMode = cudaFilterModePoint;  
		esl_texture.addressMode[0] = cudaAddressModeClamp;  
		esl_texture.addressMode[1] = cudaAddressModeClamp;
		esl_texture.addressMode[2] = cudaAddressModeClamp;
		cuda_safe_call(cudaBindTextureToArray(esl_texture, esl_array, channelDesc));
	}
	else {
		cudaExtent volumeDims = {ESL_VOLUME_DIMS, ESL_VOLUME_DIMS, ESL_VOLUME_DIMS};	
		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr   = make_cudaPitchedPtr(r.esl_volume, volumeDims.width*sizeof(unsigned char), volumeDims.width, volumeDims.height);
		copyParams.dstArray = esl_array;
		copyParams.extent   = volumeDims;
		copyParams.kind     = cudaMemcpyHostToDevice;
		cuda_safe_call(cudaMemcpy3D(&copyParams));
		//cuda_safe_call(cudaMemcpyToSymbol(esl_volume, r.esl_volume, ESL_VOLUME_SIZE * sizeof(unsigned char)));
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