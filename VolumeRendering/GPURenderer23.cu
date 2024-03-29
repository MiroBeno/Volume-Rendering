/****************************************/
// CUDA implementation using constant memory / constant memory + GL interop
/****************************************/

#include "cuda_utils.h"
#include "Renderer.h"

static __constant__ Raycaster raycaster;
static __constant__ float4 transfer_fn[TF_SIZE];
static __constant__ esl_type esl_volume[ESL_VOLUME_SIZE];

GPURenderer2::GPURenderer2(Raycaster r) {
	set_transfer_fn(r);
}

GPURenderer3::GPURenderer3(Raycaster r) {
	set_transfer_fn(r);
}

static __global__ void render_ray(uchar4 dev_buffer[]) {
	short2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.dims.x) || (pos.y >= raycaster.view.dims.y))	// terminate empty thread, when view dimensions are not divisible by 16
		return;

	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (!raycaster.intersect(origin, direction, &k_range))
		return;
	float3 pt = origin + (direction * k_range.x);
	while(k_range.x <= k_range.y) {						// empty space leaping loop
		if (raycaster.esl && raycaster.sample_data_esl(esl_volume, pt)) 
			raycaster.leap_empty_space(pt, direction, &k_range);
		else 
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	if (k_range.x > k_range.y) 
		return;
	float4 color_acc = {0, 0, 0, 0};
	while (k_range.x <= k_range.y) {					// color accumulation loop
		unsigned char sample = raycaster.volume.sample_data(pt);
		float4 color_cur = transfer_fn[sample / TF_RATIO];
		raycaster.shade(&color_cur, pt, sample);		// shading
		color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
		if (color_acc.w > raycaster.ray_threshold)		// early ray termination
			break;
		k_range.x += raycaster.ray_step;
		pt = origin + (direction * k_range.x);
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

void GPURenderer2::set_transfer_fn(Raycaster r) {
	cuda_safe_call(cudaMemcpyToSymbol(transfer_fn, r.transfer_fn, TF_SIZE * sizeof(float4)));
	cuda_safe_call(cudaMemcpyToSymbol(esl_volume, r.esl_volume, ESL_VOLUME_SIZE * sizeof(esl_type)));
}

int GPURenderer2::render_volume(uchar4 *buffer, Raycaster r) {
	if (dev_volume_data == NULL || buffer == NULL)
		return 1;
	r.volume.data = dev_volume_data;
	cuda_safe_call(cudaMemset(dev_buffer, 0, dev_buffer_size));
	cuda_safe_call(cudaMemcpyToSymbol(raycaster, &r, sizeof(Raycaster)));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(dev_buffer);
	cuda_safe_check();
	cuda_safe_call(cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost));
	return 0;
}

int GPURenderer3::render_volume(uchar4 *buffer, Raycaster r) {
	if (dev_volume_data == NULL || buffer == NULL)
		return 1;
	r.volume.data = dev_volume_data;
	cuda_safe_call(cudaMemset(buffer, 0, dev_buffer_size));
	cuda_safe_call(cudaMemcpyToSymbol(raycaster, &r, sizeof(Raycaster)));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(buffer);
	cuda_safe_check();
	return 0;
}