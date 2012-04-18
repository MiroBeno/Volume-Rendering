// Standard CUDA implementation

#include "cuda_utils.h"
#include "Renderer.h"

uchar4 *GPURenderer::dev_buffer = NULL;
int GPURenderer::dev_buffer_size = 0;
unsigned char *GPURenderer::dev_volume_data = NULL;
//unsigned char *GPURenderer::esl_volume = NULL;
dim3 GPURenderer::THREADS_PER_BLOCK(16, 16);				// pocet threadov v bloku - podla occupancy calculator
dim3 GPURenderer::num_blocks(0, 0);

static float4 *transfer_fn = NULL;
static unsigned char *esl_volume = NULL;

GPURenderer1::GPURenderer1(Raycaster r) {
	set_window_buffer(r.view);
	set_transfer_fn(r);
	set_volume(r.volume);
}

GPURenderer1::~GPURenderer1() {
	cuda_safe_call(cudaFree(dev_buffer));
	cuda_safe_call(cudaFree(dev_volume_data));
	cuda_safe_call(cudaFree(transfer_fn));
	cuda_safe_call(cudaFree(esl_volume));
}

static __global__ void render_ray(Raycaster raycaster, uchar4 dev_buffer[]) {
	short2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 
	if (raycaster.intersect(origin, direction, &k_range)) {	
		float3 pt = origin + (direction * k_range.x);
		for(; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {
			if (raycaster.esl && raycaster.sample_data_esl(raycaster.esl_volume, pt)) 
				raycaster.leap_empty_space(pt, direction, &k_range);
			else 
				break;
		}
		float4 color_acc = {0, 0, 0, 0};
		/*if (k_range.x > k_range.y) return;
		color_acc = color_acc + (make_float4(0.5f, 0.5f, 1, 0.5f) * (1 - color_acc.w));*/
		for (; k_range.x <= k_range.y; k_range.x += raycaster.ray_step, pt = origin + (direction * k_range.x)) {		
			unsigned char sample = raycaster.volume.sample_data(pt);
			float4 color_cur = raycaster.transfer_fn[sample / TF_RATIO];
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
		raycaster.write_color(color_acc, pos, dev_buffer);
	}
}

void GPURenderer1::set_transfer_fn(Raycaster r) {
	if (transfer_fn == NULL) 
		cuda_safe_call(cudaMalloc((void **)&transfer_fn, TF_SIZE * sizeof(float4)));
	if (esl_volume == NULL) 
		cuda_safe_call(cudaMalloc((void **)&esl_volume, ESL_VOLUME_SIZE * sizeof(unsigned char)));
	cuda_safe_call(cudaMemcpy(transfer_fn, r.transfer_fn, TF_SIZE * sizeof(float4), cudaMemcpyHostToDevice));
	cuda_safe_call(cudaMemcpy(esl_volume, r.esl_volume, ESL_VOLUME_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));
}

void GPURenderer1::set_window_buffer(View view) {
	if (dev_buffer != NULL)
		cuda_safe_call(cudaFree(dev_buffer));
	dev_buffer_size = view.size_px.x * view.size_px.y * 4;
	cuda_safe_call(cudaMalloc((void **)&dev_buffer, dev_buffer_size));
	num_blocks = dim3((view.size_px.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					  (view.size_px.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
}

void GPURenderer1::set_volume(Model volume) {
	if (dev_volume_data != NULL)
		cuda_safe_call(cudaFree(dev_volume_data));
	cuda_safe_call(cudaMalloc((void **)&dev_volume_data, volume.size));
	cuda_safe_call(cudaMemcpy(dev_volume_data, volume.data, volume.size, cudaMemcpyHostToDevice));
}

int GPURenderer1::render_volume(uchar4 *buffer, Raycaster r) {
	r.volume.data = dev_volume_data;
	r.transfer_fn = transfer_fn;
	r.esl_volume = esl_volume;

	cuda_safe_call(cudaMemset(dev_buffer, 0, dev_buffer_size));
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(r, dev_buffer);
	cuda_safe_check();
	cuda_safe_call(cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost));
	return 0;
}