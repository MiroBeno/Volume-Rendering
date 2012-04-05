// Standard CUDA implementation

#include "Renderer.h"

#include "cuda_runtime_api.h"

uchar4 *GPURenderer::dev_buffer = NULL;
int GPURenderer::dev_buffer_size = 0;
unsigned char *GPURenderer::dev_volume_data = NULL;
dim3 GPURenderer::THREADS_PER_BLOCK(16, 16);				// pocet threadov v bloku - podla occupancy calculator
dim3 GPURenderer::num_blocks(0, 0);

static float4 *transfer_fn = NULL;

GPURenderer1::GPURenderer1(int2 size, float4 *tf, Model volume, unsigned char *d) {
	set_window_buffer(size);
	set_transfer_fn(tf);
	set_volume(volume, d);
}

GPURenderer1::~GPURenderer1() {
	cudaFree(dev_buffer);
	cudaFree(dev_volume_data);
	cudaFree(transfer_fn);
}

__global__ void render_ray(Raycaster raycaster, uchar4 dev_buffer[], unsigned char dev_volume_data[], float4 transfer_fn[]) {
	int2 pos = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if ((pos.x >= raycaster.view.size_px.x) || (pos.y >= raycaster.view.size_px.y))	// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float4 color_acc = {0,0,0,0};
	float3 origin, direction;
	float2 k_range;
	raycaster.view.get_ray(pos, &origin, &direction); 

	if (raycaster.intersect(origin, direction, &k_range)) {				
		for (float k = k_range.x; k <= k_range.y; k += raycaster.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = raycaster.sample_color(dev_volume_data, transfer_fn, pt);
			color_acc = color_acc + (color_cur * (1 - color_acc.w)); // transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			if (color_acc.w > raycaster.ray_threshold) 
				break;
		}
	}
	raycaster.write_color(color_acc, pos, dev_buffer);
}

void GPURenderer1::set_transfer_fn(float4 *tf) {
	if (transfer_fn == NULL)
		cudaMalloc((void **)&transfer_fn, 256 * sizeof(float4));
	cudaMemcpy(transfer_fn, tf, 256 * sizeof(float4), cudaMemcpyHostToDevice);
}

void GPURenderer1::set_window_buffer(int2 size) {
	if (dev_buffer != NULL)
		cudaFree(dev_buffer);
	dev_buffer_size = size.x * size.y * 4;
	cudaMalloc((void **)&dev_buffer, dev_buffer_size);
	num_blocks = dim3((size.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, 
					  (size.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);		
			// celociselne delenie, ak su rozmery okna nedelitelne 16, spustaju sa bloky s nevyuzitimi threadmi
}

void GPURenderer1::set_volume(Model volume, unsigned char *d) {
	if (dev_volume_data != NULL)
		cudaFree(dev_volume_data);
	cudaMalloc((void **)&dev_volume_data, volume.size);
	cudaMemcpy(dev_volume_data, d, volume.size, cudaMemcpyHostToDevice);
}

int GPURenderer1::render_volume(uchar4 *buffer, Raycaster *r) {
	render_ray<<<num_blocks, THREADS_PER_BLOCK>>>(*r, dev_buffer, dev_volume_data, transfer_fn);
	cudaMemcpy(buffer, dev_buffer, dev_buffer_size, cudaMemcpyDeviceToHost);
	return 0;
}