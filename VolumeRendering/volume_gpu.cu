// Standard CUDA implementation

#include "data_utils.h"
#include "projection.h"
#include "model.h"

const int BUFFER_SIZE_CUDA = WIN_WIDTH * WIN_HEIGHT * 4;

static Volume_model volume;
static unsigned char *dev_buffer;

static cudaEvent_t start, stop; 
static float elapsedTime;

__device__ float2 intersect_1D_gpu(float pt, float dir, float min_bound, float max_bound) {
	if (dir == 0) {											// ak je zlozka vektora rovnobezna so stenou kocky
		if ((pt < min_bound) || (pt > max_bound))			// ak nelezi bod v romedzi kocky v danej osi
			return make_float2(POS_INF, NEG_INF);			// interval bude nulovy
		else
			return make_float2(NEG_INF, POS_INF);			// inak interval bude nekonecny
	}
	float k1 = (min_bound - pt) / dir;
	float k2 = (max_bound - pt) / dir;
	return k1 <= k2 ? make_float2(k1, k2) : make_float2(k2, k1); // skontroluj opacny vektor
}

__device__ float2 intersect_3D_gpu(float3 pt, float3 dir, float3 min_bound, float3 max_bound) {
	float2 xRange = intersect_1D_gpu(pt.x, dir.x, min_bound.x, max_bound.x);
	float2 yRange = intersect_1D_gpu(pt.y, dir.y, min_bound.y, max_bound.y);
	float2 zRange = intersect_1D_gpu(pt.z, dir.z, min_bound.z, max_bound.z);
	float k1 = xRange.x, k2 = xRange.y;
	if (yRange.x > k1) k1 = yRange.x;
	if (zRange.x > k1) k1 = zRange.x;
	if (yRange.y < k2) k2 = yRange.y;
	if (zRange.y < k2) k2 = zRange.y;
	return make_float2(k1, k2);					
}

__global__ void render_ray_gpu(Volume_model volume, Ortho_view view, unsigned char dev_buffer[]) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ((col >= view.size_px.x) || (row >= view.size_px.y))					// ak su rozmery okna nedelitelne 16, spustaju sa prazdne thready
		return;

	float bg = (((col / 16) + (row / 16)) % 2) * 0.1f;
	float4 bg_color = {bg, bg, bg, 1};
	float4 color_acc;

	float3 origin = {0,0,0}, direction = {0,0,0};
	view.get_view_ray(col, row, &origin, &direction);
	float2 k_range = intersect_3D_gpu(origin, direction, volume.min_bound, volume.max_bound);

	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
			k_range.x = 0;
		color_acc = make_float4(0,0,0,0);
		for (float k = k_range.x; k <= k_range.y; k += volume.ray_step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = volume.sample_color(pt);
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

	int offset = (row * WIN_WIDTH + col) * 4;
	dev_buffer[offset + 0] = map_float_int(color_acc.x,256);
	dev_buffer[offset + 1] = map_float_int(color_acc.y,256);
	dev_buffer[offset + 2] = map_float_int(color_acc.z,256);
	dev_buffer[offset + 3] = 255;
}

extern void init_gpu(Volume_model volume_model) {
	volume = volume_model;
	unsigned char *dev_volume_data;
	cudaMalloc((void **)&dev_volume_data, volume.size);
	cudaMemcpy(dev_volume_data, volume.data, volume.size, cudaMemcpyHostToDevice);
	volume.data = dev_volume_data;
	cudaMalloc((void **)&dev_buffer, BUFFER_SIZE_CUDA);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

extern void free_gpu(void) {
	cudaFree(dev_buffer);
	cudaFree(volume.data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

extern float render_volume_gpu(unsigned char *buffer, Ortho_view ortho_view) {
	int threads_dim = 16;
	dim3 threads_per_block(threads_dim, threads_dim);				// podla occupancy calculator
	dim3 num_blocks((WIN_WIDTH + threads_dim - 1) / threads_dim, (WIN_HEIGHT + threads_dim - 1) / threads_dim);		// celociselne delenie, 
																													// ak su rozmery okna nedelitelne 16, spustaju sa bloky	s nevyuzitimi threadmi
	cudaEventRecord(start, 0);
	render_ray_gpu<<<num_blocks, threads_per_block>>>(volume, ortho_view, dev_buffer);
	cudaMemcpy(buffer, dev_buffer, BUFFER_SIZE_CUDA, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}