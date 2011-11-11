//#include <stdio.h>

#include "data_types.h"
#include "math_constants.h"

extern int view_width_half_px, view_height_half_px;
extern float3 cam_position, view_vector, view_right_plane, view_up_plane;

extern int volume_size_x, volume_size_y, volume_size_z;	
extern size_t volume_size_bytes;
extern unsigned char *volume;
extern float3 min_bound, max_bound;
extern float step;		

const int DATA_SIZE_CUDA = WIN_WIDTH * WIN_HEIGHT * 4;

unsigned char *dev_volume;
unsigned char *dev_col_buffer;
unsigned char col_buffer[DATA_SIZE_CUDA];

__device__ float4 sample_color_cuda(float3 point) {
	float4 color = {(point.x+1)*0.5f, (point.y+1)*0.5f, (point.z+1)*0.5f, 0.1f};
	return color;	
	//point = (point + make_float3(1,1,1)) * 0.5f;	
	//return make_float4(point.x, point.y, point.z, 0.5f);	
}

__device__ float2 intersect_1D_cuda(float pt, float dir, float min_bound, float max_bound) {
	if (dir == 0) {											// ak je zlozka vektora rovnobezna so stenou kocky
		if ((pt < min_bound) || (pt > max_bound))			// ak nelezi bod v romedzi kocky v danej osi
			return make_float2(CUDART_MAX_NORMAL_F, CUDART_MIN_DENORM_F);			// interval bude nulovy
		else
			return make_float2(CUDART_MIN_DENORM_F, CUDART_MAX_NORMAL_F);			// inak interval bude nekonecny
	}
	float k1 = (min_bound - pt) / dir;
	float k2 = (max_bound - pt) / dir;
	return k1 <= k2 ? make_float2(k1, k2) : make_float2(k2, k1); // skontroluj opacny vektor
}

__device__ float2 intersect_3D_cuda(float3 pt, float3 dir, float3 min_bound, float3 max_bound) {
	float2 xRange = intersect_1D_cuda(pt.x, dir.x, min_bound.x, max_bound.x);
	float2 yRange = intersect_1D_cuda(pt.y, dir.y, min_bound.y, max_bound.y);
	float2 zRange = intersect_1D_cuda(pt.z, dir.z, min_bound.z, max_bound.z);
	float k1 = xRange.x, k2 = xRange.y;
	if (yRange.x > k1) k1 = yRange.x;
	if (zRange.x > k1) k1 = zRange.x;
	if (yRange.y < k2) k2 = yRange.y;
	if (zRange.y < k2) k2 = zRange.y;
	return make_float2(k1, k2);					
}

__global__ void render_ray_cuda(float3 min_bound, float3 max_bound, float step, float3 origin, float3 direction, float3 view_right_plane, float3 view_up_plane, unsigned char dev_col_buffer[]) {
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	float bg = (((col / 16) + (row / 16)) % 2) * 0.3;
	float4 bg_color = {bg, bg, bg, 1};
	float4 color_acc = bg_color;									
	origin = origin + (view_right_plane * (float) (col - (WIN_WIDTH / 2)));
	origin = origin + (view_up_plane * (float) (row - (WIN_HEIGHT / 2)));
	float2 k_range = intersect_3D_cuda(origin, direction, min_bound, max_bound);
	if ((k_range.x < k_range.y) && (k_range.y > 0)) {				// nenulovy interval koeficientu k (existuje priesecnica) A vystupny bod lezi na luci
		if ((k_range.x < 0))										// bod vzniku luca je vnutri kocky, zaciname nie vstupnym priesecnikom, ale bodom vzniku
			k_range.x = 0;
		color_acc = make_float4(0,0,0,0);
		for (float k = k_range.x; k <= k_range.y; k += step) {		
			float3 pt = origin + (direction * k);
			float4 color_cur = sample_color_cuda(pt);
			color_cur.x *= color_cur.w;								// transparency formula: C_out = C_in + C * (1-alpha_in); alpha_out = aplha_in + alpha * (1-alpha_in)
			color_cur.y *= color_cur.w;
			color_cur.z *= color_cur.w;
			color_acc = color_acc + (color_cur * (1 - color_acc.w));
			if (color_acc.w > 0.95f) 
				break;
		}
		color_acc = color_acc + (bg_color * (1 - color_acc.w));	
	}
	dev_col_buffer[(row*WIN_WIDTH + col)*4] = map_float_int(color_acc.x,256);
	dev_col_buffer[(row*WIN_WIDTH + col)*4+1] = map_float_int(color_acc.y,256);
	dev_col_buffer[(row*WIN_WIDTH + col)*4+2] = map_float_int(color_acc.z,256);
	dev_col_buffer[(row*WIN_WIDTH + col)*4+3] = 255;
}

extern void init_cuda(void) {
	cudaMalloc((void **)&dev_volume, volume_size_bytes);
	cudaMalloc((void **)&dev_col_buffer, DATA_SIZE_CUDA);			
}

extern void free_cuda(void) {
	cudaFree(dev_col_buffer);
	cudaFree(dev_volume);
}

extern unsigned char *run_kernel(void) {
	//printf("c_p:%4.2f  %4.2f %4.2f\n", cam_position.x,  cam_position.y, cam_position.z);
	dim3 threadsPerBlock(16, 16);				// podla occupancy calculator
	dim3 numBlocks(WIN_WIDTH/threadsPerBlock.x, WIN_HEIGHT/threadsPerBlock.y);				//TODO: celociselne delenie !
	render_ray_cuda<<<numBlocks,threadsPerBlock>>>(min_bound, max_bound, step, cam_position, view_vector, view_right_plane, view_up_plane, dev_col_buffer);
	cudaMemcpy(&col_buffer, dev_col_buffer, DATA_SIZE_CUDA, cudaMemcpyDeviceToHost);
	return col_buffer;
}