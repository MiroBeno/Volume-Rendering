#include <stdio.h>

extern int view_width_half_px, view_height_half_px;
extern float3 cam_position, view_vector, view_right_plane, view_up_plane;

extern int volume_size_x, volume_size_y, volume_size_z;	
extern size_t volume_size_bytes;
extern unsigned char *volume;
extern float3 min_bound, max_bound;
extern float step;				

const int DATA_SIZE = 512*512*4;

unsigned char *dev_volume;
unsigned char *dev_col_buffer;
unsigned char col_buffer[DATA_SIZE];

__device__ float4 sample_color_cuda(float3 point) {
	float4 color;
	color.x = (point.x + 1) * 0.5;
	color.y = (point.y + 1) * 0.5;
	color.z = (point.z + 1) * 0.5;
	color.w = 0.5f;
	return color;			
}

__device__ void intersect1D_cuda(float bound_min, float bound_max, float o, float d, float2 *ival) {
	if (d == 0)	{										// ak je zlozka vektora rovnobezna so stenou kocky
		if ((o < bound_min) || (o > bound_max)) {			// ak nelezi bod v romedzi kocky v danej osi
			ival->x = 100;								// interval bude nulovy
			ival->y = -100;
		}
		return;											// inak interval neovplyvni
	}
	float k1 = (bound_min - o) / d;
	float k2 = (bound_max - o) / d;
	if (k1 > k2)  {										// pri opacnom vektore
		float pom = k1;
		k1 = k2;
		k2 = pom;
	}
	if (k1 > ival->x)									// orezanie intervalu zlava (max z k1)
		ival->x = k1;
	if (k2 < ival->y)									// orezanie intervalu sprava (min z k2)
		ival->y = k2;
}

__global__ void render_ray_cuda(float3 min_bound, float3 max_bound, float step, float3 origin, float3 direction, float3 view_right_plane, float3 view_up_plane, unsigned char dev_col_buffer[]) {
	origin.x += view_right_plane.x *(float) (blockIdx.x - 256);
	origin.y += view_right_plane.y *(float) (blockIdx.x - 256);
	origin.z += view_right_plane.z *(float) (blockIdx.x - 256);
	origin.x += view_up_plane.x *(float) (threadIdx.x - 256);
	origin.y += view_up_plane.y *(float) (threadIdx.x - 256);
	origin.z += view_up_plane.z *(float) (threadIdx.x - 256);
	float2 k_range = {-100, 100};
	intersect1D_cuda(min_bound.x, max_bound.x, origin.x, direction.x, &k_range);
	intersect1D_cuda(min_bound.y, max_bound.y, origin.y, direction.y, &k_range);
	intersect1D_cuda(min_bound.z, max_bound.z, origin.z, direction.z, &k_range);
	if ((k_range.x > k_range.y) || (k_range.y <0))				
		return;//TODO bg_color;
	if ((k_range.x < 0))										
		k_range.x = 0;
	float4 color_acc = {255,0,0,0}, color_cur = {0,0,0,0};
	float3 pnt = {0,0,0};
	for (float k = k_range.x; k <= k_range.y; k += step) {		
		pnt.x = origin.x + direction.x * k;
		pnt.y = origin.y + direction.y * k;
		pnt.z = origin.z + direction.z * k;
		color_cur = sample_color_cuda(pnt);
		color_cur.x *= color_cur.w;								
		color_cur.y *= color_cur.w;
		color_cur.z *= color_cur.w;
		color_acc.x += color_cur.x * (1-color_acc.w);
		color_acc.y += color_cur.y * (1-color_acc.w);
		color_acc.z += color_cur.z * (1-color_acc.w);
		color_acc.w += color_cur.w * (1-color_acc.w);
		if (color_acc.w > 0.95f) 
			break;
	}
	dev_col_buffer[(blockIdx.x*blockDim.x + threadIdx.x)*4] = color_acc.x;
	dev_col_buffer[(blockIdx.x*blockDim.x + threadIdx.x)*4+1] = color_acc.y;
	dev_col_buffer[(blockIdx.x*blockDim.x + threadIdx.x)*4+2] = color_acc.z;
	dev_col_buffer[(blockIdx.x*blockDim.x + threadIdx.x)*4+3] = 255;
}

extern void init_cuda(void) {
	cudaMalloc((void **)&dev_volume, volume_size_bytes);
	cudaMalloc((void **)&dev_col_buffer, DATA_SIZE);
}

extern void free_cuda(void) {
	cudaFree(dev_col_buffer);
	cudaFree(dev_volume);
}

extern unsigned char *run_kernel(void) {
	
	//printf("c_p:%4.2f  %4.2f %4.2f\n", cam_position.x,  cam_position.y, cam_position.z);

	render_ray_cuda<<<512,512>>>(min_bound, max_bound, step, cam_position, view_vector, view_right_plane, view_up_plane, dev_col_buffer);
	cudaMemcpy(&col_buffer, dev_col_buffer, DATA_SIZE, cudaMemcpyDeviceToHost);
	return col_buffer;
}