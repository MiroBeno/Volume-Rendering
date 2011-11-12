#ifndef _PROJECTION_H_
#define _PROJECTION_H_

#include "data_utils.h"

#define WIN_WIDTH 512
#define WIN_HEIGHT 512
#define VIRTUAL_VIEW_SIZE 3.5f

struct Ortho_view {
	int2 size_px;
	int2 half_px;
	float3 origin; 
	float3 direction;  
	float3 right_plane;
	float3 up_plane; 

	__host__ __device__ void get_view_ray(int col, int row, float3 *origin_vector, float3 *direction_vector) {
		*direction_vector = direction;
		*origin_vector = origin + (right_plane * (float) (col - half_px.x));
		*origin_vector = *origin_vector + (up_plane * (float) (row - half_px.y));
	}
};

float3 camera_up();
float3 camera_down();
float3 camera_left();
float3 camera_right();
float3 camera_zoom_in();
float3 camera_zoom_out();
float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle);

void init_view(int width_px, int height_px, float size);
Ortho_view get_view();

#endif