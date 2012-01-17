#ifndef _PROJECTION_H_
#define _PROJECTION_H_

#include "data_utils.h"

#define WIN_WIDTH 1024
#define WIN_HEIGHT 1024
#define VIRTUAL_VIEW_SIZE 3.0f

struct View {
	int2 size_px;
	int2 half_px;
	float3 origin; 
	float3 direction;  
	float3 right_plane;
	float3 up_plane; 

	__host__ __device__ void get_ortho_ray(int col, int row, float3 *origin_vector, float3 *direction_vector) {
		*direction_vector = direction;
		*origin_vector = origin + (right_plane * (float) (col - half_px.x));
		*origin_vector = *origin_vector + (up_plane * (float) (row - half_px.y));
	}

	__host__ __device__ void get_perspective_ray(int col, int row, float3 *origin_vector, float3 *direction_vector) {
		*origin_vector = origin;
		*direction_vector = origin * 0.6f + (right_plane * (float) (col - half_px.x));
		*direction_vector = *direction_vector + (up_plane * (float) (row - half_px.y));
		*direction_vector = vector_normalize(*direction_vector + (-*origin_vector));
	}
};

float3 camera_up(float angle);
float3 camera_down(float angle);
float3 camera_left(float angle);
float3 camera_right(float angle);
float3 camera_zoom_in(float distance);
float3 camera_zoom_out(float distance);
float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle);

void update_view(int width_px, int height_px, float size);
View get_view();

#endif