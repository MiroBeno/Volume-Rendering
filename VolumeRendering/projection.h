#ifndef _PROJECTION_H_
#define _PROJECTION_H_

#include "data_utils.h"

#define WIN_WIDTH 1024
#define WIN_HEIGHT 1024

struct View {
	int2 size_px;
	int2 half_px;
	float3 origin; 
	float3 direction;  
	float3 right_plane;
	float3 up_plane; 
	bool perspective_ray;

	__host__ __device__ void get_ray(int2 pos, float3 *origin_vector, float3 *direction_vector) {
		if (perspective_ray) {
			*origin_vector = origin;
			*direction_vector = origin * 0.6f + (right_plane * (float) (pos.x - half_px.x));
			*direction_vector = *direction_vector + (up_plane * (float) (pos.y - half_px.y));
			*direction_vector = vector_normalize(*direction_vector + (-*origin_vector));
		}
		else {
			*direction_vector = direction;
			*origin_vector = origin + (right_plane * (float) (pos.x - half_px.x));
			*origin_vector = *origin_vector + (up_plane * (float) (pos.y - half_px.y));
		}
	}
};

float3 camera_down(float angle);
float3 camera_right(float angle);
float3 camera_zoom(float distance);
float3 camera_down(int pixels);
float3 camera_right(int pixels);
float3 camera_zoom(int pixels);
float3 set_camera_position(float distance, float vert_angle, float horiz_angle);
void toggle_perspective();
void set_window_size(int2 px);

View get_view();

#endif