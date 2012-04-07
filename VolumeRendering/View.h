#ifndef _VIEW_H_
#define _VIEW_H_

#include "data_utils.h"

#define INT_WIN_WIDTH 1024
#define INT_WIN_HEIGHT 1024

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

class ViewBase {
	public:
		static View view;
		static float3 camera_down(float angle);
		static float3 camera_right(float angle);
		static float3 camera_zoom(float distance);
		static float3 camera_down(int pixels);
		static float3 camera_right(int pixels);
		static float3 camera_zoom(int pixels);
		static float3 set_camera_position(float distance, float vert_angle, float horiz_angle);
		static void toggle_perspective();
		static void set_window_size(int2 px);
	private:
		static const float2 distance_limits;
		static float2 cam_angles;
		static float cam_distance;
		static float3 cam_position;
		static float cam_pixel_delta;
		static float virtual_view_size;
		static void update_view();
		static float3 compute_camera_position();
};

#endif