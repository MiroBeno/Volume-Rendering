#ifndef _VIEW_H_
#define _VIEW_H_
#include "data_utils.h"

#define INT_WIN_WIDTH 1024
#define INT_WIN_HEIGHT 1024

struct View {
	ushort2 size_px;
	float3 origin; 
	float3 direction;  
	float3 right_plane;
	float3 up_plane; 
	bool perspective;

	__host__ __device__ void get_ray(short2 pos, float3 *origin_vector, float3 *direction_vector) {
		if (perspective) {
			*origin_vector = origin;
			*direction_vector = direction + (right_plane * (float) (pos.x - size_px.x / 2));
			*direction_vector = *direction_vector + (up_plane * (float) (pos.y - size_px.y / 2));
			//*direction_vector = vector_normalize(*direction_vector);
		}
		else {
			*direction_vector = direction;
			*origin_vector = origin + (right_plane * (float) (pos.x - size_px.x / 2));
			*origin_vector = *origin_vector + (up_plane * (float) (pos.y - size_px.y / 2));
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
		static void set_window_size(ushort2 px);
	private:
		static const float2 distance_limits;
		static float2 cam_angles;
		static float cam_distance;
		static float3 cam_position;
		static float2 cam_pixel_delta;
		static float virtual_view_size;
		static void update_view();
		static float3 compute_camera_position();
};

#endif