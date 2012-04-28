#ifndef _VIEW_H_
#define _VIEW_H_
#include "data_utils.h"

#define INT_WIN_WIDTH 1024
#define INT_WIN_HEIGHT 1024

struct View {
	ushort2 dims;
	float3 origin; 
	float3 direction;  
	float3 right_plane;
	float3 up_plane;
	float3 light_pos;
	bool perspective;

	__host__ __device__ void get_ray(short2 pos, float3 *origin_vector, float3 *direction_vector) {
		if (perspective) {
			*origin_vector = origin;
			*direction_vector = direction + (right_plane * (float) (pos.x - dims.x / 2));
			*direction_vector = *direction_vector + (up_plane * (float) (pos.y - dims.y / 2));
			//*direction_vector = vector_normalize(*direction_vector);
		}
		else {
			*direction_vector = direction;
			*origin_vector = origin + (right_plane * (float) (pos.x - dims.x / 2));
			*origin_vector = *origin_vector + (up_plane * (float) (pos.y - dims.y / 2));
		}
	}
};

class ViewBase {
	public:
		static View view;
		static void camera_down(float angle);
		static void camera_right(float angle);
		static void camera_zoom(float distance);
		static void camera_down(int pixels);
		static void camera_right(int pixels);
		static void camera_zoom(int pixels);
		static void light_down(int pixels);
		static void light_right(int pixels);
		static void set_camera_position(float distance, float vert_angle, float horiz_angle);
		static void toggle_perspective(int update_mode);
		static void set_viewport_dims(unsigned short viewport_x, unsigned short viewport_y, float viewport_scale = 1.0f);
	private:
		static const float2 distance_limits;
		static float3 cam_position;
		static float3 light_position;
		static float2 cam_pixel_delta;
		static float virtual_view_size;
		static void update_view();
		static float3 angles_to_point(float3 *angles);
};

#endif