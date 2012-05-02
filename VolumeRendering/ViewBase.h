#ifndef _VIEW_H_
#define _VIEW_H_
#include "common.h"

#define INT_WIN_WIDTH 800
#define INT_WIN_HEIGHT 680

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
		static float cam_matrix[16];
		static float light_matrix[16];
		static void update_view();
		static void camera_rotate(float3 angles, bool reset = false);
		static void camera_rotate(int3 pixels);
		static void camera_rotate(int2 pixels);
		static void camera_zoom(float distance);
		static void camera_zoom(int pixels);
		static void set_camera_position(float3 angles, float distance = 3.0f);
		static void light_rotate(int2 pixels);
		static void toggle_perspective(int update_mode);
		static void set_viewport_dims(ushort2 dims, float scale = 1.0f);
	private:
		static void matrix_rotate(float matrix[], float3 angles, bool reset);
		static float3 vector_rotate(float4 v, float rot_matrix[16]);
		static const float2 distance_limits;
		static float4 cam_pos;
		static float4 light_pos;
		static float pixel_ratio_rotation;
		static float pixel_ratio_translation;
		static float virtual_view_size;
};

#endif