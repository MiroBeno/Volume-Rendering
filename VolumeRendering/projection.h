#ifndef _PROJECTION_H_
#define _PROJECTION_H_

#include "datatypes.h"

float3 camera_up();
float3 camera_down();
float3 camera_left();
float3 camera_right();
float3 camera_zoom_in();
float3 camera_zoom_out();
float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle);
float3 get_camera_position();

void init_view(int width_px, int height_px, float size);
void get_view_ray(int row, int col, float3 *origin, float3 *direction);

#endif