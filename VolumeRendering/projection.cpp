//#include <stdio.h>
#include "projection.h"

const float max_cam_distance = 10, min_cam_distance = 0.1f;
const float d_distance = 0.1f, d_vert_angle = DEG_TO_RAD(5.0f), d_horiz_angle = DEG_TO_RAD(10.0f);

static float cam_distance = 2, cam_vert_angle = 0, cam_horiz_angle = 0;
/*static*/ float3 cam_position = {2, 0, 0};
/*static*/ int view_width_half_px = WIN_WIDTH / 2, view_height_half_px = WIN_HEIGHT / 2;
static float view_px_step = 3.5f / MINIMUM(WIN_WIDTH, WIN_HEIGHT);
/*static*/ float3 view_vector = {-1, 0, 0}, view_right_plane = {0, 0, 1}, view_up_plane = {0, 1, 0};

float3 compute_camera_position() {
	float tmp = cam_distance * cos(cam_vert_angle);
	cam_position.x = tmp * cos(cam_horiz_angle);
	cam_position.y = cam_distance * sin(cam_vert_angle);
	cam_position.z = tmp * sin(cam_horiz_angle);
	//printf("vert_angle: %4.2f  horiz_angle: %4.2f     cam_position: %4.2f %4.2f %4.2f\n", cam_vert_angle / PI * 180, cam_horiz_angle / PI * 180, cam_position.x, cam_position.y, cam_position.z);
	return cam_position;
}

float3 camera_up() {
	cam_vert_angle = cam_vert_angle - d_vert_angle;
	while (cam_vert_angle < -PI) 
		cam_vert_angle += 2 * PI;
	return compute_camera_position();
}

float3 camera_down() {
	cam_vert_angle = cam_vert_angle + d_vert_angle;
	while (cam_vert_angle >= PI) 
		cam_vert_angle -= 2 * PI;
	return compute_camera_position();
}

float3 camera_left() {
	cam_horiz_angle -= d_horiz_angle; 
	while (cam_horiz_angle < 0) 
		cam_horiz_angle += 2 * PI;
	return compute_camera_position();
}

float3 camera_right() {
	cam_horiz_angle += d_horiz_angle; 
	while (cam_horiz_angle >= 2 * PI) 
		cam_horiz_angle -= 2 * PI;
	return compute_camera_position();
}

float3 camera_zoom_in() {
	cam_distance = MAXIMUM(cam_distance - d_distance, min_cam_distance);
	return compute_camera_position();
}

float3 camera_zoom_out() {
	cam_distance = MINIMUM(cam_distance + d_distance, max_cam_distance);
	return compute_camera_position();
}

float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle) {
	cam_distance = distance;
	cam_vert_angle = DEG_TO_RAD(vert_angle);
	cam_horiz_angle = DEG_TO_RAD(horiz_angle);
	return compute_camera_position();
}

float3 get_camera_position() {
	return cam_position;
}

void init_view(int width_px, int height_px, float size) {
	view_width_half_px = width_px / 2;
	view_height_half_px = height_px / 2;
	view_px_step = size / MINIMUM(width_px, height_px);			
	view_vector = -cam_position;						// pozicia kamery je uprostred viewu (pozicia kamery), smerujeme vzdy do [0,0,0] staci vynasobit -1 (= 0 - center)
	view_vector = vector_normalize(view_vector);

	if ((view_vector.x == 0) && (view_vector.z == 0))					// specialny pripad pri vektore rovnobeznom s osou y (nemozme pouzit vektor y na vypocet kolmeho vektora)
		view_vector.x = 0.000001f;
	
	float3 y_vector = {0, 1, 0}; 
	if ((cam_vert_angle >= -PI/2) && (cam_vert_angle <= PI/2)) 
		y_vector = -y_vector;
	view_right_plane = cross_product(y_vector, view_vector);		// pravidlo pravej ruky v pravotocivej sustave, vektorove nasobenie vrati pravy uhol na dva vektory
	view_up_plane = cross_product(view_right_plane, view_vector);

	//printf("view vector:%4.2f %4.2f %4.2f\nup vector:%4.2f %4.2f %4.2f\nright vector:%4.2f %4.2f %4.2f\n\n", view_vector.x, view_vector.y, view_vector.z, view_up_plane.x, view_up_plane.y, view_up_plane.z, view_right_plane.x, view_right_plane.y, view_right_plane.z);

	view_right_plane = vector_normalize(view_right_plane);		
	view_up_plane = vector_normalize(view_up_plane);			
	view_right_plane = view_right_plane * view_px_step;	
	view_up_plane = view_up_plane * view_px_step;
}

void get_view_ray(int row, int col, float3 *origin, float3 *direction) {
	*direction = view_vector;
	*origin = cam_position + (view_right_plane * (float) (col - view_width_half_px));
	*origin = *origin + (view_up_plane * (float) (row - view_height_half_px));
}


