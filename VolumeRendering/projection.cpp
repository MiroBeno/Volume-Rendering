//#include <stdio.h>
#include "projection.h"

const float2 distance_limits = {10, 0.1f};

static float2 cam_angles = {0,0};			// x - horizontalny uhol, y - vertikalny uhol
static float cam_distance = 2; 
static float3 cam_position = {2, 0, 0};

static Ortho_view view = {	{WIN_WIDTH, WIN_HEIGHT}, 
							{WIN_WIDTH / 2, WIN_HEIGHT / 2}, 
							{2, 0, 0},
							{-1, 0, 0},
							make_float3(0, 0, -1) * (VIRTUAL_VIEW_SIZE / MINIMUM(WIN_WIDTH, WIN_HEIGHT)),
							make_float3(0, 1, 0) * (VIRTUAL_VIEW_SIZE / MINIMUM(WIN_WIDTH, WIN_HEIGHT)),
						 };

float3 compute_camera_position() {
	float tmp = cam_distance * cos(cam_angles.y);
	cam_position.x = tmp * cos(cam_angles.x);
	cam_position.y = cam_distance * sin(cam_angles.y);
	cam_position.z = tmp * sin(cam_angles.x);
	//printf("vert_angle: %4.2f  horiz_angle: %4.2f     cam_position: %4.2f %4.2f %4.2f\n", cam_angles.y / PI * 180, cam_angles.x / PI * 180, cam_position.x, cam_position.y, cam_position.z);
	return cam_position;
}

float3 camera_up(float angle) {
	cam_angles.y -= DEG_TO_RAD(angle);
	while (cam_angles.y < -PI) 
		cam_angles.y += 2 * PI;
	return compute_camera_position();
}

float3 camera_down(float angle) {
	cam_angles.y += DEG_TO_RAD(angle);
	while (cam_angles.y >= PI) 
		cam_angles.y -= 2 * PI;
	return compute_camera_position();
}

float3 camera_left(float angle) {
	cam_angles.x -= DEG_TO_RAD(angle); 
	while (cam_angles.x < 0) 
		cam_angles.x += 2 * PI;
	return compute_camera_position();
}

float3 camera_right(float angle) {
	cam_angles.x += DEG_TO_RAD(angle); 
	while (cam_angles.x >= 2 * PI) 
		cam_angles.x -= 2 * PI;
	return compute_camera_position();
}

float3 camera_zoom_in(float distance) {
	cam_distance = MAXIMUM(cam_distance - distance, distance_limits.y);
	return compute_camera_position();
}

float3 camera_zoom_out(float distance) {
	cam_distance = MINIMUM(cam_distance + distance, distance_limits.x);
	return compute_camera_position();
}

float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle) {
	cam_distance = distance;
	cam_angles.y = DEG_TO_RAD(vert_angle);
	cam_angles.x = DEG_TO_RAD(horiz_angle);
	return compute_camera_position();
}

void update_view(int width_px, int height_px, float virtual_size) {
	view.size_px.x = width_px; 
	view.size_px.y = height_px;
	view.half_px.x = width_px / 2; 
	view.half_px.y = height_px / 2;	
	view.origin = cam_position;					
	view.direction = vector_normalize(-view.origin);				// pozicia kamery je uprostred viewu (pozicia kamery), smerujeme vzdy do [0,0,0] staci vynasobit -1 (= 0 - center)

	if ((view.direction.x == 0) && (view.direction.z == 0))			// specialny pripad pri vektore rovnobeznom s osou y (nemozme pouzit vektor y na vypocet kolmeho vektora)
		view.direction.x = 0.0001f;
	
	float3 y_vector = {0, 1, 0}; 
	if ((cam_angles.y >= -PI/2) && (cam_angles.y <= PI/2)) 
		y_vector = -y_vector;
	view.right_plane = cross_product(y_vector, view.direction);		// pravidlo pravej ruky v pravotocivej sustave, vektorove nasobenie vrati pravy uhol na dva vektory
	view.up_plane = cross_product(view.right_plane, view.direction);

	//printf("view vector:%4.2f %4.2f %4.2f\nup vector:%4.2f %4.2f %4.2f\nright vector:%4.2f %4.2f %4.2f\n\n", view_vector.x, view_vector.y, view_vector.z, view_up_plane.x, view_up_plane.y, view_up_plane.z, view_right_plane.x, view_right_plane.y, view_right_plane.z);

	view.right_plane = vector_normalize(view.right_plane);		
	view.up_plane = vector_normalize(view.up_plane);	
	float step_px = virtual_size / MINIMUM(width_px, height_px);
	view.right_plane = view.right_plane * step_px;	
	view.up_plane = view.up_plane * step_px;
}

Ortho_view get_view() {
	return view;
}




