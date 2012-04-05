#include <stdio.h>
#include "projection.h"

//dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
const float2 distance_limits = {0.1f, 10};

static float2 cam_angles = {0, 0};			// x - horizontalny uhol, y - vertikalny uhol
static float cam_distance = 2;
static float3 cam_position = {2, 0, 0};

static float cam_pixel_delta = 180.0f / (MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT));
static float virtual_view_size = 3.0f;		// velkost virtualneho okna v priestore

static View view = {	{INT_WIN_WIDTH, INT_WIN_HEIGHT}, 
						{INT_WIN_WIDTH / 2, INT_WIN_HEIGHT / 2}, 
						{2, 0, 0},
						{-1, 0, 0},
						make_float3(0, 0, -1) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						make_float3(0, 1, 0) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						false
					 };

void update_view() {	
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
	float step_px = virtual_view_size / MINIMUM(view.size_px.x, view.size_px.y);
	view.right_plane = view.right_plane * step_px;	
	view.up_plane = view.up_plane * step_px;
}

float3 compute_camera_position() {
	float tmp = cam_distance * cos(cam_angles.y);
	cam_position.x = tmp * cos(cam_angles.x);
	cam_position.y = cam_distance * sin(cam_angles.y);
	cam_position.z = tmp * sin(cam_angles.x);
	//printf("vert_angle: %4.2f  horiz_angle: %4.2f     cam_position: %4.2f %4.2f %4.2f\n", cam_angles.y / PI * 180, cam_angles.x / PI * 180, cam_position.x, cam_position.y, cam_position.z);
	update_view();
	return cam_position;
}

float3 camera_down(float angle) {
	cam_angles.y += DEG_TO_RAD(angle);
	while (cam_angles.y < -PI) 
		cam_angles.y += 2 * PI;
	while (cam_angles.y >= PI) 
		cam_angles.y -= 2 * PI;
	return compute_camera_position();
}

float3 camera_right(float angle) {
	cam_angles.x += DEG_TO_RAD(angle); 
	while (cam_angles.x < 0) 
		cam_angles.x += 2 * PI;
	while (cam_angles.x >= 2 * PI) 
		cam_angles.x -= 2 * PI;
	return compute_camera_position();
}

float3 camera_zoom(float distance) {
	cam_distance = CLAMP(cam_distance + distance, distance_limits.x, distance_limits.y);
	return compute_camera_position();
}

float3 camera_down(int pixels) {
	return camera_down(pixels * cam_pixel_delta);
}

float3 camera_right(int pixels) {
	return camera_right(pixels * cam_pixel_delta);
}

float3 camera_zoom(int pixels) {
	return camera_zoom(pixels * cam_pixel_delta);
}

float3 set_camera_position(float distance, float vert_angle, float horiz_angle) {
	cam_distance = distance;
	cam_angles.y = DEG_TO_RAD(vert_angle);
	cam_angles.x = DEG_TO_RAD(horiz_angle);
	return compute_camera_position();
}

void toggle_perspective() {
	view.perspective_ray = !view.perspective_ray;
	if (view.perspective_ray) 
		virtual_view_size = 2.0f;
	else
		virtual_view_size = 3.0f;
	printf("Perspective: %s\n", view.perspective_ray ? "on" : "off");
	update_view();
}

void set_window_size(int2 px) {
	view.size_px = px; 
	view.half_px.x = px.x / 2; 
	view.half_px.y = px.y / 2;
	cam_pixel_delta = 180.0f / (MINIMUM(px.x, px.y));
	update_view();
}

View get_view() {
	return view;
}




