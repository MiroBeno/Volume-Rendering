#include <stdio.h>
#include "View.h"

View ViewBase::view = {	{INT_WIN_WIDTH, INT_WIN_HEIGHT}, 
						{3, 0, 0},
						{-1, 0, 0},
						make_float3(0, 0, -1) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						make_float3(0, 1, 0) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						false
					 };

const float2 ViewBase::distance_limits = {0.1f, 3};

float2 ViewBase::cam_angles = {0, 0};			// x - horizontalny uhol, y - vertikalny uhol
float ViewBase::cam_distance = 3;
float3 ViewBase::cam_position = {3, 0, 0};

float2 ViewBase::cam_pixel_delta = {180.0f / (MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),					// pixel na uhol rotacie kamery
									(distance_limits.y - distance_limits.x) / (INT_WIN_HEIGHT / 2)};			// pixel na vzdialenost kamery
float ViewBase::virtual_view_size = 3.0f;		// velkost virtualneho okna v priestore

void ViewBase::update_view() {	
	view.origin = cam_position;										//dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
	view.direction = vector_normalize(-view.origin);				// pozicia kamery je uprostred viewu, smerujeme vzdy do [0,0,0] staci vynasobit -1 (= 0 - center)

	if ((view.direction.x == 0) && (view.direction.z == 0))			// specialny pripad pri vektore rovnobeznom s osou y (nemozme pouzit vektor y na vypocet kolmeho vektora)
		view.direction.x = 0.0001f;
	
	float3 y_vector = {0, 1, 0}; 
	if ((cam_angles.y > -PI/2) && (cam_angles.y < PI/2)) 
		y_vector = -y_vector;
	view.right_plane = cross_product(y_vector, view.direction);		// pravidlo pravej ruky v pravotocivej sustave, vektorove nasobenie vrati pravy uhol na dva vektory
	view.up_plane = cross_product(view.right_plane, view.direction);

	//printf("vd:%4.2f %4.2f %4.2f\nup vector:%4.2f %4.2f %4.2f\nright vector:%4.2f %4.2f %4.2f\n\n", view.direction.x, view.direction.y, view.direction.z, view.up_plane.x, view.up_plane.y, view.up_plane.z, view.right_plane.x, view.right_plane.y, view.right_plane.z);
	view.right_plane = vector_normalize(view.right_plane);		
	view.up_plane = vector_normalize(view.up_plane);	
	float step_px = virtual_view_size / MINIMUM(view.size_px.x, view.size_px.y);
	view.right_plane = view.right_plane * step_px;	
	view.up_plane = view.up_plane * step_px;
}

float3 ViewBase::compute_camera_position() {
	float tmp = cam_distance * cos(cam_angles.y);
	cam_position.x = tmp * cos(cam_angles.x);
	cam_position.y = cam_distance * sin(cam_angles.y);
	cam_position.z = tmp * sin(cam_angles.x);
	//printf("vert_angle: %4.2f  horiz_angle: %4.2f     cam_position: %4.2f %4.2f %4.2f\n", cam_angles.y / PI * 180, cam_angles.x / PI * 180, cam_position.x, cam_position.y, cam_position.z);
	update_view();
	return cam_position;
}

float3 ViewBase::camera_down(float angle) {
	cam_angles.y += DEG_TO_RAD(angle);
	while (cam_angles.y < -PI) 
		cam_angles.y += 2 * PI;
	while (cam_angles.y >= PI) 
		cam_angles.y -= 2 * PI;
	return compute_camera_position();
}

float3 ViewBase::camera_right(float angle) {
	cam_angles.x += DEG_TO_RAD(angle); 
	while (cam_angles.x < 0) 
		cam_angles.x += 2 * PI;
	while (cam_angles.x >= 2 * PI) 
		cam_angles.x -= 2 * PI;
	return compute_camera_position();
}

float3 ViewBase::camera_zoom(float distance) {
	cam_distance = CLAMP(cam_distance + distance, distance_limits.x, distance_limits.y);
	return compute_camera_position();
}

float3 ViewBase::camera_down(int pixels) {
	return camera_down(pixels * cam_pixel_delta.x);
}

float3 ViewBase::camera_right(int pixels) {
	return camera_right(pixels * cam_pixel_delta.x);
}

float3 ViewBase::camera_zoom(int pixels) {
	return camera_zoom(pixels * cam_pixel_delta.y);
}

float3 ViewBase::set_camera_position(float distance, float vert_angle, float horiz_angle) {
	cam_angles.y = 0;
	camera_down(vert_angle);
	cam_angles.x = 0;
	camera_right(horiz_angle);
	cam_distance = 0;
	return camera_zoom(distance);
}

void ViewBase::toggle_perspective() {
	view.perspective = !view.perspective;
	virtual_view_size = view.perspective ? 1.5f : 3.0f;
	printf("Perspective rays: %s\n", view.perspective ? "on" : "off");
	update_view();
}

void ViewBase::set_window_size(ushort2 px) {
	view.size_px = px; 
	cam_pixel_delta.x = 180.0f / (MINIMUM(px.x, px.y));
	cam_pixel_delta.y = (distance_limits.y - distance_limits.x) / (view.size_px.y / 2);
	update_view();
}





