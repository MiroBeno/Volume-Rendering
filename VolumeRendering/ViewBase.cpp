#include <stdio.h>
#include "View.h"

View ViewBase::view = {	{INT_WIN_WIDTH, INT_WIN_HEIGHT}, 
						{3, 0, 0},
						{-1, 0, 0},
						make_float3(0, 0, -1) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						make_float3(0, 1, 0) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						{3, 0, 0},
						false
					 };

const float2 ViewBase::distance_limits = {0.1f, 3};

float3 ViewBase::cam_position = {0, 0, 3};			// x - horizontalny uhol, y - vertikalny uhol, z - vzdialenost
float3 ViewBase::light_position = {0, 0, 3};	

float2 ViewBase::cam_pixel_delta = {180.0f / (MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),					// pixel na uhol rotacie kamery
									(distance_limits.y - distance_limits.x) / (INT_WIN_HEIGHT / 2)};			// pixel na vzdialenost kamery
float ViewBase::virtual_view_size = 3.0f;		// velkost virtualneho okna v priestore

void ViewBase::update_view() {										//dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
	view.direction = vector_normalize(-view.origin);				// pozicia kamery je uprostred viewu, smerujeme vzdy do [0,0,0] staci vynasobit -1 (= 0 - center)

	if ((view.direction.x == 0) && (view.direction.z == 0))			// specialny pripad pri vektore rovnobeznom s osou y (nemozme pouzit vektor y na vypocet kolmeho vektora)
		view.direction.x = 0.0001f;
	
	float3 y_vector = {0, 1, 0}; 
	if ((cam_position.y > -PI/2) && (cam_position.y < PI/2)) 
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

float3 ViewBase::angles_to_point(float3 *angles) {
	while (angles->x < -PI) angles->x += 2 * PI;
	while (angles->x >= PI) angles->x -= 2 * PI;
	while (angles->y < -PI) angles->y += 2 * PI;
	while (angles->y >= PI) angles->y -= 2 * PI;
	float tmp = angles->z * cos(angles->y);
	return make_float3(	tmp * cos(angles->x),
						angles->z * sin(angles->y),
						tmp * sin(angles->x));
}

void ViewBase::camera_down(float angle) {
	cam_position.y += DEG_TO_RAD(angle);
	view.origin = angles_to_point(&cam_position);
	update_view();
}

void ViewBase::camera_right(float angle) {
	cam_position.x += DEG_TO_RAD(angle); 
	view.origin = angles_to_point(&cam_position);
	update_view();
}

void ViewBase::camera_zoom(float distance) {
	cam_position.z = CLAMP(cam_position.z + distance, distance_limits.x, distance_limits.y);
	view.origin = angles_to_point(&cam_position);
	update_view();
}

void ViewBase::camera_down(int pixels) {
	camera_down(pixels * cam_pixel_delta.x);
}

void ViewBase::camera_right(int pixels) {
	camera_right(pixels * cam_pixel_delta.x);
}

void ViewBase::camera_zoom(int pixels) {
	camera_zoom(pixels * cam_pixel_delta.y);
}

void ViewBase::set_camera_position(float distance, float vert_angle, float horiz_angle) {
	cam_position.y = DEG_TO_RAD(vert_angle);
	cam_position.x = DEG_TO_RAD(horiz_angle);
	camera_zoom(distance);
}

void ViewBase::light_down(int pixels) {
	light_position.y += DEG_TO_RAD(pixels * cam_pixel_delta.x);
	view.light_pos = angles_to_point(&light_position);
}

void ViewBase::light_right(int pixels) {
	light_position.x += DEG_TO_RAD(pixels * cam_pixel_delta.x); 
	view.light_pos = angles_to_point(&light_position);
}

void ViewBase::toggle_perspective(int update_mode) {
	if (!update_mode)
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





