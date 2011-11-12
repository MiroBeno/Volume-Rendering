//#include <stdio.h>
#include "projection.h"

const float2 distance_limits = {10, 0.1f};
const float2 d_angles = {DEG_TO_RAD(10.0f), DEG_TO_RAD(5.0f)};
const float d_distance = 0.1f;

static float2 cam_angles = {0,0};			// x - horizontalny uhol, y - vertikalny uhol
static float cam_distance = 2; 
/*static*/ float3 cam_position = {2, 0, 0};

static int2 view_half_px = {WIN_WIDTH / 2, WIN_HEIGHT / 2};
static float view_px_step = 3.5f / MINIMUM(WIN_WIDTH, WIN_HEIGHT);
/*static*/ float3 view_vector = {-1, 0, 0}, view_right_plane = {0, 0, 1}, view_up_plane = {0, 1, 0};

float3 compute_camera_position() {
	float tmp = cam_distance * cos(cam_angles.y);
	cam_position.x = tmp * cos(cam_angles.x);
	cam_position.y = cam_distance * sin(cam_angles.y);
	cam_position.z = tmp * sin(cam_angles.x);
	//printf("vert_angle: %4.2f  horiz_angle: %4.2f     cam_position: %4.2f %4.2f %4.2f\n", cam_angles.y / PI * 180, cam_angles.x / PI * 180, cam_position.x, cam_position.y, cam_position.z);
	return cam_position;
}

float3 camera_up() {
	cam_angles.y -= d_angles.y;
	while (cam_angles.y < -PI) 
		cam_angles.y += 2 * PI;
	return compute_camera_position();
}

float3 camera_down() {
	cam_angles.y -= d_angles.y;
	while (cam_angles.y >= PI) 
		cam_angles.y -= 2 * PI;
	return compute_camera_position();
}

float3 camera_left() {
	cam_angles.x -= d_angles.x; 
	while (cam_angles.x < 0) 
		cam_angles.x += 2 * PI;
	return compute_camera_position();
}

float3 camera_right() {
	cam_angles.x += d_angles.x; 
	while (cam_angles.x >= 2 * PI) 
		cam_angles.x -= 2 * PI;
	return compute_camera_position();
}

float3 camera_zoom_in() {
	cam_distance = MAXIMUM(cam_distance - d_distance, distance_limits.y);
	return compute_camera_position();
}

float3 camera_zoom_out() {
	cam_distance = MINIMUM(cam_distance + d_distance, distance_limits.x);
	return compute_camera_position();
}

float3 set_camera_position_deg(float distance, float vert_angle, float horiz_angle) {
	cam_distance = distance;
	cam_angles.y = DEG_TO_RAD(vert_angle);
	cam_angles.x = DEG_TO_RAD(horiz_angle);
	return compute_camera_position();
}

void init_view(int width_px, int height_px, float size) {
	view_half_px.x = width_px / 2;
	view_half_px.y = height_px / 2;
	view_px_step = size / MINIMUM(width_px, height_px);			
	view_vector = -cam_position;						// pozicia kamery je uprostred viewu (pozicia kamery), smerujeme vzdy do [0,0,0] staci vynasobit -1 (= 0 - center)
	view_vector = vector_normalize(view_vector);

	if ((view_vector.x == 0) && (view_vector.z == 0))					// specialny pripad pri vektore rovnobeznom s osou y (nemozme pouzit vektor y na vypocet kolmeho vektora)
		view_vector.x = 0.0001f;
	
	float3 y_vector = {0, 1, 0}; 
	if ((cam_angles.y >= -PI/2) && (cam_angles.y <= PI/2)) 
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
	*origin = cam_position + (view_right_plane * (float) (col - view_half_px.x));
	*origin = *origin + (view_up_plane * (float) (row - view_half_px.y));
}


