#include "glew.h"
#include "ViewBase.h"

View ViewBase::view = {	{INT_WIN_WIDTH, INT_WIN_HEIGHT}, 
						{0, 0, 3},
						{0, 0, -1},
						make_float3(0, 0, -1) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						make_float3(0, 1, 0) * (3.0f / MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT)),
						{0, 0, 3},
						false
					 };

const float2 ViewBase::distance_limits = {0.1f, 3.0f};
float4 ViewBase::cam_pos = {0, 0, 3, 1};
float4 ViewBase::light_pos = {0, 0, 3, 1};	
float ViewBase::cam_matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
float ViewBase::light_matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
float ViewBase::pixel_ratio_rotation = 180.0f / (MINIMUM(INT_WIN_WIDTH, INT_WIN_HEIGHT));					// pixel na uhol rotacie kamery
float ViewBase::pixel_ratio_translation = (distance_limits.y - distance_limits.x) / (INT_WIN_HEIGHT / 2);	// pixel na vzdialenost kamery
float ViewBase::virtual_view_size = 3.0f;		// velkost virtualneho okna v priestore

float3 ViewBase::vector_rotate(float4 v, float rot_matrix[16]) {					
	float3 r;
	r.x = dot_product(v, make_float4(rot_matrix[0], rot_matrix[1], rot_matrix[2], rot_matrix[3]));
    r.y = dot_product(v, make_float4(rot_matrix[4], rot_matrix[5], rot_matrix[6], rot_matrix[7]));
    r.z = dot_product(v, make_float4(rot_matrix[8], rot_matrix[9], rot_matrix[10], rot_matrix[11]));
	return r;
}

void ViewBase::matrix_rotate(float matrix[], float3 angles, bool reset) {
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	if (reset) {
		glLoadIdentity();
		glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	}
	glLoadMatrixf(matrix);
	glRotatef(angles.x, matrix[0], matrix[4], matrix[8]);
    glRotatef(angles.y, matrix[1], matrix[5], matrix[9]);
	glRotatef(angles.z, matrix[2], matrix[6], matrix[10]);
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    glPopMatrix();
}

void ViewBase::update_view() {										//dlzka najvacsej hrany je 2 a stred kvadra v [0,0,0]
    view.origin = vector_rotate(cam_pos, cam_matrix);
	view.direction = vector_normalize(-view.origin);	
	float step_px = virtual_view_size / MINIMUM(view.dims.x, view.dims.y);
	view.right_plane = vector_rotate(make_float4(1, 0, 0, 0), cam_matrix) * step_px;
    view.up_plane = vector_rotate(make_float4(0, 1, 0, 0), cam_matrix) * step_px;
}

void ViewBase::camera_rotate(float3 angles, bool reset) {
	matrix_rotate(cam_matrix, angles, reset);
	update_view();
}

void ViewBase::camera_rotate(int2 pixels) {
	camera_rotate(make_float3(pixels.y * pixel_ratio_rotation, 
								pixels.x * pixel_ratio_rotation, 
								0));
}

void ViewBase::camera_rotate(int3 pixels) {
	camera_rotate(make_float3(pixels.y * pixel_ratio_rotation, 
								pixels.x * pixel_ratio_rotation, 
								pixels.z * pixel_ratio_rotation));
}

void ViewBase::camera_zoom(float distance) {
	cam_pos.z = CLAMP(cam_pos.z + distance, distance_limits.x, distance_limits.y);
	if (!view.perspective)
		virtual_view_size = cam_pos.z;
	update_view();
}

void ViewBase::camera_zoom(int pixels) {
	camera_zoom(pixels * pixel_ratio_translation);
}

void ViewBase::set_camera_position(float3 angles, float distance) {
	cam_pos.z = 0;
	camera_zoom(distance);
	camera_rotate(angles, true);
}

void ViewBase::light_rotate(int2 pixels) {
	matrix_rotate(light_matrix, 
					make_float3(pixels.y * pixel_ratio_rotation, 
								pixels.x * pixel_ratio_rotation, 
								0),
					false);
	view.light_pos = vector_rotate(light_pos, light_matrix);
}

void ViewBase::toggle_perspective(int update_mode) {
	if (!update_mode)
		view.perspective = !view.perspective;
	virtual_view_size = view.perspective ? 1.5f : cam_pos.z;
	update_view();
}

void ViewBase::set_viewport_dims(ushort2 dims, float scale) {
	view.dims.x = (unsigned short) (dims.x * scale);
	view.dims.y = (unsigned short) (dims.y * scale);
	pixel_ratio_rotation = 180.0f / (MINIMUM(dims.x, dims.y));	
	pixel_ratio_translation = (distance_limits.y - distance_limits.x) / (dims.y / 2);
	update_view();
}





