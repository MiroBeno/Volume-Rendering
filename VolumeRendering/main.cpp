#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "glew.h"
#include "glut.h"

#include "model.h"
#include "projection.h"
#include "raycaster.h"

#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

const char *APP_NAME = "Naive VR.";
const int timer_msecs = 1;
//const char *FILE_NAME = "Bucky.raw";						// 32x32x32 x 8bit
//const char *FILE_NAME = "Foot.raw";						// 256x256x256 x 8bit
const char *FILE_NAME = "VisMale.raw";						// 128x256x256 x 8bit
//const char *FILE_NAME = "XMasTree.raw";					// 512x499x512 x 8bit

static int window_id, subwindow_id;
static int2 window_size = {WIN_WIDTH, WIN_HEIGHT};
static bool window_resize_flag = false;
static GLuint pbo_gl_id = NULL;

static int gpu_id;
static cudaGraphicsResource *pbo_cuda_id;

static int renderer_id = 2;
static int4 mouse_state = make_int4(0, 0, 0, GLUT_UP);
static int2 auto_rotate_vector = {0, 0};

cudaEvent_t start, stop; 
float elapsed_time = 0;
char title_string[256];
char append_string[256];

extern float render_volume_gpu(uchar4 *buffer, Raycaster *current_raycaster);
extern void init_gpu(Volume_model volume, int2 window_size);
extern void set_transfer_fn_gpu(float4 *transfer_fn);
extern void resize_gpu(int2 window_size);
extern void free_gpu();

extern float render_volume_gpu2(uchar4 *buffer, Raycaster *current_raycaster);
extern float render_volume_gpu3(uchar4 *buffer, Raycaster *current_raycaster);
extern void set_transfer_fn_gpu23(float4 *transfer_fn);

extern float render_volume_gpu4(uchar4 *buffer, Raycaster *current_raycaster);
extern void init_gpu4(Volume_model volume);
extern void set_transfer_fn_gpu4(float4 *transfer_fn);
extern void free_gpu4();

extern float render_volume_cpu(uchar4 *buffer, Raycaster *current_raycaster);
extern void set_transfer_fn_cpu(float4 *transfer_fn);

void reset_PBO() {
	printf("Setting PBO...\n");
    if (pbo_gl_id != NULL) {
		cudaGraphicsUnregisterResource(pbo_cuda_id);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_gl_id);
    }
	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, window_size.x * window_size.y * 4, NULL, GL_DYNAMIC_DRAW_ARB);		//GL_STREAM_DRAW_ARB ??  // int CHANNEL_COUNT = 4;
	cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard);
}

uchar4 *prepare_PBO() {							//GLubyte *
	if (renderer_id <4) {
		return (uchar4 *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	}
	else {
		uchar4 *dev_buffer;
		size_t dev_buffer_size;
		cudaGraphicsMapResources(1, &pbo_cuda_id, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dev_buffer, &dev_buffer_size, pbo_cuda_id);
		return dev_buffer;
	}
}

void finalize_PBO() {
	if (renderer_id <4) {
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	}
	else {
		cudaGraphicsUnmapResources(1, &pbo_cuda_id, 0);
	}		
}

void draw_volume() {
	//printf("Drawing volume...\n");
	glutSetWindow(window_id);
	if (window_resize_flag) {
		reset_PBO();
		resize_gpu(window_size);
		set_window_size(window_size);
		window_resize_flag = false;
	}
	set_raycaster_view(get_view());
	uchar4 *pbo_array = prepare_PBO();
	cudaEventRecord(start, 0);
	switch (renderer_id) {
		case 1:
			render_volume_cpu(pbo_array, get_raycaster());
			sprintf(append_string, "CPU");
			break;
		case 2 :
			render_volume_gpu(pbo_array, get_raycaster());
			sprintf(append_string, "CUDA Straightforward");
			break;
		case 3 :
			render_volume_gpu2(pbo_array, get_raycaster());
			sprintf(append_string, "CUDA Constant Memory");
			break;
		case 4 :
			render_volume_gpu3(pbo_array, get_raycaster());
			sprintf(append_string, "CUDA CM + GL interop");
			break;
		case 5 :
			render_volume_gpu4(pbo_array, get_raycaster());
			sprintf(append_string, "CUDA CM + 3D Texture Memory + GL interop");
			break;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	finalize_PBO();
	sprintf(title_string, "%s %s @ %3.4f ms", APP_NAME, append_string, elapsed_time);
	glutSetWindowTitle(title_string);
	glutPostRedisplay();
}

void keyboard_callback(unsigned char key, int x, int y) {
	if (key=='t') {
		glClearColor(1,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutWireTeapot(0.5);
	}
	if (key=='y') {
		GLubyte *pbo_array = (GLubyte *)prepare_PBO();
		for(int i = 0; i < WIN_WIDTH * WIN_HEIGHT; i++) {
				*pbo_array++ = (i / WIN_WIDTH) % 256;
				*pbo_array++ = i % 256;
				*pbo_array++ = rand() % 256;
				*pbo_array++ = 255;
			}
		finalize_PBO();
		glutPostRedisplay();
	}
	if (key=='b') {
		glutSwapBuffers();
	}
	if (key=='v') {
		draw_volume();
	}
	if (key=='r') {
		if (auto_rotate_vector.x == 0 && auto_rotate_vector.y == 0) {
			auto_rotate_vector = make_int2(-5, -5);
			printf("Autorotation: on\n");
		}
		else {
			auto_rotate_vector = make_int2(0, 0);
			printf("Autorotation: off\n");
		}
	}
	switch (key) {
		case 'w': camera_down(-5.0f); break;
		case 's': camera_down(5.0f); break;
		case 'a': camera_right(-5.0f); break;
		case 'd': camera_right(5.0f); break;
		case 'q': camera_zoom(0.1f); break;
		case 'e': camera_zoom(-0.1f); break;
		case 'o': change_ray_step(0.01f, false); break;
		case 'p': change_ray_step(-0.01f, false); break;
		case 'k': change_tf_offset(0.025f, false); break;
		case 'l': change_tf_offset(-0.025f, false); break;
		case 'n': change_ray_threshold(0.05f, false); break;
		case 'm': change_ray_threshold(-0.05f, false); break;
		case '-': toggle_perspective(); break;
		case '1': renderer_id = 1; break;
		case '2': renderer_id = 2; break;
		case '3': renderer_id = 3; break;
		case '4': renderer_id = 4; break;
		case '5': renderer_id = 5; break;
		case '7': set_camera_position(2,45,45); break;
		case '8': set_camera_position(2,135,225); break;
		case '9': set_camera_position(2,225,225); break;
		case '0': set_camera_position(2,0,0); break;
	}
	if (key==27) {
		cudaGraphicsUnregisterResource(pbo_cuda_id);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_gl_id);
		free_gpu4();
		free_gpu();
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		glutDestroyWindow(window_id);
		exit(0);
	}
}

void display_callback(void) {
	//printf("Main window display callback...\n");
	glDrawPixels(window_size.x, window_size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

void timer_callback(int value) {
	//printf("Timer ticked...\n");
	if (auto_rotate_vector.x != 0 && mouse_state.w == GLUT_UP)
		camera_right(auto_rotate_vector.x);
	if (auto_rotate_vector.y != 0 && mouse_state.w == GLUT_UP) 
		camera_down(auto_rotate_vector.y);
	draw_volume();
	glutTimerFunc(timer_msecs, timer_callback, 0);
}

void mouse_callback(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	mouse_state.w = state;
	if (state == GLUT_DOWN) 
		auto_rotate_vector = make_int2(0, 0);
	if (state == GLUT_UP) {
		if (abs(auto_rotate_vector.x) < 8 && abs(auto_rotate_vector.y) < 8)
			auto_rotate_vector = make_int2(0, 0);
	}
	//printf("Mouse click: button:%i state:%i x:%i y:%i\n", button, state, x, y);
}

void motion_callback(int x, int y) {
	if (mouse_state.z == GLUT_LEFT_BUTTON) {	  
		auto_rotate_vector.x = x - mouse_state.x;
		auto_rotate_vector.y = y - mouse_state.y;
		camera_right(auto_rotate_vector.x); 
		camera_down(auto_rotate_vector.y);
	}
	mouse_state.x = x;
	mouse_state.y = y;
}

void reshape_callback(int w, int h) {
	//printf("Resizing main window...\n");
	if (window_size.x != w || window_size.y != h) {
		window_size.x = w;
		window_size.y = h;
		window_resize_flag = true;
	}
	glutSetWindow(subwindow_id);
	glutPositionWindow(window_size.x/20, (window_size.y/20)*17);
	glutReshapeWindow((window_size.x*18)/20, window_size.y/10);
	glutSetWindow(window_id);
}

void display_callback_sub(void) {
	//printf("Drawing subwindow...\n");
	glClear(GL_COLOR_BUFFER_BIT);	
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	float4 *transfer_fn = get_transfer_fn();
	for (int i=0; i < 255; i++) {
		glBegin(GL_POLYGON);
		glColor4f(transfer_fn[i].x, transfer_fn[i].y, transfer_fn[i].z, 1.0f);
		glVertex2f(i, 0);
		glVertex2f(i, transfer_fn[i].w);
		glColor4f(transfer_fn[i+1].x, transfer_fn[i+1].y, transfer_fn[i+1].z, 1.0f);
		glVertex2f(i+1, transfer_fn[i+1].w);
		glVertex2f(i+1, 0);
		glEnd();
	}
	glDisable(GL_BLEND);
}

void motion_callback_sub(int x, int y) {
	float4 *transfer_fn = get_transfer_fn();
	float win_width = (float)glutGet(GLUT_WINDOW_WIDTH), win_height = (float)glutGet(GLUT_WINDOW_HEIGHT);
	int steps = abs(x - mouse_state.x);
	int x_delta = mouse_state.x < x ? 1 : -1;
	float y_delta = (steps == 0) ? 0 : (y - mouse_state.y) / (float)steps;
	for (int i = 0; i <= steps; i++) {
		int sample = CLAMP((int)((mouse_state.x + i * x_delta) / (win_width / 256)), 0, 255);
		float intensity = CLAMP(1.0f - (mouse_state.y + i * y_delta) / win_height, 0, 1.0f);
		transfer_fn[sample].w = intensity;
	}
	mouse_state.x = x;
	mouse_state.y = y;
	glutPostRedisplay();
	set_transfer_fn_cpu(transfer_fn);
	set_transfer_fn_gpu(transfer_fn);
	set_transfer_fn_gpu23(transfer_fn);
	set_transfer_fn_gpu4(transfer_fn);
}

void mouse_callback_sub(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	motion_callback_sub(x, y);
}

int main(int argc, char **argv) {

	if (load_model(FILE_NAME) != 0) {
		fprintf(stderr, "File error: %s\n", FILE_NAME);
		exit(1);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);		//GLUT_DOUBLE
	glutInitWindowSize(window_size.x, window_size.y);
	glutInitWindowPosition(100,1);
	window_id = glutCreateWindow(APP_NAME);
	glutDisplayFunc(display_callback);
	glutKeyboardFunc(keyboard_callback);
	glutMouseFunc(mouse_callback);
    glutMotionFunc(motion_callback);
	glutTimerFunc(timer_msecs, timer_callback, 0);
	glutReshapeFunc(reshape_callback);

	subwindow_id = glutCreateSubWindow(window_id, window_size.x/20, (window_size.y/20)*17, 256, 1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho (0, 256, 0, 1, 0, 1);
	glMatrixMode (GL_MODELVIEW);
	glDisable(GL_DEPTH_TEST);
	glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
	glutDisplayFunc(display_callback_sub);
	glutMouseFunc(mouse_callback_sub);
	glutMotionFunc(motion_callback_sub);
	glutSetWindow(window_id);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
		glutDestroyWindow(window_id);
		exit(1);
	}
	printf("Using GLEW %s\n\n", glewGetString(GLEW_VERSION));
	if (!GLEW_VERSION_2_0) {
		printf("Error: OpenGL 2.0 is not supported.\n");
		glutDestroyWindow(window_id);
		exit(1);
	}

	printf("Use '12345' to change renderer\n    'wasd' and '7890' to manipulate camera position\n");
	printf("    'op' to change ray sampling rate\n    'kl' to change transfer function offset\n");
	printf("    'nm' to change ray accumulation threshold\n    'r' to toggle autorotation\n");
	printf("    '-' to toggle perspective and orthogonal projection\n\n");

	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.major = 0;
	cudaChooseDevice(&gpu_id, &prop);
	cudaGLSetGLDevice(gpu_id);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	reset_PBO();
	set_raycaster_model(get_model());
	init_gpu(get_model(), window_size);
	init_gpu4(get_model());
	set_transfer_fn_cpu(get_transfer_fn());
	set_transfer_fn_gpu(get_transfer_fn());
	set_transfer_fn_gpu23(get_transfer_fn());
	set_transfer_fn_gpu4(get_transfer_fn());

	printf("Raycaster data size: %i B\n",sizeof(Raycaster));
	glutMainLoop();
	return 1;
}
