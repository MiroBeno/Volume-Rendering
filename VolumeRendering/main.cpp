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

const char *APP_NAME = "Naive VR. ";
const int timer_msecs = 1;
const GLsizeiptr DATA_SIZE = WIN_WIDTH * WIN_HEIGHT * 4;	// int CHANNEL_COUNT = 4;
//const char *FILE_NAME = "Bucky.raw";						// 32x32x32 x 8bit
//const char *FILE_NAME = "Foot.raw";						// 256x256x256 x 8bit
const char *FILE_NAME = "VisMale.raw";					// 128x256x256 x 8bit
//const char *FILE_NAME = "XMasTree.raw";					// 512x499x512 x 8bit

static int window_id;
static GLuint pbo_gl_id;

static int gpu_id;
static cudaGraphicsResource *pbo_cuda_id;

static int renderer_id = 2;
static int3 mouse_state;
static bool auto_rotate = true;

float elapsed_time = 0;
char title_string[256];
char append_string[256];

extern float render_volume_gpu(uchar4 *buffer, Raycaster current_raycaster);
extern void init_gpu(Volume_model volume);
extern void free_gpu();

extern float render_volume_gpu2(uchar4 *buffer, Raycaster current_raycaster);

extern float render_volume_gpu3(uchar4 *buffer, Raycaster current_raycaster);
extern void init_gpu3(Volume_model volume);
extern void free_gpu3();

extern float render_volume_gpu4(uchar4 *buffer, Raycaster current_raycaster);
extern void init_gpu4();
extern void free_gpu4();

extern float render_volume_cpu(uchar4 *buffer, Raycaster current_raycaster);

uchar4 *prepare_PBO() {							//GLubyte *
	if (renderer_id <5)
		return (uchar4 *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	else {
		uchar4 *dev_buffer;
		size_t dev_buffer_size;
		cudaGraphicsMapResources(1, &pbo_cuda_id, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dev_buffer, &dev_buffer_size, pbo_cuda_id);
		return dev_buffer;
	}
}

void finalize_PBO() {
	if (renderer_id <5)
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	else {
		cudaGraphicsUnmapResources(1, &pbo_cuda_id, 0);
	}		
}

void draw_volume() {
	set_raycaster_view(get_view());
	uchar4 *pbo_array = prepare_PBO();
	switch (renderer_id) {
		case 1:
			elapsed_time = render_volume_cpu(pbo_array, get_raycaster());
			sprintf(append_string, "CPU @ %3.4f ms", elapsed_time);
			break;
		case 2 :
			elapsed_time = render_volume_gpu(pbo_array, get_raycaster());
			sprintf(append_string, "Standard CUDA @ %3.4f ms", elapsed_time);
			break;
		case 3 :
			elapsed_time = render_volume_gpu2(pbo_array, get_raycaster());
			sprintf(append_string, "Constant Memory @ %3.4f ms", elapsed_time);
			break;
		case 4 :
			elapsed_time = render_volume_gpu3(pbo_array, get_raycaster());
			sprintf(append_string, "CM + 3D Texture Memory @ %3.4f ms", elapsed_time);
			break;
		case 5 :
			elapsed_time = render_volume_gpu4(pbo_array, get_raycaster());
			sprintf(append_string, "CM + 3D Texture Memory + GL interop @ %3.4f ms", elapsed_time);
			break;
	}
	finalize_PBO();
	strcpy(title_string, APP_NAME);
	strcat(title_string, append_string);
	glutSetWindowTitle(title_string);
	glutPostRedisplay();
}

void display_callback(void) {
	glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	//glutSwapBuffers();
}

void timer_callback(int value) {
	if (auto_rotate) {
		camera_right(-0.5f);
		camera_down(-0.5f);
	}
	draw_volume();
	glutTimerFunc(timer_msecs, timer_callback, 0);
}

void keyboard_callback(unsigned char key, int x, int y) {
	if (key=='t') {
		glClearColor(1,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutWireTeapot(0.5);
	}
	if (key=='y') {
		renderer_id = 1;
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
		auto_rotate = !auto_rotate;
		printf("Autorotation: %s\n", auto_rotate ? "on" : "off");
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
		glutDestroyWindow(window_id);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_gl_id);
		cudaGraphicsUnregisterResource(pbo_cuda_id);
		free_gpu4();
		free_gpu3();
		free_gpu();
		exit(0);
	}
}

void mouse_callback(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = (button==0) ? state : mouse_state.z;
	//printf("click button:%i state:%i x:%i y:%i\n", button, state, x, y);
}

void motion_callback(int x, int y) {
	if (mouse_state.z == 0) {
		camera_right(x - mouse_state.x); 
		camera_down(y - mouse_state.y);	  
	}
	mouse_state.x = x;
	mouse_state.y = y;
}

int main(int argc, char **argv) {

	if (load_model(FILE_NAME) != 0) {
		fprintf(stderr, "File error: %s\n", FILE_NAME);
		exit(1);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);		//GLUT_DOUBLE (+ glutSwapBuffers().  Rozdiely? Pomalsie pri double znacne, aj ked sa meria iba cas kernelu!)
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutInitWindowPosition(100,1);
	window_id = glutCreateWindow(APP_NAME);
	glutDisplayFunc(display_callback);
	glutKeyboardFunc(keyboard_callback);
	glutMouseFunc(mouse_callback);
    glutMotionFunc(motion_callback);
	glutTimerFunc(timer_msecs, timer_callback, 0);
	//glutReshapeFunc(reshape_callback);

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

	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.major = 0;
	cudaChooseDevice(&gpu_id, &prop);
	cudaGLSetGLDevice(gpu_id);

	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, DATA_SIZE, NULL, GL_DYNAMIC_DRAW_ARB);

	cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard);

	printf("Use '12345' to change renderer\n    'wasd' and '7890' to manipulate camera position\n");
	printf("    'op' to change ray sampling rate\n    'kl' to change transfer function offset\n");
	printf("    'nm' to change ray accumulation threshold\n    'r' to toggle autorotation\n");
	printf("    '-' to toggle perspective and orthogonal projection\n\n");

	set_raycaster_model(get_model());
	init_gpu(get_model());
	init_gpu3(get_model());
	init_gpu4();

	glutMainLoop();
	return 0;
}
