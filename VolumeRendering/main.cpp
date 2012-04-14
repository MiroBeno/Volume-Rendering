#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glew.h"
#include "glut.h"

#include "Model.h"
#include "View.h"
#include "Raycaster.h"
#include "Renderer.h"

#include "cuda_utils.h"
#include "cuda_gl_interop.h"


const char *APP_NAME = "VR:";
const int TIMER_MSECS = 1;
const int RENDERERS_COUNT = 5;
const char *FILE_NAME = "Bucky.pvm";						// 32x32x32 x 8bit
//const char *FILE_NAME = "Foot.pvm";						// 256x256x256 x 8bit
//const char *FILE_NAME = "VisMale.pvm";					// 128x256x256 x 8bit
//const char *FILE_NAME = "Bonsai1-LO.pvm";					// 512x512x182 x 16 bit

static int window_id, subwindow_id;
static ushort2 window_size = {INT_WIN_WIDTH, INT_WIN_HEIGHT};
static bool window_resize_flag = false;
static GLuint pbo_gl_id = NULL;
static GLuint tex_gl_id = NULL;

static int gpu_id;
static cudaGraphicsResource *pbo_cuda_id;

static short4 mouse_state = {0, 0, 0, GLUT_UP};
static short2 auto_rotate_vector = {0, 0};

static cudaEvent_t start, stop, frame; 
static float2 elapsed_time = {0, 0};
static char title_string[256];

static int renderer_id = 1;
static Renderer *renderers[RENDERERS_COUNT];
static char renderer_names[RENDERERS_COUNT][256]= {"CPU", "CUDA Straightforward", "CUDA Constant Memory", "CUDA CM + GL interop", "CUDA CM + 3D Texture Memory + GLI"};

static RaycasterBase raycaster_base;
static ModelBase model_base;
static ViewBase view_base;

void delete_PBO_texture() {
    if (pbo_gl_id != NULL) {
		cuda_safe_call(cudaGraphicsUnregisterResource(pbo_cuda_id));
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_gl_id);
    }
	if (tex_gl_id != NULL) {
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &tex_gl_id);
	}
}

void reset_PBO_texture() {
	printf("Setting PBO...\n");
	delete_PBO_texture();
	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, window_size.x * window_size.y * 4, NULL, GL_STREAM_DRAW_ARB);		//GL_STREAM_DRAW_ARB|GL_DYNAMIC_DRAW_ARB ??  // int CHANNEL_COUNT = 4;
	cuda_safe_call(cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &tex_gl_id);
	glBindTexture(GL_TEXTURE_2D, tex_gl_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_size.x, window_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

uchar4 *prepare_PBO() {							//GLubyte *
	if (renderer_id < 3) {
		return (uchar4 *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	}
	else {
		uchar4 *dev_buffer;
		size_t dev_buffer_size;
		cuda_safe_call(cudaGraphicsMapResources(1, &pbo_cuda_id, 0));
		cuda_safe_call(cudaGraphicsResourceGetMappedPointer((void **)&dev_buffer, &dev_buffer_size, pbo_cuda_id));
		return dev_buffer;
	}
}

void finalize_PBO() {
	if (renderer_id < 3) {
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	}
	else {
		cuda_safe_call(cudaGraphicsUnmapResources(1, &pbo_cuda_id, 0));
	}		
}

void draw_volume() {
	//printf("Drawing volume...\n");
	glutSetWindow(window_id);
	if (window_resize_flag) {
		if (window_size.x == 0 || window_size.y == 0)
			return;
		reset_PBO_texture();
		glViewport(0, 0, window_size.x, window_size.y);
		view_base.set_window_size(window_size);
		for (int i=0; i < RENDERERS_COUNT; i++)
			renderers[i]->set_window_buffer(view_base.view);
		window_resize_flag = false;
	}
	raycaster_base.raycaster.view = view_base.view;
	uchar4 *pbo_array = prepare_PBO();
	cuda_safe_call(cudaEventRecord(start, 0));
	renderers[renderer_id]->render_volume(pbo_array, raycaster_base.raycaster);
	cuda_safe_call(cudaEventRecord(stop, 0));
	cuda_safe_call(cudaEventSynchronize(stop));
	cuda_safe_call(cudaEventElapsedTime(&elapsed_time.x, start, stop));
	cuda_safe_call(cudaEventElapsedTime(&elapsed_time.y, frame, stop));
	finalize_PBO();
	sprintf(title_string, "%s %s @ %.2f ms / %.2f ms (%dx%d)", APP_NAME, renderer_names[renderer_id], elapsed_time.x, elapsed_time.y, window_size.x, window_size.y);
	glutSetWindowTitle(title_string);
	cuda_safe_call(cudaEventRecord(frame, 0));
	glutPostRedisplay();
}

void keyboard_callback(unsigned char key, int x, int y) {
	switch (key) {
		case 'w': view_base.camera_down(-5.0f); break;
		case 's': view_base.camera_down(5.0f); break;
		case 'a': view_base.camera_right(-5.0f); break;
		case 'd': view_base.camera_right(5.0f); break;
		case 'q': view_base.camera_zoom(0.1f); break;
		case 'e': view_base.camera_zoom(-0.1f); break;
		case 'o': raycaster_base.change_ray_step(0.01f, false); break;
		case 'p': raycaster_base.change_ray_step(-0.01f, false); break;
		case 'n': raycaster_base.change_ray_threshold(0.05f, false); break;
		case 'm': raycaster_base.change_ray_threshold(-0.05f, false); break;
		case 'l': raycaster_base.toggle_esl(); break;
		case '-': view_base.toggle_perspective(); break;
		case '`': renderer_id = 0; break;
		case '1': renderer_id = 1; break;
		case '2': renderer_id = 2; break;
		case '3': renderer_id = 3; break;
		case '4': renderer_id = 4; break;
		case '7': view_base.set_camera_position(3,-45,-45); break;
		case '8': view_base.set_camera_position(3,0,0); break;
		case '9': view_base.set_camera_position(3,-90,0); break;
		case '0': view_base.set_camera_position(3,180,-90); break;
		case 'r':	if (auto_rotate_vector.x == 0 && auto_rotate_vector.y == 0) {
						auto_rotate_vector = make_short2(-5, -5);
						printf("Autorotation: on\n");
					}
					else {
						auto_rotate_vector = make_short2(0, 0);
						printf("Autorotation: off\n");
					}
					break;
		case 'b': glutSwapBuffers(); break;
		case 'v': draw_volume(); break;
		case 'y':	glClearColor(1,0,0,1);
					glClear(GL_COLOR_BUFFER_BIT);
					glutWireTeapot(0.5);
					break;
		case 'u':	GLubyte *pbo_array = (GLubyte *)prepare_PBO();
					for(int i = 0; i < INT_WIN_WIDTH * INT_WIN_HEIGHT; i++) {
							*pbo_array++ = (i / INT_WIN_WIDTH) % 256;
							*pbo_array++ = i % 256;
							*pbo_array++ = rand() % 256;
							*pbo_array++ = 255;
						}
					finalize_PBO();
					glutPostRedisplay();
					break;
	}
	if (key==27) {
		cuda_safe_call(cudaEventDestroy(start));
		cuda_safe_call(cudaEventDestroy(stop));
		cuda_safe_call(cudaEventDestroy(frame));
		delete_PBO_texture();
		for (int i = RENDERERS_COUNT - 1; i >=0 ; i--)
			delete renderers[i];
		free(model_base.volume.data);
		free(raycaster_base.raycaster.transfer_fn);
		free(raycaster_base.raycaster.esl_min_max);
		free(raycaster_base.raycaster.esl_volume);
		glutDestroyWindow(window_id);
		exit(0);
	}
}

void display_callback(void) {
	//printf("Main window display callback...\n");
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_size.x, window_size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(0, 0);
	glTexCoord2f(1, 0); glVertex2f(1, 0);
	glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(0, 1);
	glEnd();
	glDisable(GL_BLEND);
	glutSwapBuffers();
}

void idle_callback(){
	if (mouse_state.w == GLUT_UP || mouse_state.z != GLUT_LEFT_BUTTON) {
		if (auto_rotate_vector.x != 0)
			view_base.camera_right(auto_rotate_vector.x);
		if (auto_rotate_vector.y != 0) 
			view_base.camera_down(auto_rotate_vector.y);
	}
    draw_volume();
}

void timer_callback(int value) {
	if (mouse_state.w == GLUT_UP || mouse_state.z != GLUT_LEFT_BUTTON) {
		if (auto_rotate_vector.x != 0)
			view_base.camera_right(auto_rotate_vector.x);
		if (auto_rotate_vector.y != 0) 
			view_base.camera_down(auto_rotate_vector.y);
	}
	draw_volume();
	glutTimerFunc(TIMER_MSECS, timer_callback, 0);
}

void mouse_callback(int button, int state, int x, int y) {
	//printf("Mouse click: button:%i state:%i x:%i y:%i\n", button, state, x, y);
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	mouse_state.w = state;
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) 
			auto_rotate_vector = make_short2(0, 0);
		if (state == GLUT_UP) {
			if (abs(auto_rotate_vector.x) < 8 && abs(auto_rotate_vector.y) < 8)
				auto_rotate_vector = make_short2(0, 0);
		}
	}
}

void motion_callback(int x, int y) {
	if (mouse_state.z == GLUT_LEFT_BUTTON) {	  
		auto_rotate_vector.x = x - mouse_state.x;
		auto_rotate_vector.y = y - mouse_state.y;
		view_base.camera_right(auto_rotate_vector.x); 
		view_base.camera_down(auto_rotate_vector.y);
	}
	if (mouse_state.z == GLUT_RIGHT_BUTTON) {	  
		view_base.camera_zoom(y - mouse_state.y); 
	}
	mouse_state.x = x;
	mouse_state.y = y;
}

void reshape_callback(int w, int h) {				
	//printf("Resizing main window... %i %i\n", w, h);
	if (window_size.x != w || window_size.y != h) {
		window_size.x = w;
		window_size.y = h;
		window_resize_flag = true;
	}
	if (window_size.x > 8 && window_size.y > 8) {
		glutSetWindow(subwindow_id);
		glutPositionWindow(window_size.x/20, (window_size.y/20)*17);
		glutReshapeWindow((window_size.x*18)/20, window_size.y/8);
		glutSetWindow(window_id);
	}
}

void display_callback_sub(void) {
	//printf("Drawing subwindow...\n");
	glClear(GL_COLOR_BUFFER_BIT);	
	float4 *tf = raycaster_base.raycaster.transfer_fn;
	glBegin(GL_QUAD_STRIP);
	for (int i=0; i < TF_SIZE; i++) {
		glColor4f(tf[i].x, tf[i].y, tf[i].z, 1.0f);
		glVertex2f(i*2, 0);
		glVertex2f(i*2, sqrt(sqrt(tf[i].w)));
	}
	glEnd();
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBegin(GL_QUAD_STRIP);
	for (int i=0; i < 256; i++) {
		glColor4f(1, 1, 1, 0.9f);
		glVertex2f(i, 0);
		glColor4f(1, 1, 1, 0.1f);
		glVertex2f(i, model_base.histogram[i]);
	}
	glEnd();
	glDisable(GL_BLEND);
	glutSwapBuffers();
}

void motion_callback_sub(int x, int y) {
	float win_width = (float)glutGet(GLUT_WINDOW_WIDTH), win_height = (float)glutGet(GLUT_WINDOW_HEIGHT);
	int steps = abs(x - mouse_state.x);
	int x_delta = mouse_state.x < x ? 1 : -1;
	float y_delta = (steps == 0) ? 0 : (y - mouse_state.y) / (float)steps;
	for (int i = 0; i <= steps; i++) {
		int sample = CLAMP((int)((mouse_state.x + i * x_delta) / (win_width / TF_SIZE)), 0, (TF_SIZE-1));
		float intensity;
		if (mouse_state.z == GLUT_LEFT_BUTTON) {
			intensity = CLAMP(1.0f - (mouse_state.y + i * y_delta) / win_height, 0, 1.0f);
			intensity = pow(intensity, 4);
		}
		if (mouse_state.z != GLUT_LEFT_BUTTON) 
			intensity = 0;
		raycaster_base.raycaster.transfer_fn[sample].w = intensity;
	}
	mouse_state.x = x;
	mouse_state.y = y;
	glutPostRedisplay();
	raycaster_base.update_esl_volume();
	for (int i=0; i < RENDERERS_COUNT; i++)
		renderers[i]->set_transfer_fn(raycaster_base.raycaster);
}

void mouse_callback_sub(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	motion_callback_sub(x, y);
}

int main(int argc, char **argv) {

	if (model_base.load_model(FILE_NAME) != 0) {
		exit(EXIT_FAILURE);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);		//GLUT_DOUBLE | GLUT_MULTISAMPLE
	glutInitWindowSize(window_size.x, window_size.y);
	glutInitWindowPosition(800, 1);

	window_id = glutCreateWindow(APP_NAME);
	glutDisplayFunc(display_callback);
	glutKeyboardFunc(keyboard_callback);
	glutMouseFunc(mouse_callback);
    glutMotionFunc(motion_callback);
	//glutTimerFunc(1, timer_callback, 0);
	glutIdleFunc(idle_callback);
	glutReshapeFunc(reshape_callback);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
    //glViewport(0, 0, window_size.x, window_size.y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glClearColor(0.25, 0.25, 0.25, 1);

	subwindow_id = glutCreateSubWindow(window_id, window_size.x/20, (window_size.y/20)*17, (window_size.x*18)/20, window_size.y/8);
	glutDisplayFunc(display_callback_sub);
	glutMouseFunc(mouse_callback_sub);
	glutMotionFunc(motion_callback_sub);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glOrtho(0.0, 255.0, 0.0, 1.0, 0.0, 1.0);
	glClearColor(0, 0, 0, 1);

	glutSetWindow(window_id);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: Initializing GLEW failed: %s\n", glewGetErrorString(err));
		glutDestroyWindow(window_id);
		exit(EXIT_FAILURE);
	}
	printf("Using GLEW %s\n\n", glewGetString(GLEW_VERSION));
	if (!GLEW_VERSION_2_0) {
		printf("Error: OpenGL 2.0 is not supported\n");
		glutDestroyWindow(window_id);
		exit(EXIT_FAILURE);
	}

	printf("Use '`1234' to change renderer\n    'wasd' and '7890' to manipulate camera position\n");
	printf("    'op' to change ray sampling rate\n");
	printf("    'nm' to change ray accumulation threshold\n    'r' to toggle autorotation\n");
	printf("    '-' to toggle perspective and orthogonal projection\n\n");

        
    int device_count = 0;
	cuda_safe_call(cudaGetDeviceCount(&device_count));
	if (device_count == 0) {
		printf("Error: No device supporting CUDA found\n");
		exit(EXIT_FAILURE);
	}
	printf("Number of CUDA devices found: %d\n", device_count);
  
	cudaDeviceProp device_prop;
	int max_compute_perf = 0, max_perf_device = -1, best_arch = 0;
	for (int i = 0; i < device_count; i++) {
		cuda_safe_call(cudaGetDeviceProperties(&device_prop, i));
		if ((!device_prop.tccDriver) &&
			(device_prop.major > 0 && device_prop.major < 9999)) 
				best_arch = MAXIMUM(best_arch, device_prop.major);
	}
	for (int i = 0; i < device_count; i++) {
		cuda_safe_call(cudaGetDeviceProperties(&device_prop, i));
		if ((device_prop.major == 9999 && device_prop.minor == 9999) 
			|| (device_prop.tccDriver)
			|| (device_prop.major != best_arch))
			continue;
		int cores_per_mp = 0;
		switch (device_prop.major) {
			case 1: cores_per_mp = 8; break;
			case 2: switch (device_prop.minor) {
						case 1: cores_per_mp = 32; break;
						case 2: cores_per_mp = 48; break;
					}
					break;
			case 3: cores_per_mp = 192; break;
		}
		int compute_perf = device_prop.multiProcessorCount * cores_per_mp * device_prop.clockRate;
		if (compute_perf > max_compute_perf) {
			max_compute_perf = compute_perf;
			max_perf_device  = i;
		}
	}
	gpu_id = max_perf_device;

    cuda_safe_call(cudaGetDeviceProperties(&device_prop, gpu_id));
    printf("\nUsing device %d: \"%s\"\n", gpu_id, device_prop.name);
	int version = 0;
	cuda_safe_call(cudaDriverGetVersion(&version));
	printf("  CUDA Driver Version:                      %d.%d\n", version/1000, version%100);
	cuda_safe_call(cudaRuntimeGetVersion(&version));
	printf("  CUDA Runtime Version:                     %d.%d\n", version/1000, version%100);
    printf("  CUDA Capability version number:           %d.%d\n", device_prop.major, device_prop.minor);
	int cores_per_mp = 0;
	switch (device_prop.major) {
		case 1: cores_per_mp = 8; break;
		case 2: switch (device_prop.minor) {
					case 1: cores_per_mp = 32; break;
					case 2: cores_per_mp = 48; break;
				}
				break;
		case 3: cores_per_mp = 192; break;
	}
	if (cores_per_mp != 0)
		printf("  Multiprocessors x Cores/MP = Cores:       %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
			device_prop.multiProcessorCount,
			cores_per_mp,
			cores_per_mp * device_prop.multiProcessorCount);
	printf("  Total amount of global memory:            %llu bytes\n", (unsigned long long) device_prop.totalGlobalMem);
	printf("  Total amount of constant memory:          %u bytes\n", device_prop.totalConstMem); 
	printf("  Total amount of shared memory per block:  %u bytes\n", device_prop.sharedMemPerBlock);
	printf("  Total number of registers per block:	    %d\n", device_prop.regsPerBlock);
	printf("  Clock rate:                               %.2f GHz\n", device_prop.clockRate * 1e-6f);
	printf("  Integrated:                               %s\n", device_prop.integrated ? "Yes" : "No");
	printf("\n");

	cuda_safe_call(cudaGLSetGLDevice(gpu_id));
	cuda_safe_call(cudaEventCreate(&start));
	cuda_safe_call(cudaEventCreate(&stop));
	cuda_safe_call(cudaEventCreate(&frame));
	cuda_safe_call(cudaEventRecord(frame, 0));

	reset_PBO_texture();

	for (int i =0; i < TF_SIZE; i++) {
		raycaster_base.raycaster.transfer_fn[i] = make_float4(i <= TF_SIZE/3 ? (i*3)/128.0f : 0.0f, 
										(i > TF_SIZE/3) && (i <= TF_SIZE/3*2) ? ((i-TF_SIZE/3)*3)/128.0f : 0.0f, 
										i > TF_SIZE/3*2 ? ((i-TF_SIZE/3*2)*3)/128.0f : 0.0f, 
										i > 10 ? i/128.0f : 0.0f);
	}

	raycaster_base.raycaster.view = view_base.view;
	raycaster_base.set_volume(model_base.volume);

	renderers[0] = new CPURenderer(raycaster_base.raycaster);	
	renderers[1] = new GPURenderer1(raycaster_base.raycaster);
	renderers[2] = new GPURenderer2(raycaster_base.raycaster);
	renderers[3] = new GPURenderer3(raycaster_base.raycaster);
	renderers[4] = new GPURenderer4(raycaster_base.raycaster);
	
	printf("Raycaster data size: %i B\n", sizeof(Raycaster));
	glutMainLoop();
	return EXIT_FAILURE;
}
