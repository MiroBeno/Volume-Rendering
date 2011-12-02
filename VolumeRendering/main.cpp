#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "glew.h"
#include "glut.h"

#include "model.h"
#include "projection.h"

//#include "cuda_runtime.h"
//#include "cuda_runtime_api.h"

const GLsizeiptr DATA_SIZE = WIN_WIDTH * WIN_HEIGHT * 4;	// int CHANNEL_COUNT = 4;
const char *FILE_NAME = "Bucky.raw";						// 32x32x32  x unsigned char
//const char *FILE_NAME = "nucleon.raw";						// 41x41x41  x unsigned char

static int window_id;
static GLuint pbo_id;

static int renderer_id = 1;
static bool auto_rotate = true;

extern float render_volume_gpu(uchar4 *buffer, Ortho_view ortho_view);
extern void init_gpu(Volume_model volume_model);
extern void free_gpu(void);

extern float render_volume_gpu2(uchar4 *buffer, Ortho_view ortho_view);
extern void init_gpu2(Volume_model volume_model);
extern void free_gpu2(void);

extern float render_volume_gpu3(uchar4 *buffer, Ortho_view ortho_view);
extern void init_gpu3(Volume_model volume_model);
extern void free_gpu3(void);

extern void render_volume_cpu(unsigned char *buffer, Ortho_view ortho_view);
extern void init_cpu(Volume_model volume_model);

extern cudaEvent_t start, stop;
extern float elapsedTime;


GLubyte *gl_prepare_PBO() {
	return (GLubyte *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
}

void gl_finalize_PBO() {
	glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
}

void draw_volume() {
	GLubyte *pbo_array = gl_prepare_PBO();
	update_view(WIN_WIDTH, WIN_HEIGHT, VIRTUAL_VIEW_SIZE);
	float elapsedTime = 0;
	char titleString[256] = "Naive Volume Rendering. ";
	char appendString[256] = "";
	switch (renderer_id) {
		case 1:
			render_volume_cpu((unsigned char *)pbo_array, get_view());
			break;
		case 2 :
			elapsedTime = render_volume_gpu((uchar4 *)pbo_array, get_view());
			sprintf(appendString, "Standard CUDA @ %3.1f fps", 1000.f / elapsedTime);
			break;
		case 3 :
			elapsedTime = render_volume_gpu2((uchar4 *)pbo_array, get_view());
			sprintf(appendString, "Constant Memory @ %3.1f fps", 1000.f / elapsedTime);
			break;
		case 4 :
			elapsedTime = render_volume_gpu3((uchar4 *)pbo_array, get_view());
			sprintf(appendString, "CM + 3D Texture Memory @ %3.1f fps", 1000.f / elapsedTime);
			break;
	}
	strcat(titleString, appendString);
	glutSetWindowTitle(titleString);
	gl_finalize_PBO();
	glutPostRedisplay();
}

void draw_random() {
	GLubyte *pbo_array = gl_prepare_PBO();
	for(int i = 0; i < WIN_WIDTH * WIN_HEIGHT; i++)
		{
			*pbo_array++ = (i / WIN_WIDTH) % 256;
			*pbo_array++ = i % 256;
			*pbo_array++ = rand() % 256;
			*pbo_array++ = 255;
		}
	gl_finalize_PBO();
	glutPostRedisplay();
}

void display_callback(void) {
	glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	//glutSwapBuffers();

/*	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	char titleString[256] = "Naive Volume Rendering. ";
	char appendString[256] = "";
	sprintf(appendString, "Test @ %3.1f fps", 1000.f / elapsedTime);
	strcat(titleString, appendString);
	glutSetWindowTitle(titleString);*/
}

void keyboard_callback(unsigned char key, int x, int y) {
	if (key=='t') {
		glClearColor(1,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutWireTeapot(0.5);
		//glutSwapBuffers();
	}
	if (key=='y') {
		draw_random();
	}
	if (key=='b') {
		glutSwapBuffers();
	}
	if (key=='v') {
		draw_volume();
	}
	if (key=='r') {
		auto_rotate = !auto_rotate;
	}
	if (strchr("wsadqe1234567890", key)) {
		switch (key) {
			case 'w': camera_up(5.0f); break;
			case 's': camera_down(5.0f); break;
			case 'a': camera_left(5.0f); break;
			case 'd': camera_right(5.0f); break;
			case 'q': camera_zoom_in(0.1f); break;
			case 'e': camera_zoom_out(0.1f); break;
			case '1': renderer_id = 1; break;
			case '2': renderer_id = 2; break;
			case '3': renderer_id = 3; break;
			case '4': renderer_id = 4; break;
			case '7': set_camera_position_deg(2,45,45); break;
			case '8': set_camera_position_deg(2,135,225); break;
			case '9': set_camera_position_deg(2,225,225); break;
			case '0': set_camera_position_deg(2,0,0); break;
		}
		draw_volume();
	}
	if (key==27) {
		glutDestroyWindow(window_id);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_id);
		free_gpu();
		free_gpu2();
		free_gpu3();
		exit(0);
	}

}

void idle_callback()
{
	if (auto_rotate) {
		camera_left(1.0f);
		camera_up(1.0f);
		draw_volume();
	}
}

int main(int argc, char **argv) {

	if (load_model(FILE_NAME) != 0) {
		fprintf(stderr, "File error: %s\n", FILE_NAME);
		exit(1);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);		//GLUT_DOUBLE (+ glutSwapBuffers().  Rozdiely? Pomalsie pri dobule znacne, aj ked sa meria iba cas kernelu!)
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutInitWindowPosition(100,1);
	window_id = glutCreateWindow("Naive Volume Rendering");
	glutDisplayFunc(display_callback);
	glutKeyboardFunc(keyboard_callback);
	//glutReshapeFunc(reshape_callback);
    glutIdleFunc(idle_callback);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
		glutDestroyWindow(window_id);
		exit(1);
	}
	printf("Using GLEW %s\n", glewGetString(GLEW_VERSION));
	if (!GLEW_VERSION_2_0) {
		printf("Error: OpenGL 2.0 is not supported.\n");
		glutDestroyWindow(window_id);
		exit(1);
	}

	glGenBuffersARB(1, &pbo_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, DATA_SIZE, NULL, GL_DYNAMIC_DRAW_ARB);

	init_cpu(get_model());
	init_gpu(get_model());
	init_gpu2(get_model());
	init_gpu3(get_model());
	draw_volume();
	glutMainLoop();
	return 0;
}
