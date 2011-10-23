#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "glew.h"
#include "glut.h"

#include "model.h"
#include "projection.h"

const int WIN_WIDTH = 512;
const int WIN_HEIGHT = 512;
const GLsizeiptr DATA_SIZE = WIN_WIDTH * WIN_HEIGHT * 4;	// int CHANNEL_COUNT = 4;
const char *FILE_NAME = "Bucky.raw";						// 32x32x32  x unsigned char

static int window_id;
static GLuint pbo_id;

GLubyte *gl_prepare_PBO() {
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, DATA_SIZE, NULL, GL_DYNAMIC_DRAW_ARB);
	return (GLubyte *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
}

void gl_finalize_PBO() {
	glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void draw_volume() {
	GLubyte *pbo_array = gl_prepare_PBO();

	init_view(WIN_WIDTH, WIN_HEIGHT, 3.5f);	
	float3 origin = {0,0,0}, direction = {0,0,0};
	for(int row = 0; row < WIN_HEIGHT; row++)
		for(int col = 0; col < WIN_WIDTH; col++)
		{	
			get_view_ray(row, col, &origin, &direction);
			float4 color = render_ray_alt(origin, direction);
			*pbo_array++ = (unsigned char) map_interval(color.r,256);
			*pbo_array++ = (unsigned char) map_interval(color.g,256);
			*pbo_array++ = (unsigned char) map_interval(color.b,256);
			*pbo_array++ = 255;
		}

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
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
		glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glutSwapBuffers();
}

void keyboard_callback(unsigned char key, int x, int y) {
	if (key=='t') {
		glClearColor(1,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		glutWireTeapot(0.5);
		glutSwapBuffers();
	}
	if (key=='r') {
		draw_random();
	}
	if (key=='b') {
		glutSwapBuffers();
	}
	if (key=='v') {
		draw_volume();
	}
	if (strchr("wsadqe1234", key)) {
		switch (key) {
			case 'w': camera_up(); break;
			case 's': camera_down(); break;
			case 'a': camera_left(); break;
			case 'd': camera_right(); break;
			case 'q': camera_zoom_in(); break;
			case 'e': camera_zoom_out(); break;
			case '1': set_camera_position_deg(2,45,45); break;
			case '2': set_camera_position_deg(2,135,225); break;
			case '3': set_camera_position_deg(2,225,225); break;
			case '4': set_camera_position_deg(-2,0,0); break;
		}
		draw_volume();
	}
	if (key=='z') {
		glutDestroyWindow(window_id);
		glDeleteBuffersARB(1, &pbo_id);
		exit(0);
	}

}

int main(int argc, char **argv) {

	if (load_model(FILE_NAME) != 0) {
		fprintf(stderr, "File error: %s\n", FILE_NAME);
		exit(1);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutInitWindowPosition(100,100);
	window_id = glutCreateWindow("Naive Volume Rendering");
	glutDisplayFunc(display_callback);
	glutKeyboardFunc(keyboard_callback);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
		glutDestroyWindow(window_id);
		exit(1);
	}
	printf("Using GLEW %s\n", glewGetString(GLEW_VERSION));
	if (!GLEW_VERSION_2_0) {
		printf("OpenGL 2.0 is not supported. Exiting...\n");
		glutDestroyWindow(window_id);
		exit(1);
	}
	printf("OpenGL 2.0 is supported.\n");

	glGenBuffersARB(1, &pbo_id);					// pouzivat ARB extensions metody, alebo nativne?

	draw_volume();
	glutMainLoop();
	return 0;
}
