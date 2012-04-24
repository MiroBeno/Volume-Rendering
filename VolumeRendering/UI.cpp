#include "UI.h"

char UI::app_name[256] = "VR:";
int UI::window_id, UI::subwindow_id;
GLUI *UI::glui_subwindow;
ushort2 UI::window_size = {INT_WIN_WIDTH, INT_WIN_HEIGHT};
bool UI::window_resize_flag = false;
bool UI::subwindow_visible = true;

extern const int RENDERERS_COUNT;
extern int renderer_id;
extern void draw_volume();
extern void cleanup_and_exit();
extern Renderer *renderers[];
extern float2 elapsed_time;

static short4 mouse_state = {0, 0, GLUT_LEFT_BUTTON, GLUT_UP};
static short2 auto_rotate = {0, 0};
static char title_string[256];
static char renderer_names[/*RENDERERS_COUNT*/5][256]= {"CPU", "CUDA Straightforward", "CUDA Constant Memory", "CUDA CM + GL interop", "CUDA CM + 3D Texture Memory + GLI"};


void toggle_auto_rotate() {
	if (auto_rotate.x == 0 && auto_rotate.y == 0) {
		auto_rotate = make_short2(-5, -5);
		printf("Autorotation: on\n");
	}
	else {
		auto_rotate = make_short2(0, 0);
		printf("Autorotation: off\n");
	}
}

void display_callback(void) {
	//printf("Main window display callback...\n");
	draw_volume();
	sprintf(title_string, "%s %s @ %.2f ms / %.2f ms (%dx%d)", UI::app_name, renderer_names[renderer_id], 
		elapsed_time.x, elapsed_time.y, UI::window_size.x, UI::window_size.y);
	glutSetWindowTitle(title_string);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, UI::window_size.x, UI::window_size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(0, 0);
	glTexCoord2f(1, 0); glVertex2f(1, 0);
	glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(0, 1);
	glEnd();
	glDisable(GL_BLEND);
	glutSwapBuffers();
}

void idle_callback(void) {
	//printf("Idle callback...\n");
	if (mouse_state.w == GLUT_UP || mouse_state.z != GLUT_LEFT_BUTTON) {
		if (auto_rotate.x != 0)
			ViewBase::camera_right(auto_rotate.x);
		if (auto_rotate.y != 0) 
			ViewBase::camera_down(auto_rotate.y);
	}
	glutSetWindow(UI::window_id);
    glutPostRedisplay();
}

void keyboard_callback(unsigned char key, int x, int y) {
	switch (key) {
		case 'w': ViewBase::camera_down(-5.0f); break;
		case 's': ViewBase::camera_down(5.0f); break;
		case 'a': ViewBase::camera_right(-5.0f); break;
		case 'd': ViewBase::camera_right(5.0f); break;
		case 'q': ViewBase::camera_zoom(0.1f); break;
		case 'e': ViewBase::camera_zoom(-0.1f); break;
		case 'o': RaycasterBase::change_ray_step(0.01f, false); break;
		case 'p': RaycasterBase::change_ray_step(-0.01f, false); break;
		case 'n': RaycasterBase::change_ray_threshold(0.05f, false); break;
		case 'm': RaycasterBase::change_ray_threshold(-0.05f, false); break;
		case 'l': RaycasterBase::toggle_esl(); break;
		case '[': RaycasterBase::raycaster.light_kd -= 0.05; printf("Kd: %.2f\n", RaycasterBase::raycaster.light_kd); break;
		case ']': RaycasterBase::raycaster.light_kd += 0.05; printf("Kd: %.2f\n", RaycasterBase::raycaster.light_kd); break;
		case '-': ViewBase::toggle_perspective(); break;
		case '`': renderer_id = 0; break;
		case '1': renderer_id = 1; break;
		case '2': renderer_id = 2; break;
		case '3': renderer_id = 3; break;
		case '4': renderer_id = 4; break;
		case '7': ViewBase::set_camera_position(3,-45,-45); break;
		case '8': ViewBase::set_camera_position(3,0,0); break;
		case '9': ViewBase::set_camera_position(3,-90,0); break;
		case '0': ViewBase::set_camera_position(3,180,-90); break;
		case 'r': toggle_auto_rotate(); break;
		case 't': UI::toggle_tf_editor(); break;
		case 'v': idle_callback(); break;
	}
	if (key==27) {
		cleanup_and_exit();
	}
	//glui_subwindow->sync_live();
}

void mouse_callback(int button, int state, int x, int y) {
	//printf("Mouse click: button:%i state:%i x:%i y:%i\n", button, state, x, y);
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	mouse_state.w = state;
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) 
			auto_rotate = make_short2(0, 0);
		if (state == GLUT_UP) {
			if (abs(auto_rotate.x) < 8 && abs(auto_rotate.y) < 8)
				auto_rotate = make_short2(0, 0);
		}
	}
}

void motion_callback(int x, int y) {
	if (mouse_state.z == GLUT_LEFT_BUTTON) {	  
		auto_rotate.x = x - mouse_state.x;
		auto_rotate.y = y - mouse_state.y;
		ViewBase::camera_right(auto_rotate.x); 
		ViewBase::camera_down(auto_rotate.y);
	}
	if (mouse_state.z == GLUT_RIGHT_BUTTON) {	  
		ViewBase::camera_zoom(y - mouse_state.y); 
	}
	if (mouse_state.z == GLUT_MIDDLE_BUTTON) {	  
		ViewBase::light_right(x - mouse_state.x); 
		ViewBase::light_down(y - mouse_state.y);
	}
	mouse_state.x = x;
	mouse_state.y = y;
}

void reshape_callback(int w, int h) {				
	//printf("Resizing main window... %i %i\n", w, h);
	int width, height, left, top;
	GLUI_Master.get_viewport_area(&left, &top, &width, &height);
	if (UI::window_size.x == width && UI::window_size.y == height)
		return;
	UI::window_size.x = width;
	UI::window_size.y = height;
	UI::window_resize_flag = true;
	if (UI::window_size.x > 16 && UI::window_size.y > 16) {
		glutSetWindow(UI::window_id);
		glViewport(0, 0, UI::window_size.x, UI::window_size.y);
		glutSetWindow(UI::subwindow_id);										
		glutPositionWindow(UI::window_size.x/20, (UI::window_size.y/20)*17);
		glutReshapeWindow((UI::window_size.x*18)/20, UI::window_size.y/8);
	}
}

void display_callback_sub(void) {
	//printf("Drawing subwindow...\n");
	glClear(GL_COLOR_BUFFER_BIT);	
	float4 *tf = RaycasterBase::base_transfer_fn;
	glBegin(GL_QUAD_STRIP);
	for (int i=0; i < TF_SIZE; i++) {
		glColor4f(tf[i].x, tf[i].y, tf[i].z, 1.0f);
		glVertex2f(i, 0);
		glVertex2f(i, sqrt(sqrt(tf[i].w)));
	}
	glEnd();
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBegin(GL_QUAD_STRIP);
	for (int i=0; i < 256; i++) {
		glColor4f(1, 1, 1, 0.9f);
		glVertex2f(i / (float)TF_RATIO, 0);
		glColor4f(1, 1, 1, 0.1f);
		glVertex2f(i / (float)TF_RATIO, ModelBase::histogram[i]);
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
			//RaycasterBase::base_transfer_fn[sample] = make_float4(0.23f, 0.23f, 0.0f, 0);
		}
		if (mouse_state.z != GLUT_LEFT_BUTTON) 
			intensity = 0;
		RaycasterBase::base_transfer_fn[sample].w = intensity;
	}
	mouse_state.x = x;
	mouse_state.y = y;
	glutPostRedisplay();
	RaycasterBase::update_transfer_fn();
	for (int i=0; i < RENDERERS_COUNT; i++)
		renderers[i]->set_transfer_fn(RaycasterBase::raycaster);
}

void mouse_callback_sub(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	motion_callback_sub(x, y);
}

void UI::toggle_tf_editor() {
	glutSetWindow(subwindow_id);
	subwindow_visible ? glutHideWindow() : glutShowWindow();
	subwindow_visible = !subwindow_visible;
}

void UI::cleanup() {
	glutDestroyWindow(window_id);
}

void UI::init_gl(int argc, char **argv) {
	printf("Initializing GLUT and GLEW...\n");

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);		//GLUT_DOUBLE | GLUT_MULTISAMPLE
	glutInitWindowSize(window_size.x + 165, window_size.y);
	glutInitWindowPosition(500, 1);

	window_id = glutCreateWindow(app_name);
	glutMotionFunc(motion_callback);
	GLUI_Master.set_glutDisplayFunc(display_callback);
	GLUI_Master.set_glutIdleFunc(NULL);
	GLUI_Master.set_glutKeyboardFunc(keyboard_callback);
	GLUI_Master.set_glutMouseFunc(mouse_callback);
	GLUI_Master.set_glutReshapeFunc(reshape_callback);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glViewport(0, 0, window_size.x, window_size.y);
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
	glOrtho(0.0, TF_SIZE - 1, 0.0, 1.0, 0.0, 1.0);
	glClearColor(0, 0, 0, 1);
	//if (!subwindow_visible)
	//	glutHideWindow();

	glui_subwindow = GLUI_Master.create_glui_subwindow(window_id, GLUI_SUBWINDOW_RIGHT);
	//glui_subwindow->set_main_gfx_window(window_id);
	new GLUI_Checkbox(glui_subwindow, "Empty Space Leaping", (int *) &RaycasterBase::raycaster.esl);
	GLUI_Translation *trans_z = 
		new GLUI_Translation( glui_subwindow, "Zoom", GLUI_TRANSLATION_Z, NULL);
	trans_z->set_speed(0.005f);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: Initializing GLEW failed: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Using GLEW %s\n\n", glewGetString(GLEW_VERSION));
	if (!GLEW_VERSION_2_0) {
		printf("Error: OpenGL 2.0 is not supported\n");
		exit(EXIT_FAILURE);
	}
	glutSetWindow(window_id);
}

void UI::start() {
	glutMainLoop();
}