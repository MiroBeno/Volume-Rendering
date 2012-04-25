#include "UI.h"

char UI::app_name[256] = "VR:";
ushort2 UI::window_size = {INT_WIN_WIDTH, INT_WIN_HEIGHT};
bool UI::window_resize_flag = false;
int UI::main_window_id, UI::tf_editor_id;
GLUI *UI::glui_panel;
bool UI::tf_editor_visible = true;

static Renderer **renderers;
static int *renderer_id;
static void (*draw_function)();
static void (*exit_function)();
static short4 mouse_state = {0, 0, GLUT_LEFT_BUTTON, GLUT_UP};
static short2 auto_rotate = {0, 0};
static char title_string[256];
static char renderer_names[RENDERER_COUNT][256] = {"CPU", "CUDA Straightforward", "CUDA Constant Memory", "CUDA CM + GL interop", "CUDA CM + 3D Texture Memory + GLI"};
/**/static float view_rotate[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
/**/static float light_rotate[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

/****************************************/
// Callbacks helper functions
/****************************************/

void toggle_auto_rotate(int value) {
	if (auto_rotate.x == 0 && auto_rotate.y == 0) {
		auto_rotate = make_short2(-5, -5);
		printf("Autorotation: on\n");
	}
	else {
		auto_rotate = make_short2(0, 0);
		printf("Autorotation: off\n");
	}
}

void set_renderer(int id) {
	if ((id < 0) || (id >= RENDERER_COUNT))
		return;
	*renderer_id = id;
	printf("Setting renderer: %s\n", renderer_names[id]);
}

/****************************************/
// Main graphic window callbacks
/****************************************/

void display_callback(void) {
	//printf("Main window display callback...\n");
	//draw_volume();
	draw_function();
	sprintf(title_string, "%s %s @ %.2f ms / %.2f ms (%dx%d)", UI::app_name, renderer_names[*renderer_id], 
		Profiler::time_ms, 0.0f, UI::window_size.x, UI::window_size.y);
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
	glutSetWindow(UI::main_window_id);
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
		case 'o': RaycasterBase::change_ray_step(-0.001f, false); break;
		case 'p': RaycasterBase::change_ray_step(0.001f, false); break;
		case 'n': RaycasterBase::change_ray_threshold(-0.05f, false); break;
		case 'm': RaycasterBase::change_ray_threshold(0.05f, false); break;
		case 'l': RaycasterBase::toggle_esl(); break;
		case '[': RaycasterBase::change_light_intensity(-0.05f, false); break;
		case ']': RaycasterBase::change_light_intensity(0.05f, false); break;
		case '-': ViewBase::toggle_perspective(); break;
		case '`': set_renderer(0); break;
		case '1': set_renderer(1); break;
		case '2': set_renderer(2); break;
		case '3': set_renderer(3); break;
		case '4': set_renderer(4); break;
		case '7': ViewBase::set_camera_position(3,-45,-45); break;
		case '8': ViewBase::set_camera_position(3,0,0); break;
		case '9': ViewBase::set_camera_position(3,-90,0); break;
		case '0': ViewBase::set_camera_position(3,180,-90); break;
		case 'r': toggle_auto_rotate(0); break;
		case 't': UI::toggle_tf_editor(); break;
		case 'v': idle_callback(); break;
	}
	if (key==27) {
		exit_function();
	}
	UI::glui_panel->sync_live();
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
	UI::set_viewport_size(width, height);
}

/****************************************/
// Transfer function editor window callbacks
/****************************************/

void display_tfe_callback(void) {
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

void motion_tfe_callback(int x, int y) {
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
	for (int i=0; i < RENDERER_COUNT; i++)
		renderers[i]->set_transfer_fn(RaycasterBase::raycaster);
}

void mouse_tfe_callback(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	motion_tfe_callback(x, y);
}

/****************************************/
// Common UI class functions
/****************************************/

void UI::set_viewport_size(int width, int height) {				
	if (window_size.x == width && window_size.y == height)
		return;
	window_size.x = width;
	window_size.y = height;
	window_resize_flag = true;
	if (window_size.x > 16 && window_size.y > 16) {
		glutSetWindow(main_window_id);
		glViewport(0, 0, window_size.x, window_size.y);
		glutSetWindow(tf_editor_id);										
		glutPositionWindow(window_size.x/20, (window_size.y/20)*17);
		glutReshapeWindow((window_size.x*18)/20, window_size.y/8);
	}
}

void UI::toggle_tf_editor() {
	glutSetWindow(tf_editor_id);
	tf_editor_visible ? glutHideWindow() : glutShowWindow();
	tf_editor_visible = !tf_editor_visible;
}

void UI::destroy() {
	glutDestroyWindow(main_window_id);
}

void UI::init_glui() {
	printf("Initializing GLUI version %.2f...\n", GLUI_Master.get_version());
	glui_panel = GLUI_Master.create_glui_subwindow(main_window_id, GLUI_SUBWINDOW_RIGHT);
	//glui_subwindow->set_main_gfx_window(window_id);
	GLUI_Rollout *renderers_panel = new GLUI_Rollout(glui_panel, "Renderers", true);
	new GLUI_StaticText(renderers_panel, "CPU:");
	new GLUI_Button(renderers_panel, "CPU", 0, set_renderer);
	new GLUI_StaticText(renderers_panel, "GPU:");
	new GLUI_Button(renderers_panel, "CUDA", 1, set_renderer);
	new GLUI_Button(renderers_panel, "CUDA CMem", 2, set_renderer);
	new GLUI_Button(renderers_panel, "CUDA GL Interop", 3, set_renderer);
	new GLUI_Button(renderers_panel, "CUDA TexMem", 4, set_renderer);

	GLUI_Rollout *view_panel = new GLUI_Rollout(glui_panel, "View", true);
	GLUI_Panel *camera_panel = new GLUI_Panel(view_panel, "", false);
	new GLUI_Column(camera_panel, false);
	GLUI_Rotation *rotation = 
		new GLUI_Rotation(camera_panel, "Rotation", view_rotate);
	rotation->set_spin(1.0);
	GLUI_Button *auto_rotate_button = new GLUI_Button(camera_panel, "Auto", 0, toggle_auto_rotate);
	auto_rotate_button->set_h(18);
	auto_rotate_button->set_w(18);
	new GLUI_Column(camera_panel, false);
	GLUI_Translation *translation = 
		new GLUI_Translation(camera_panel, "Translation", GLUI_TRANSLATION_Z, NULL);
	translation->set_speed(0.005f);
	GLUI_RadioGroup *projection = new GLUI_RadioGroup(view_panel);
	projection->set_alignment(GLUI_ALIGN_CENTER);
	new GLUI_RadioButton(projection, "Orthogonal");
	new GLUI_RadioButton(projection, "Perspective");

	GLUI_Rollout *optimizations_panel = new GLUI_Rollout(glui_panel, "Optimizations", true);
	new GLUI_Checkbox(optimizations_panel, "Empty space leaping", (int *) &RaycasterBase::raycaster.esl);
	new GLUI_Checkbox(optimizations_panel, "Early ray termination:");
	GLUI_Spinner *ert = new GLUI_Spinner(optimizations_panel, "  Threshold", &RaycasterBase::raycaster.ray_threshold);
	ert->set_float_limits(0.5f, 1.0f);
	ert->set_speed(10.0f);
	new GLUI_Checkbox(optimizations_panel, "Ray sampling rate:");
	GLUI_Scrollbar *step = new GLUI_Scrollbar(optimizations_panel, "Rate", GLUI_SCROLL_HORIZONTAL, &RaycasterBase::raycaster.ray_step);
	step->set_float_limits(RaycasterBase::ray_step_limits.x, RaycasterBase::ray_step_limits.y);
	new GLUI_Checkbox(optimizations_panel, "Image downscaling:");
	new GLUI_StaticText(optimizations_panel, "  Resolution: x ");
	GLUI_Scrollbar *scale = new GLUI_Scrollbar(optimizations_panel, "Scale", GLUI_SCROLL_HORIZONTAL);
	scale->set_float_limits(0.25f, 1.0f);

	GLUI_Rollout *lighting_panel = new GLUI_Rollout(glui_panel, "Lighting", true);
	new GLUI_Checkbox(lighting_panel, "Enabled");
	GLUI_Rotation *light_rotation = 
		new GLUI_Rotation(lighting_panel, "Light position", light_rotate);
	light_rotation->set_spin(1.0);
	new GLUI_StaticText(lighting_panel, "Intensity:");
	GLUI_Scrollbar *intensity = new GLUI_Scrollbar(lighting_panel, "Intensity", GLUI_SCROLL_HORIZONTAL, &RaycasterBase::raycaster.light_kd);
	intensity->set_float_limits(0.0f, 2.0f);

	GLUI_Rollout *controls_panel = new GLUI_Rollout(glui_panel, "Controls", true);
	new GLUI_Checkbox(controls_panel, "Fullscreen");
	new GLUI_Checkbox(controls_panel, "Control panel");
	new GLUI_Checkbox(controls_panel, "Render time stats");
	new GLUI_Checkbox(controls_panel, "Transfer function editor");
	/*new GLUI_Checkbox(controls_panel, "Data histogram");
	new GLUI_Checkbox(controls_panel, "High precision (^4)");
	new GLUI_Button(controls_panel, "Reset");*/
	GLUI_Button *quit_button = new GLUI_Button(glui_panel, "Quit");
	quit_button->set_w(150);
	GLUI_FileBrowser *fb = new GLUI_FileBrowser(glui_panel, "Load File");
	new GLUI_StaticText(fb, "Current file: filename");
	new GLUI_StaticText(fb, "Dimensions: x x ");
	new GLUI_Column(fb, false);
}

void UI::init(Renderer **rends, int *rend_id, void (*draw_fn)(), void (*exit_fn)()) {
	renderers = rends;
	renderer_id = rend_id;
	draw_function = draw_fn;
	exit_function = exit_fn;

	printf("Initializing GLUT...\n");
	int dummy_i = 1;
    char* dummy = "";
    glutInit(&dummy_i, &dummy);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);
	glutInitWindowSize(window_size.x + 165, window_size.y);
	glutInitWindowPosition(700, 1);

	main_window_id = glutCreateWindow(app_name);
	glutMotionFunc(motion_callback);
	GLUI_Master.set_glutDisplayFunc(display_callback);
	GLUI_Master.set_glutIdleFunc(idle_callback);
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

	tf_editor_id = glutCreateSubWindow(main_window_id, window_size.x/20, (window_size.y/20)*17, (window_size.x*18)/20, window_size.y/8);
	glutDisplayFunc(display_tfe_callback);
	glutMouseFunc(mouse_tfe_callback);
	glutMotionFunc(motion_tfe_callback);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glOrtho(0.0, TF_SIZE - 1, 0.0, 1.0, 0.0, 1.0);
	glClearColor(0, 0, 0, 1);
	//if (!subwindow_visible)
	//	glutHideWindow();

	init_glui();
	glutSetWindow(main_window_id);
	//glutFullScreen();

	printf("Initializing GLEW version %s...\n", glewGetString(GLEW_VERSION));
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	if (!GLEW_VERSION_2_0) {
		fprintf(stderr, "Error: OpenGL 2.0 is not supported\n");
		exit(EXIT_FAILURE);
	}
}

void UI::print_usage() {
	printf("\nUse '`1234' to change renderer\n"); 
	printf("    'wasd' and '7890' and left mouse button to manipulate camera rotation\n");
	printf("    'r' to toggle autorotation\n");
	printf("    'qe' and right mouse button to manipulate camera translation\n");
	printf("    'op' to change ray sampling rate\n");
	printf("    'nm' to change ray accumulation threshold\n");
	printf("    'l' to toggle empty space leaping\n");
	printf("    '[]' to change volume illumination intensity\n");
	printf("    middle mouse button to change light source position\n");
	printf("    '-' to toggle perspective and orthogonal projection\n");
	printf("    't' to toggle transfer function editor\n");
	printf("\n");
}

void UI::start() {
	glutMainLoop();
}
