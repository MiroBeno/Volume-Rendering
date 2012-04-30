#include "UI.h"
#include "glui.h"		// includes glut.h

char UI::app_name[256] = "VR:";
bool UI::viewport_resized_flag = true;
Renderer **UI::renderers;
int *UI::renderer_id;
void (*UI::draw_function)();
void (*UI::exit_function)();

static GLUI *glui_panel;
static GLUI_Checkbox *ert_checkbox, *light_enabled_checkbox;
static GLUI_Spinner *ert_spinner;
static GLUI_Scrollbar *ray_step_scroll, *light_intensity_scroll, *scale_scroll, *bg_color_scroll;
static GLUI_StaticText *light_text, *resolution_text, *gpu_name_text, *file_name_text, *volume_dims_text;
static GLUI_Rotation *light_rotation;
static GLUI_FileBrowser *file_browser;

static int main_window_id, tf_editor_id;
static int fullscreen = false, tf_editor_visible = true, glui_panel_visible = true, profiler_graph_visible = false;
static int2 pre_fullscreen_size;
static float bg_color = 0.25f;
static float viewport_scale = 1.0f;
static short4 mouse_state = {0, 0, GLUT_LEFT_BUTTON, GLUT_UP};
static short2 auto_rotate = {0, 0};
static float4 profiler_pos = {0.7, 0.92, 0.98, 0.98};
static float2 profiler_delta = {(profiler_pos.z - profiler_pos.x) / (LAST_SAMPLE_COUNT - 1), (profiler_pos.w - profiler_pos.y)};
static char renderer_names[RENDERER_COUNT][256] = {"CPU", "CUDA Straightforward", "CUDA Constant Memory", "CUDA CM + GL interop", "CUDA CM + 3D Texture Memory + GLI"};
static char text_buffer[200];

/****************************************/
// Callbacks helper functions
/****************************************/

void set_renderer_callback(int id) {
	if ((id < 0) || (id >= RENDERER_COUNT))
		return;
	*UI::renderer_id = id;
	printf("Setting renderer: %s\n", renderer_names[id]);
}

void exit_callback(int value) {
	UI::exit_function();
}

void sync_glui() {
	if (RaycasterBase::raycaster.ray_threshold < 1.0f) {
		ert_checkbox->set_int_val(1);
		ert_spinner->enable();
	}
	if (RaycasterBase::raycaster.light_kd > 0.0f) {
		light_enabled_checkbox->set_int_val(1);
		light_intensity_scroll->enable();
		light_text->enable();
		light_rotation->enable();
	}
	glui_panel->sync_live();
}

/****************************************/
// Main graphic window callbacks
/****************************************/

void draw_profiler_graph() {
	glBegin(GL_QUADS);
	glColor4f(0, 0, 0, 0.2f);
	glVertex2f(profiler_pos.x, profiler_pos.y); glVertex2f(profiler_pos.z, profiler_pos.y);
	glVertex2f(profiler_pos.z, profiler_pos.w); glVertex2f(profiler_pos.x, profiler_pos.w);
	glEnd();
	glBegin(GL_QUAD_STRIP);
	for (int i=0; i < LAST_SAMPLE_COUNT; i++) {
		int index = (i + Profiler::last_times_counter) % LAST_SAMPLE_COUNT;
		float sample = Profiler::last_times[index] / 50.f;
		glColor4f(1 - (sample - 2.0f), 1 - (sample - 1.0f), 1 - (sample - 1.0f), 0.7f);
		glVertex2f(profiler_pos.x + i * profiler_delta.x, profiler_pos.y);
		glVertex2f(profiler_pos.x + i * profiler_delta.x, profiler_pos.y + flmin(sample, 1.0f) * profiler_delta.y);
	}
	glEnd();
}

void display_callback(void) {
	//printf("Main window display callback...\n");
	/**/float frame = 0;//Profiler::stop();
	UI::draw_function();
	sprintf(text_buffer, "%s %s @ %.2f ms / %.2f ms (%dx%d)", UI::app_name, renderer_names[*UI::renderer_id], 
		Profiler::time_ms, frame, ViewBase::view.dims.x, ViewBase::view.dims.y);
	glutSetWindowTitle(text_buffer);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ViewBase::view.dims.x, ViewBase::view.dims.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glEnable(GL_TEXTURE_2D);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(0, 0);
	glTexCoord2f(1, 0); glVertex2f(1, 0);
	glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(0, 1);
	glEnd();
	glDisable(GL_TEXTURE_2D);
	if (profiler_graph_visible)
		draw_profiler_graph();
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
	glutSetWindow(main_window_id);
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
		case '-': ViewBase::toggle_perspective(false); break;
		case '`': set_renderer_callback(0); break;
		case '1': set_renderer_callback(1); break;
		case '2': set_renderer_callback(2); break;
		case '3': set_renderer_callback(3); break;
		case '4': set_renderer_callback(4); break;
		case '7': ViewBase::set_camera_position(3,-45,-45); break;
		case '8': ViewBase::set_camera_position(3,0,0); break;
		case '9': ViewBase::set_camera_position(3,-90,0); break;
		case '0': ViewBase::set_camera_position(3,180,-90); break;
		case 'r': UI::toggle_auto_rotate(false); break;
		case 'c': UI::toggle_glui_panel(false); break;
		case 't': UI::toggle_tf_editor(false); break;
		case 'y': profiler_graph_visible = !profiler_graph_visible; break;
		case 'f': UI::toggle_fullscreen(false); break;
		case 'v': idle_callback(); break;
	}
	if (key==27) {
		exit_callback(0);
	}
	sync_glui();
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
	int left, top;
	if (glui_panel_visible) 
		GLUI_Master.get_viewport_area(&left, &top, &w, &h);
	glutSetWindow(main_window_id);
	glViewport(0, 0, w, h);
	if (w >= 16 && h >= 16) {
		glutSetWindow(tf_editor_id);										
		glutPositionWindow(w/20, (h/20)*17);
		glutReshapeWindow((w*18)/20, h/8);
	}
	ViewBase::set_viewport_dims(w, h, viewport_scale);
	UI::viewport_resized_flag = true;
	sprintf(text_buffer, "  Resolution: %dx%d", ViewBase::view.dims.x, ViewBase::view.dims.y);
	resolution_text->set_text(text_buffer);
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
		UI::renderers[i]->set_transfer_fn(RaycasterBase::raycaster);
}

void mouse_tfe_callback(int button, int state, int x, int y) {
	mouse_state.x = x;
	mouse_state.y = y;
	mouse_state.z = button;
	motion_tfe_callback(x, y);
}

/****************************************/
// GLUI callback
/****************************************/

void glui_callback(GLUI_Control *source) {
	if (source == ert_checkbox) {
		if (ert_checkbox->get_int_val()) {
			RaycasterBase::change_ray_threshold(0.95f, true);
			ert_spinner->enable();
		}
		else {
			RaycasterBase::change_ray_threshold(1, true);
			ert_spinner->disable();
		}
		glui_panel->sync_live();
	}
	else if (source == light_enabled_checkbox) {
		if (light_enabled_checkbox->get_int_val()) {
			RaycasterBase::change_light_intensity(0.6f, true);
			light_intensity_scroll->enable();
			light_text->enable();
			light_rotation->enable();
		}
		else {
			RaycasterBase::change_light_intensity(0.0f, true);
			light_intensity_scroll->disable();
			light_text->disable();
			light_rotation->disable();
		}
		glui_panel->sync_live();
	}
	else if (source == scale_scroll) {
		glutSetWindow(main_window_id);
		reshape_callback(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
	}
	else if (source == light_rotation) {
		float light_val[16];
		light_rotation->get_float_array_val(light_val);
		/*printf("%.3f %.3f %.3f \n%.3f %.3f %.3f \n%.3f %.3f %.3f\n\n", 
			light_val[0], light_val[1], light_val[2], light_val[4], light_val[5], light_val[6], 
			light_val[8], light_val[9], light_val[10]);*/
	}
	else if (source == file_browser) {
		int file_loaded = ModelBase::load_model(file_browser->get_file());
		if (file_loaded == 0) {
			RaycasterBase::set_volume(ModelBase::volume);
			for (int i=0; i < RENDERER_COUNT; i++) {
				UI::renderers[i]->set_volume(RaycasterBase::raycaster.volume);
				UI::renderers[i]->set_transfer_fn(RaycasterBase::raycaster);
			}
			float ray_step_value = RaycasterBase::raycaster.ray_step;
			ray_step_scroll->set_float_limits(RaycasterBase::ray_step_limits.x, RaycasterBase::ray_step_limits.y);
			ray_step_scroll->set_float_val(ray_step_value);
			sprintf(text_buffer, "File: %s", file_browser->get_file());
			file_name_text->set_text(text_buffer);
			sprintf(text_buffer, "Dimensions: %dx%dx%d", ModelBase::volume.dims.x, ModelBase::volume.dims.y, ModelBase::volume.dims.z);
			volume_dims_text->set_text(text_buffer);
		}
	}
	else if (source == bg_color_scroll) {
		glutSetWindow(main_window_id);
		glClearColor(bg_color, bg_color, bg_color, 1);
	}
}

/****************************************/
// Public UI class functions
/****************************************/

void UI::toggle_fullscreen(int update_mode) {
	if (!update_mode)
		fullscreen = !fullscreen;
	glutSetWindow(main_window_id);
	if (fullscreen) {
		pre_fullscreen_size = make_int2(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
		glutFullScreen();
	}
	else 
		glutReshapeWindow(pre_fullscreen_size.x, pre_fullscreen_size.y);
}

void UI::toggle_tf_editor(int update_mode) {
	if (!update_mode)
		tf_editor_visible = !tf_editor_visible;
	glutSetWindow(tf_editor_id);
	tf_editor_visible ? glutShowWindow() : glutHideWindow();
}

void UI::toggle_glui_panel(int update_mode) {
	if (!update_mode)
		glui_panel_visible = !glui_panel_visible;
	glui_panel_visible ? glui_panel->show() : glui_panel->hide();
	glutSetWindow(main_window_id);
	reshape_callback(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
}

void UI::toggle_auto_rotate(int update_mode) {
	if (auto_rotate.x == 0 && auto_rotate.y == 0) 
		auto_rotate = make_short2(-5, -5);
	else 
		auto_rotate = make_short2(0, 0);
}

void UI::set_gpu_name(const char *name) {
	gpu_name_text->set_text(name);					// osetrit velkost
}

void UI::destroy() {
	glutDestroyWindow(main_window_id);
}

void UI::init_glui() {
	printf("Initializing GLUI version %.2f...\n", GLUI_Master.get_version());
	glui_panel = GLUI_Master.create_glui_subwindow(main_window_id, GLUI_SUBWINDOW_RIGHT);
	glui_panel->set_main_gfx_window(main_window_id);

	GLUI_Rollout *renderers_panel = new GLUI_Rollout(glui_panel, "Renderers", true);
	new GLUI_StaticText(renderers_panel, "CPU");
	new GLUI_Button(renderers_panel, "CPU", 0, set_renderer_callback);
	gpu_name_text = new GLUI_StaticText(renderers_panel, "GPU");
	new GLUI_Button(renderers_panel, "CUDA", 1, set_renderer_callback);
	new GLUI_Button(renderers_panel, "CUDA CMem", 2, set_renderer_callback);
	new GLUI_Button(renderers_panel, "CUDA GL Interop", 3, set_renderer_callback);
	new GLUI_Button(renderers_panel, "CUDA TexMem", 4, set_renderer_callback);

	GLUI_Rollout *view_panel = new GLUI_Rollout(glui_panel, "View", true);
	GLUI_Panel *camera_panel = new GLUI_Panel(view_panel, "", false);
	new GLUI_Column(camera_panel, false);
	GLUI_Rotation *rotation = new GLUI_Rotation(camera_panel, "Rotation", NULL, 0, glui_callback);
	rotation->set_spin(1.0);
	GLUI_Button *auto_rotate_button = new GLUI_Button(camera_panel, "Auto", 0, toggle_auto_rotate);
	auto_rotate_button->set_h(18);
	auto_rotate_button->set_w(18);
	new GLUI_Column(camera_panel, false);
	GLUI_Translation *translation = new GLUI_Translation(camera_panel, "Translation", GLUI_TRANSLATION_Z, NULL);
	translation->set_speed(0.005f);
	GLUI_RadioGroup *projection = new GLUI_RadioGroup(view_panel, (int *) &ViewBase::view.perspective, true, ViewBase::toggle_perspective);
	projection->set_alignment(GLUI_ALIGN_CENTER);
	new GLUI_RadioButton(projection, "Orthogonal");
	new GLUI_RadioButton(projection, "Perspective");

	GLUI_Rollout *optimizations_panel = new GLUI_Rollout(glui_panel, "Optimizations", true);
	new GLUI_Checkbox(optimizations_panel, "Empty space leaping", (int *) &RaycasterBase::raycaster.esl);
	ert_checkbox = new GLUI_Checkbox(optimizations_panel, "Early ray termination:", 0, 0, glui_callback);
	ert_checkbox->set_int_val(1);
	ert_spinner = new GLUI_Spinner(optimizations_panel, "  Threshold", &RaycasterBase::raycaster.ray_threshold);
	ert_spinner->set_float_limits(0.5f, 1.0f);
	ert_spinner->set_speed(10.0f);
	new GLUI_StaticText(optimizations_panel, "Ray sampling step:");
	ray_step_scroll = new GLUI_Scrollbar(optimizations_panel, "Step", GLUI_SCROLL_HORIZONTAL, &RaycasterBase::raycaster.ray_step);
	ray_step_scroll->set_float_limits(RaycasterBase::ray_step_limits.x, RaycasterBase::ray_step_limits.y);
	new GLUI_StaticText(optimizations_panel, "Image downscaling:");
	resolution_text = new GLUI_StaticText(optimizations_panel, "  Resolution: ?");
	scale_scroll = new GLUI_Scrollbar(optimizations_panel, "Scale", GLUI_SCROLL_HORIZONTAL, &viewport_scale, 0, glui_callback);
	scale_scroll->set_float_limits(0.25f, 1.0f);

	GLUI_Rollout *lighting_panel = new GLUI_Rollout(glui_panel, "Lighting", true);
	light_enabled_checkbox = new GLUI_Checkbox(lighting_panel, "Enabled", 0, 0, glui_callback);
	light_enabled_checkbox->set_int_val(1);
	light_rotation = new GLUI_Rotation(lighting_panel, "Light position", NULL, 0, glui_callback);
	light_rotation->set_spin(1.0);
	light_text = new GLUI_StaticText(lighting_panel, "Intensity:");
	light_intensity_scroll = new GLUI_Scrollbar(lighting_panel, "Intensity", GLUI_SCROLL_HORIZONTAL, &RaycasterBase::raycaster.light_kd);
	light_intensity_scroll->set_float_limits(0.0f, 2.0f);

	GLUI_Rollout *controls_panel = new GLUI_Rollout(glui_panel, "Controls", true);
	new GLUI_Checkbox(controls_panel, "Fullscreen", &fullscreen, true, UI::toggle_fullscreen);
	new GLUI_Checkbox(controls_panel, "Control panel", &glui_panel_visible, true, UI::toggle_glui_panel);
	new GLUI_Checkbox(controls_panel, "Render time graph", &profiler_graph_visible);
	new GLUI_Checkbox(controls_panel, "Transfer function editor", &tf_editor_visible, true, UI::toggle_tf_editor);
	new GLUI_StaticText(controls_panel, "Background");
	bg_color_scroll = new GLUI_Scrollbar(controls_panel, "Background", GLUI_SCROLL_HORIZONTAL, &bg_color, 0, glui_callback);
	bg_color_scroll->set_float_limits(0.0f, 1.0f);
	/*new GLUI_Checkbox(controls_panel, "Data histogram");
	new GLUI_Checkbox(controls_panel, "High precision (^4)");
	new GLUI_Button(controls_panel, "Reset");*/

	GLUI_Button *quit_button = new GLUI_Button(glui_panel, "Quit", 0, exit_callback);
	quit_button->set_w(150);

	file_browser = new GLUI_FileBrowser(glui_panel, "Load File", 1, 0, glui_callback);
	file_name_text = new GLUI_StaticText(file_browser, "File: ?");
	sprintf(text_buffer, "Dimensions: %dx%dx%d", ModelBase::volume.dims.x, ModelBase::volume.dims.y, ModelBase::volume.dims.z);
	volume_dims_text = new GLUI_StaticText(file_browser, text_buffer);
	new GLUI_Column(file_browser, false);
}

void UI::init(Renderer **rends, int *rend_id, void (*draw_fn)(), void (*exit_fn)()) {
	renderers = rends;
	renderer_id = rend_id;
	draw_function = draw_fn;
	exit_function = exit_fn;

	printf("Initializing GLUT...\n");
	int dummy_i = 1;
    char* dummy = "";
	ushort2 view_size = ViewBase::view.dims;
    glutInit(&dummy_i, &dummy);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);
	glutInitWindowSize(view_size.x + 192, view_size.y);
	glutInitWindowPosition(700, 1);

	main_window_id = glutCreateWindow(app_name);
	glutMotionFunc(motion_callback);
	glutDisplayFunc(display_callback);
	GLUI_Master.set_glutIdleFunc(idle_callback);
	GLUI_Master.set_glutKeyboardFunc(keyboard_callback);
	GLUI_Master.set_glutMouseFunc(mouse_callback);
	GLUI_Master.set_glutReshapeFunc(reshape_callback);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, view_size.x, view_size.y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glClearColor(bg_color, bg_color, bg_color, 1);

	tf_editor_id = glutCreateSubWindow(main_window_id, view_size.x/20, (view_size.y/20)*17, (view_size.x*18)/20, view_size.y/8);
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
}

void UI::print_usage() {
	printf("\nUse '`1234' to change renderer\n"); 
	printf("    'wasd' and '7890' and left mouse button to manipulate camera rotation\n");
	printf("    'r' to toggle autorotation\n");
	printf("    'qe' and right mouse button to manipulate camera translation\n");
	printf("    'op' to change ray sampling step size\n");
	printf("    'nm' to change ray accumulation threshold\n");
	printf("    'l' to toggle empty space leaping\n");
	printf("    '[]' to change volume illumination intensity\n");
	printf("    middle mouse button to change light source position\n");
	printf("    '-' to toggle perspective and orthogonal projection\n");
	printf("    't' to toggle transfer function editor\n");
	printf("    'y' to toggle render time graph\n");
	printf("\n");
}

void UI::start() {
	printf("Entering main event loop...\n");
	glutMainLoop();
}
