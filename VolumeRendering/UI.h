#ifndef _UI_H_
#define _UI_H_

#include "glew.h"
#include "glui.h"

#include "Raycaster.h"
#include "Renderer.h"
#include "Profiler.h"
#include "constants.h"

class UI {
	public:
		static char app_name[];
		static ushort2 window_size;
		static bool window_resize_flag;
		static int main_window_id, tf_editor_id;
		static GLUI *glui_panel;
		static bool	tf_editor_visible;
		static bool auto_rotation;
		static void toggle_tf_editor();
		static void set_viewport_size(int width, int height);
		static void destroy();
		static void init(Renderer *rends[], int *rend_id, void (*draw_fn)(), void (*exit_fn)());
		static void print_usage();
		static void start();
	private:
		static void init_glui();
};

#endif