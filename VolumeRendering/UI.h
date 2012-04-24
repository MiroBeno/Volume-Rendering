#ifndef _UI_H_
#define _UI_H_

#include "glew.h"
#include "glui.h"

#include "Raycaster.h"
#include "Renderer.h"

class UI {
	public:
		static char app_name[];
		static int window_id, subwindow_id;
		static GLUI *glui_subwindow;
		static ushort2 window_size;
		static bool window_resize_flag;
		static bool	subwindow_visible;
		static void toggle_tf_editor();
		static void cleanup();
		static void init_gl(int argc, char **argv);
		static void start();
};

#endif