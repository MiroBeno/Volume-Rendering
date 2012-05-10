/****************************************/
// User interface
/****************************************/

#ifndef _UI_H_
#define _UI_H_

#include "common.h"
#include "RaycasterBase.h"
#include "Renderer.h"
#include "Profiler.h"

class UI {
	public:
		static char app_name[];
		static bool viewport_resized_flag;
		static Renderer **renderers;
		static int *renderer_id;
		static void (*draw_function)();
		static void (*exit_function)();
		static void toggle_fullscreen(int update_mode);
		static void toggle_tf_editor(int update_mode);
		static void toggle_glui_panel(int update_mode);
		static void toggle_auto_rotate(int update_mode);
		static void set_gpu_name(const char *name);
		static void destroy();
		static void init(Renderer *rends[], int *rend_id, void (*draw_fn)(), void (*exit_fn)());
		static void print_usage();
		static void start();
	private:
		static void init_glui();
};

#endif