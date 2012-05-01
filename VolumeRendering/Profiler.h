#ifndef _PROFILER_H_
#define _PROFILER_H_

#include "common.h"

#define MAX_CONFIG_COUNT 10
#define MAX_SAMPLE_COUNT 1000
#define LAST_SAMPLE_COUNT 300

class Profiler {

	private:
		static float data[MAX_CONFIG_COUNT][RENDERER_COUNT][MAX_SAMPLE_COUNT];
		static int counters[MAX_CONFIG_COUNT][RENDERER_COUNT];
		static int current_config, current_renderer, current_method;

	public:
		static void init();
		static void destroy();
		static void reset();
		static void set_config(int config);
		static void start(int renderer, int method);
		static float stop();
		static float average(int config, int renderer);
		static float maximum(int config, int renderer);
		static void print_config(int config);
		static float time_ms;
		static float last_times[LAST_SAMPLE_COUNT];
		static int last_times_counter;
};

#endif