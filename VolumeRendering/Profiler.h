#ifndef _PROFILER_H_
#define _PROFILER_H_

#include "common.h"

#define MAX_CONFIG_COUNT 100
#define LAST_SAMPLE_COUNT 300
#define MIN_SAMPLE_STAT 8

struct Stat {
	unsigned int samples;
	float time_max;
	double time_sum;
};

class Profiler {

	private:
		static Stat statistics[MAX_CONFIG_COUNT][RENDERER_COUNT];
		static int current_config, current_renderer, current_method;
		
	public:
		static void init();
		static void destroy();
		static void reset_config(int config);
		static void start(int renderer, int method);
		static float stop();
		static void print_samples(int config);
		static void print_avg(int config);
		static void print_max(int config);
		static float time_ms;
		static float last_times[LAST_SAMPLE_COUNT];
		static int last_times_counter;
};

#endif