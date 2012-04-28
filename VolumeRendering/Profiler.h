#ifndef _PROFILER_H_
#define _PROFILER_H_

#include <ctime>
#include "cuda_utils.h"
#include "constants.h"

#define CONFIGURATION_COUNT 10
#define MAX_SAMPLE_COUNT 1000
#define LAST_SAMPLE_COUNT 300

class Profiler {

private:
	static float data[RENDERER_COUNT][CONFIGURATION_COUNT][MAX_SAMPLE_COUNT];
	static int counters[RENDERER_COUNT][CONFIGURATION_COUNT];
	static int current_renderer, current_configuration, current_method;
	static clock_t last_clock;
	static cudaEvent_t start_event, stop_event;

public:
	static void init();
	static void destroy();
	static void reset();
	static void start(int renderer_id, int configuration, int method);
	static float stop();
	static float average(int renderer_id, int configuration);
	static void dump(const char *filePath);
	static float time_ms;
	static float last_times[LAST_SAMPLE_COUNT];
	static int last_times_counter;
};

#endif