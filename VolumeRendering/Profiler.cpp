#include <time.h>

#include "cuda_utils.h"
#include "Profiler.h"

float Profiler::data[MAX_CONFIG_COUNT][RENDERER_COUNT][MAX_SAMPLE_COUNT];
int Profiler::counters[MAX_CONFIG_COUNT][RENDERER_COUNT];
int Profiler::current_config, Profiler::current_renderer, Profiler::current_method;
float Profiler::time_ms;
float Profiler::last_times[LAST_SAMPLE_COUNT];
int Profiler::last_times_counter;

static clock_t last_clock;
static cudaEvent_t start_event, stop_event;

void Profiler::init() {
	reset();
	cuda_safe_call(cudaEventCreate(&start_event));
	cuda_safe_call(cudaEventCreate(&stop_event));
}

void Profiler::destroy() {
	cuda_safe_call(cudaEventDestroy(start_event));
	cuda_safe_call(cudaEventDestroy(stop_event));
}

void Profiler::reset() {
	for (int c = 0; c < MAX_CONFIG_COUNT; c++)
		for (int r = 0; r < RENDERER_COUNT; r++)
			counters[c][r] = 0;
	current_renderer = current_method = -1;
	current_config = 0;
	time_ms = -1;
	for (int i = 0; i < LAST_SAMPLE_COUNT; i++)
		last_times[i] = 0;
	last_times_counter = 0;
}

void Profiler::set_config(int config) {
	current_config = CLAMP(config, 0, MAX_CONFIG_COUNT);
}

void Profiler::start(int renderer, int method) {
	current_renderer = renderer;
	current_method = method;
	if (current_method == 0) {
		last_clock = clock();
	} else {
		cuda_safe_call(cudaEventRecord(start_event, 0));
	}
}

float Profiler::stop() {
	if (current_renderer == -1 || current_method == -1)
		return -1;
	// measure time
	time_ms = -1;
	if (current_method == 0) {
		time_ms = (clock() - last_clock) / (CLOCKS_PER_SEC / 1000.0f);
	} else {
		cuda_safe_call(cudaEventRecord(stop_event, 0));
		cuda_safe_call(cudaEventSynchronize(stop_event));
		cuda_safe_call(cudaEventElapsedTime(&time_ms, start_event, stop_event));
	}
	// update measurements
	int sample = counters[current_config][current_renderer];
	if (sample < MAX_SAMPLE_COUNT) {
		data[current_config][current_renderer][sample] = time_ms;
		counters[current_config][current_renderer]++;
	}
	// reset profiler control data
	//current_renderer = current_method = -1;
	last_times[last_times_counter] = time_ms;
	last_times_counter = ++last_times_counter % LAST_SAMPLE_COUNT;
	return time_ms;
}

float Profiler::average(int config, int renderer) {
	int counter = counters[config][renderer];
	if (counter == 0) 
		return -1;
	float sum = 0;
	for (int sample = 0; sample < counter; sample++)
		sum += data[config][renderer][sample];
	return sum / counter;
}

float Profiler::maximum(int config, int renderer) {
	int counter = counters[config][renderer];
	float max_value = -1;
	for (int sample = 0; sample < counter; sample++)
		if (data[config][renderer][sample] > max_value)
			max_value = data[config][renderer][sample];
	return max_value;
}

void Profiler::print_config(int config) {
	Logger::log("Renderer ID, Samples, Max (ms), Avg (ms)");
	Logger::log("\n");
	for (int r = 0; r < RENDERER_COUNT; r++) {
		Logger::log("%11i, ", r);
		Logger::log("%7i, ", counters[config][r]);
		Logger::log("%8.2f, ", maximum(config, r));
		Logger::log("%8.2f", average(config, r));
		Logger::log("\n");
	}
	Logger::log("\n");
}
