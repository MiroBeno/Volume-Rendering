#include <time.h>

#include "cuda_utils.h"
#include "Profiler.h"

Stat Profiler::statistics[MAX_CONFIG_COUNT][RENDERER_COUNT];
int Profiler::current_config, Profiler::current_renderer, Profiler::current_method;
float Profiler::time_ms;
float Profiler::last_times[LAST_SAMPLE_COUNT];
int Profiler::last_times_counter;

static clock_t last_clock;
static cudaEvent_t start_event, stop_event;

void Profiler::init() {
	for (int c = 0; c < MAX_CONFIG_COUNT; c++) {
		for (int r = 0; r < RENDERER_COUNT; r++)
			statistics[c][r] = Stat();
		reset_config(c);
	}
	current_config = 0;
	for (int i = 0; i < LAST_SAMPLE_COUNT; i++)
		last_times[i] = 0;
	last_times_counter = 0;
	cuda_safe_call(cudaEventCreate(&start_event));
	cuda_safe_call(cudaEventCreate(&stop_event));
}

void Profiler::destroy() {
	cuda_safe_call(cudaEventDestroy(start_event));
	cuda_safe_call(cudaEventDestroy(stop_event));
}

void Profiler::reset_config(int config) {
	for (int r = 0; r < RENDERER_COUNT; r++)
		statistics[config][r].time_sum = statistics[config][r].time_max = statistics[config][r].samples = 0;
	current_renderer = current_method = -1;
	time_ms = -1;
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

	time_ms = -1;
	if (current_method == 0) {
		time_ms = (clock() - last_clock) / (CLOCKS_PER_SEC / 1000.0f);
	} else {
		cuda_safe_call(cudaEventRecord(stop_event, 0));
		cuda_safe_call(cudaEventSynchronize(stop_event));
		cuda_safe_call(cudaEventElapsedTime(&time_ms, start_event, stop_event));
	}

	statistics[current_config][current_renderer].samples++;
	statistics[current_config][current_renderer].time_sum += time_ms;
	if (time_ms > statistics[current_config][current_renderer].time_max)
		statistics[current_config][current_renderer].time_max = time_ms;
	last_times[last_times_counter] = time_ms;
	last_times_counter = ++last_times_counter % LAST_SAMPLE_COUNT;

	current_renderer = current_method = -1;
	return time_ms;
}

void Profiler::print_samples(int config) {
	Logger::log("%9s", "Samples,");
	for (int r = 0; r < RENDERER_COUNT; r++) {
		Logger::log("%8i", statistics[config][r].samples);
		if (r != RENDERER_COUNT - 1)
			Logger::log(",");
	}
	Logger::log("\n");
}

void Profiler::print_avg(int config) {
	Logger::log("%9s", "Avg(ms),");
	for (int r = 0; r < RENDERER_COUNT; r++) {
		if (statistics[config][r].samples >= MIN_SAMPLE_STAT)
			Logger::log("%8.2f", statistics[config][r].time_sum / statistics[config][r].samples);
		else
			Logger::log("%8s", "N/A");
		if (r != RENDERER_COUNT - 1)
			Logger::log(",");
	}
	Logger::log("\n");
}

void Profiler::print_max(int config) {
	Logger::log("%9s", "Max(ms),");
	for (int r = 0; r < RENDERER_COUNT; r++) {
		if (statistics[config][r].samples >= MIN_SAMPLE_STAT)
			Logger::log("%8.2f", statistics[config][r].time_max);
		else
			Logger::log("%8s", "N/A");
		if (r != RENDERER_COUNT - 1)
			Logger::log(",");
	}
	Logger::log("\n");
}
