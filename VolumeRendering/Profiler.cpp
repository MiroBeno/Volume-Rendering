#include <fstream>

#include "Profiler.h"

using namespace std;

float Profiler::data[RENDERER_COUNT][CONFIGURATION_COUNT][MAX_SAMPLE_COUNT];
int Profiler::counters[RENDERER_COUNT][CONFIGURATION_COUNT];
int Profiler::current_renderer, Profiler::current_configuration, Profiler::current_method;
clock_t Profiler::last_clock;
cudaEvent_t Profiler::start_event, Profiler::stop_event;
float Profiler::time_ms;

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
	for (int r = 0; r < RENDERER_COUNT; r++)
		for (int c = 0; c < CONFIGURATION_COUNT; c++)
			counters[r][c] = 0;
	current_renderer = current_configuration = current_method = -1;
	time_ms = -1;
}

void Profiler::start(int renderer_id, int configuration, int method) {
	current_renderer = renderer_id;
	current_configuration = configuration;
	current_method = method;
	if (current_method == 0) {
		last_clock = clock();
	} else {
		cuda_safe_call(cudaEventRecord(start_event, 0));
	}
}

float Profiler::stop() {
	if (current_renderer == -1 || current_configuration == -1 || current_method == -1)
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
	int sample = counters[current_renderer][current_configuration];
	if (sample < MAX_SAMPLE_COUNT) {
		data[current_renderer][current_configuration][sample] = time_ms;
		counters[current_renderer][current_configuration]++;
	}
	// reset profiler control data
	//current_renderer = current_configuration = current_method = -1;
	return time_ms;
}

float Profiler::average(int renderer_id, int configuration) {
	int counter = counters[renderer_id][configuration];
	if (counter == 0) 
		return -1;
	float sum = 0;
	for (int sample = 0; sample < counter; sample++)
		sum += data[renderer_id][configuration][sample];
	return sum / counter;
}

void Profiler::dump(const char *filePath) {
	ofstream file(filePath);
	for (int r = 0; r < RENDERER_COUNT; r++) {
		for (int c = 0; c < CONFIGURATION_COUNT; c++) {
			file << average(r, c);
			if (c < CONFIGURATION_COUNT - 1)
				file << ", ";
		}
		file << endl;
	}
}
