/******************************************************************************************/
// VolR - Volume rendering engine using CUDA, by Miroslav Beno, STU FIIT 2012
/******************************************************************************************/

/****************************************/
// Main module
/****************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GL/glew.h"
#include "cuda_gl_interop.h"

#include "ModelBase.h"
#include "ViewBase.h"
#include "RaycasterBase.h"
#include "Renderer.h"
#include "UI.h"
#include "Profiler.h"
#include "Logger.h"
#include "common.h"
#include "cuda_utils.h"

#define MAX_BENCH_SAMPLE 7500		// benchmark timeout

static char file_name[100] = "VisMale.pvm";	
static char log_file[100] = "VolR.log";
bool NO_SAFE = false;

static int benchmark_mode = 0;
static int config = 0;
static char config_names[50][80] = {"Interactive", 
		"Bucky", "Daisy", "VisMale", "Engine", "Foot", "Pig", "Porsche",
		"Foot: No optims", "F: ERT on", "F: ERT+ESL on", 
		"Scale 0.9", "Scale 0.8", "Scale 0.7", "Scale 0.6", "Scale 0.5", "Scale 0.4", "Scale 0.3",
		"Ray step *1.1", "Ray step *1.2", "Ray step *1.3", "Ray step *1.4", "Ray step *1.5", "Ray step *1.6", "Ray step *1.7" };

static GLuint pbo_gl_id = NULL;
static GLuint tex_gl_id = NULL;

static int gpu_id = 0;
static cudaGraphicsResource *pbo_cuda_id; 

static int renderer_id = 4;
static Renderer *renderers[RENDERER_COUNT];

void delete_PBO_texture() {
    if (pbo_gl_id != NULL) {
		cuda_safe_call(cudaGraphicsUnregisterResource(pbo_cuda_id));
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffersARB(1, &pbo_gl_id);
    }
	if (tex_gl_id != NULL) {
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &tex_gl_id);
	}
}

void reset_PBO_texture() {							//!: main glut window must be set		
	delete_PBO_texture();
	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 
		ViewBase::view.dims.x * ViewBase::view.dims.y * 4, NULL, GL_STREAM_DRAW_ARB);		
	cuda_safe_call(cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &tex_gl_id);
	glBindTexture(GL_TEXTURE_2D, tex_gl_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
		ViewBase::view.dims.x, ViewBase::view.dims.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

uchar4 *prepare_PBO() {							
	if (renderer_id < 3) {
		return (uchar4 *) glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	}
	else {
		uchar4 *dev_buffer;
		size_t dev_buffer_size;
		cuda_safe_call(cudaGraphicsMapResources(1, &pbo_cuda_id, 0));
		cuda_safe_call(cudaGraphicsResourceGetMappedPointer((void **)&dev_buffer, &dev_buffer_size, pbo_cuda_id));
		return dev_buffer;
	}
}

void finalize_PBO() {
	if (renderer_id < 3) {
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	}
	else {
		cuda_safe_call(cudaGraphicsUnmapResources(1, &pbo_cuda_id, 0));
	}		
}

void draw_volume() {
	if (UI::viewport_resized_flag) {
		if (ViewBase::view.dims.x < 16 || ViewBase::view.dims.y < 16)
			return;
		reset_PBO_texture();
		for (int i=0; i < RENDERER_COUNT; i++)
			renderers[i]->set_window_buffer(ViewBase::view);
		UI::viewport_resized_flag = false;
	}
	RaycasterBase::set_view(ViewBase::view);
	uchar4 *pbo_array = prepare_PBO();
	Profiler::start(renderer_id, renderer_id == 0 ? 0 : 1);
	renderers[renderer_id]->render_volume(pbo_array, RaycasterBase::raycaster);
	Profiler::stop();
	finalize_PBO();
}

void glew_init() {		// needs preceding initialization of OpenGL context, handled in glut
	Logger::log("Initializing GLEW version %s...\n", glewGetString(GLEW_VERSION));
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		Logger::log("Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	if (!GLEW_VERSION_2_0) {
		Logger::log("Error: OpenGL 2.0 is not supported\n");
		exit(EXIT_FAILURE);
	}
}

void cuda_init() {
	Logger::log("Initializing CUDA...\n");
    int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
		if (device_count == 0) {
			Logger::log("Error: No device supporting CUDA found\n");			// todo: undefined device_count if wrong driver detected
		}
		cuda_safe_check();
		Logger::log("Error: Update your display drivers - need at least CUDA driver version 3.2\n");
	}

	Logger::log("Number of CUDA devices found: %d\n", device_count);
  
	cudaDeviceProp device_prop;
	int max_compute_perf = 0, max_perf_device = 0, max_perf_device_cpm = 0, best_arch = 0;
	for (int i = 0; i < device_count; i++) {
		cuda_safe_call(cudaGetDeviceProperties(&device_prop, i));
		if ((!device_prop.tccDriver) &&
			(device_prop.major > 0 && device_prop.major < 9999)) 
				best_arch = MAXIMUM(best_arch, device_prop.major);
	}
	for (int i = 0; i < device_count; i++) {
		cuda_safe_call(cudaGetDeviceProperties(&device_prop, i));
		if ((device_prop.major == 9999 && device_prop.minor == 9999) 
			|| (device_prop.tccDriver)
			|| (device_prop.major != best_arch))
			continue;
		int cores_per_mp = 0;
		switch (device_prop.major) {
			case 1: cores_per_mp = 8; break;
			case 2: switch (device_prop.minor) {
						case 1: cores_per_mp = 32; break;
						case 2: cores_per_mp = 48; break;
					}
					break;
			case 3: cores_per_mp = 192; break;
		}
		int compute_perf = device_prop.multiProcessorCount * cores_per_mp * device_prop.clockRate;
		if (compute_perf > max_compute_perf) {
			max_compute_perf = compute_perf;
			max_perf_device_cpm = cores_per_mp;
			max_perf_device  = i;
		}
	}
	gpu_id = max_perf_device;

    cuda_safe_call(cudaGetDeviceProperties(&device_prop, gpu_id));
    Logger::log("\nUsing device %d: \"%s\"\n", gpu_id, device_prop.name);
	int version = 0;
	cuda_safe_call(cudaDriverGetVersion(&version));
	Logger::log("  CUDA Driver Version:      %d.%d\n", version/1000, version%100);
	cuda_safe_call(cudaRuntimeGetVersion(&version));
	Logger::log("  CUDA Runtime Version:     %d.%d\n", version/1000, version%100);
    Logger::log("  CUDA Capability:          %d.%d\n", device_prop.major, device_prop.minor);
	if (max_perf_device_cpm != 0)
		Logger::log("  Multiprocessors, Cores:   %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
			device_prop.multiProcessorCount,
			max_perf_device_cpm,
			max_perf_device_cpm * device_prop.multiProcessorCount);
	Logger::log("  Global memory:            %lu MB\n", (unsigned long) device_prop.totalGlobalMem / (1024*1024));
	Logger::log("  Constant memory:          %u bytes\n", device_prop.totalConstMem); 
	Logger::log("  Shared memory per block:  %u bytes\n", device_prop.sharedMemPerBlock);
	Logger::log("  Registers per block:	    %d\n", device_prop.regsPerBlock);
	Logger::log("  Clock rate:               %.2f GHz\n", device_prop.clockRate * 1e-6f);
	Logger::log("  Integrated:               %s\n", device_prop.integrated ? "Yes" : "No");
	Logger::log("\n");

	UI::set_gpu_name(device_prop.name);

	cuda_safe_call(cudaGLSetGLDevice(gpu_id));
}

void print_profiler() {
	Logger::log("\nSummary profiler report:\n");
	for(int r = 0; r < RENDERER_COUNT; r++)
		Logger::log(" Rend.%2i: %s\n", r, renderers[r]->get_name());
	Logger::log("%15s,%8s,", "Configuration", "Value");
	for(int r = 0; r < RENDERER_COUNT; r++) {
		Logger::log(" Rend.%2i", r);
		if (r != RENDERER_COUNT - 1)
			Logger::log(",");
	}
	Logger::log("\n");
	for (int i=0; i<=config; i++) {
		if (i == 0) {
			Logger::log("%15s,", config_names[i]);
			Profiler::print_samples(i);
		}
		Logger::log("%15s,", config_names[i]);
		Profiler::print_avg(i);
		if (i == 0) {
			Logger::log("%15s,", config_names[i]);
			Profiler::print_max(i);
		}
	}
}

void benchmark_config_loop() {
	Profiler::reset_config(config);	
	for(renderer_id = 0; renderer_id < RENDERER_COUNT; renderer_id++) {
		if (renderer_id == 0 && config > 6 && config < 14)
			continue;
		if (renderer_id == 0 && benchmark_mode == 2)
			continue;
		ViewBase::view.perspective = false;
		for (int p = 0; p<2; p++) {
			ViewBase::toggle_perspective(true);
			ViewBase::set_camera_position(make_float3(0, 0, 0), 2);
			draw_volume();
			if (Profiler::time_ms > MAX_BENCH_SAMPLE) break;
			ViewBase::set_camera_position(make_float3(-45, -45, 0), 2);
			draw_volume();
			if (Profiler::time_ms > MAX_BENCH_SAMPLE) break;
			ViewBase::set_camera_position(make_float3(90, 0, 0), 2);
			draw_volume();
			if (Profiler::time_ms > MAX_BENCH_SAMPLE) break;
			ViewBase::set_camera_position(make_float3(180, 90, 0), 2);
			draw_volume();
			if (Profiler::time_ms > MAX_BENCH_SAMPLE) break;
			ViewBase::view.perspective = true;
		}
	}
	Profiler::print_avg(config);
	config++;
	Logger::log("\n");
}

int benchmark_load_file(const char* file_name) {
	char name[256] = "";
	sprintf(name, "%s.pvm", file_name);
	int file_error = ModelBase::load_model(name);
	if (!file_error) {
		RaycasterBase::set_volume(ModelBase::volume);
		for (int i=0; i < RENDERER_COUNT; i++) {
			if (UI::renderers[i]->set_volume(RaycasterBase::raycaster.volume) != 0)
				return 1;
			UI::renderers[i]->set_transfer_fn(RaycasterBase::raycaster);
		}
	}
	return file_error;
}

void benchmark() {
	Logger::log("Entering benchmark loop...\n\n");
	config = 1;

	while (config <= 7) {
		Logger::log("%s benchmark\n", config_names[config]);
		if (benchmark_load_file(config_names[config]) != 0) {
			config++;
			continue;
			}
		benchmark_config_loop();
	}

	if (benchmark_load_file("Foot") == 0) {
		Logger::log("%s benchmark\n", config_names[config]);
		RaycasterBase::toggle_esl();
		RaycasterBase::change_ray_threshold(1.0f, true);
		benchmark_config_loop();
		Logger::log("%s benchmark\n", config_names[config]);
		RaycasterBase::change_ray_threshold(0.95f, true);
		benchmark_config_loop();
		Logger::log("%s benchmark\n", config_names[config]);
		RaycasterBase::toggle_esl();
		benchmark_config_loop();
	}
	else
		return;

	float viewport_scale = 1.0f;
	ushort2 original_size = {ViewBase::view.dims.x, ViewBase::view.dims.y};
	while (viewport_scale > 0.3f) {
		Logger::log("%s benchmark\n", config_names[config]);
		viewport_scale -= 0.1f;
		ViewBase::set_viewport_dims(original_size, viewport_scale);
		UI::viewport_resized_flag = true;
		Logger::log("Resolution: %dx%d\n", ViewBase::view.dims.x, ViewBase::view.dims.y);
		benchmark_config_loop();
	}
	ViewBase::set_viewport_dims(original_size);
	UI::viewport_resized_flag = true;

	float original_raystep = RaycasterBase::raycaster.ray_step;
	float raystep_factor = 1.0;
	while (raystep_factor <= 1.7f) {
		Logger::log("%s benchmark\n", config_names[config]);
		raystep_factor += 0.1f;
		RaycasterBase::change_ray_step(original_raystep * raystep_factor, true);
		benchmark_config_loop();
	}
	RaycasterBase::reset_ray_step();
	config--;
}

void cleanup_and_exit() {
	print_profiler();
	Logger::log("\nCleaning...\n");
	delete_PBO_texture();
	Profiler::destroy();
	for (int i = RENDERER_COUNT - 1; i >=0 ; i--)
		delete renderers[i];
	free(ModelBase::volume.data);
	free(RaycasterBase::raycaster.transfer_fn);
	free(RaycasterBase::raycaster.esl_volume);
	UI::destroy();
	Logger::log("Bye!\n\n");
	Logger::close();
	exit(EXIT_SUCCESS);
}

void print_usage() {
	printf("VolR - Volume rendering engine using CUDA, by Miroslav Beno, STU FIIT 2012\n\n");
	printf("Usage:\n");
	printf("VolR.exe [-h] [-f <filename>] [-r <id>] [-s <width> <height>] [-b] [-bg] [-nosafe]\n");
	printf("  -h : Show this help\n");
	printf("  -f : Load specified file with volume data - either .pvm or .raw format\n");
	printf("  -r : Set specified renderer: 0 - CPU, 1-4 GPU renderers\n");
	printf("  -s : Set specified viewport size - must be in range <128; 2048>\n");
	printf("  -b : Run in benchmark mode - can take a few minutes\n");
	printf("  -bg : Run in benchmark mode with CPU renderer disabled - shorter\n");
	printf("  -nosafe : CUDA errors will not cause fatal exit - experimental\n");
}

int main(int argc, char **argv) {

	if (argc > 1)
		if (strncmp(argv[1], "-h", 2) == 0) {
			print_usage();
			return(EXIT_SUCCESS);
		}

	Logger::init(log_file, 'a');

	for (int i = 1; i < argc; i++) {
		char *arg = argv[i];
		if (strncmp(arg, "-f", 2) == 0) {
			if (i + 1 >= argc) {
				Logger::log("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			i++;
			Logger::log("Setting volume data file '%s'...\n", argv[i]);
			strcpy(file_name, argv[i]);
		} else if (strncmp(arg, "-r", 2) == 0) {
			if (i + 1 >= argc) {
				Logger::log("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			int r_id = atoi(argv[++i]);
			if (r_id < 0 || r_id >= RENDERER_COUNT) {
				Logger::log("%s error: Wrong parameters. Use -h to help.\n", arg);
				continue;
			}
			Logger::log("Setting initial renderer id %d...\n", r_id);
			renderer_id = r_id;
		} else if (strncmp(arg, "-s", 2) == 0) {
			if (i + 2 >= argc) {
				Logger::log("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			ushort2 viewport_dims;
			viewport_dims.x = atoi(argv[++i]);
			viewport_dims.y = atoi(argv[++i]);
			if (viewport_dims.x < 128 || viewport_dims.y < 128 || viewport_dims.x > 2048 || viewport_dims.y > 2048) {
				Logger::log("%s error: Wrong parameters. Use -h to help.\n", arg);
				continue;
			}
			Logger::log("Setting viewport size %dx%d...\n", viewport_dims.x, viewport_dims.y);
			ViewBase::set_viewport_dims(viewport_dims);
		} else if (strncmp(arg, "-bg", 3) == 0) {
			Logger::log("Setting benchmark mode without CPU renderer...\n");
			benchmark_mode = 2;
		} else if (strncmp(arg, "-b", 2) == 0) {
			Logger::log("Setting benchmark mode...\n");
			benchmark_mode = true;
		} else if (strncmp(arg, "-nosafe", 7) == 0) {
			Logger::log("Supressing CUDA fatal errors...\n");
			NO_SAFE = true;
		} else {
			Logger::log("Warning: unknown argument: %s\n", arg);
		}
	}

	if (ModelBase::load_model(file_name) != 0) 
		Logger::log("Warning: Default volume data file not loaded. Use load file on control panel...\n");

	RaycasterBase::set_view(ViewBase::view);
	RaycasterBase::reset_transfer_fn();
	RaycasterBase::set_volume(ModelBase::volume);

	UI::init(renderers, &renderer_id, draw_volume, cleanup_and_exit);
	glew_init();		
	cuda_init();
	Profiler::init();

	Logger::log("Initializing renderers 0 - %d...\n", RENDERER_COUNT - 1);
	renderers[0] = new CPURenderer(RaycasterBase::raycaster);	
	renderers[1] = new GPURenderer1(RaycasterBase::raycaster);
	renderers[2] = new GPURenderer2(RaycasterBase::raycaster);
	renderers[3] = new GPURenderer3(RaycasterBase::raycaster);
	renderers[4] = new GPURenderer4(RaycasterBase::raycaster);

	if (benchmark_mode) {
		benchmark();
		cleanup_and_exit();
	}

	ViewBase::set_camera_position(make_float3(120, 0, 200));

	Logger::log("Entering main event loop...\n");
	UI::print_usage();
	UI::start();
	return EXIT_FAILURE;
}
