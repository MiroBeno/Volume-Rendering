#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glew.h"
#include "cuda_gl_interop.h"

#include "ModelBase.h"
#include "ViewBase.h"
#include "RaycasterBase.h"
#include "Renderer.h"
#include "UI.h"
#include "Profiler.h"
#include "common.h"
#include "cuda_utils.h"

const char *INIT_FILE_NAME = "VisMale.pvm";	
bool NO_SAFE = false;

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

void reset_PBO_texture() {							// ! musi byt setnute main glut window, inak padne		// bug pri nastavenom downsampling a rozmere 0 pri cuda r
	//printf("Setting pixel buffer object...\n");
	delete_PBO_texture();
	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 
		ViewBase::view.dims.x * ViewBase::view.dims.y * 4, NULL, GL_STREAM_DRAW_ARB);		//GL_STREAM_DRAW_ARB|GL_DYNAMIC_DRAW_ARB ??  // int CHANNEL_COUNT = 4;
	cuda_safe_call(cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &tex_gl_id);
	glBindTexture(GL_TEXTURE_2D, tex_gl_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
		ViewBase::view.dims.x, ViewBase::view.dims.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

uchar4 *prepare_PBO() {							//GLubyte *
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
	//printf("Drawing volume...\n");
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
	Profiler::start(renderer_id, 0, renderer_id == 0 ? 0 : 1);
	renderers[renderer_id]->render_volume(pbo_array, RaycasterBase::raycaster);
	Profiler::stop();
	finalize_PBO();
}

void glew_init() {		// needs preceding initialization of OpenGL context, handled in glut
	printf("Initializing GLEW version %s...\n", glewGetString(GLEW_VERSION));
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	if (!GLEW_VERSION_2_0) {
		fprintf(stderr, "Error: OpenGL 2.0 is not supported\n");
		exit(EXIT_FAILURE);
	}
}

void cuda_init() {
	printf("Initializing CUDA...\n");
    int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
		fprintf(stderr, "Error: Update your display drivers - need at least CUDA driver version 3.2\n");
	}
	cuda_safe_check();
	if (device_count == 0) {
		fprintf(stderr, "Error: No device supporting CUDA found\n");
		exit(EXIT_FAILURE);
	}
	printf("Number of CUDA devices found: %d\n", device_count);
  
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
    printf("\nUsing device %d: \"%s\"\n", gpu_id, device_prop.name);
	int version = 0;
	cuda_safe_call(cudaDriverGetVersion(&version));
	printf("  CUDA Driver Version:                      %d.%d\n", version/1000, version%100);
	cuda_safe_call(cudaRuntimeGetVersion(&version));
	printf("  CUDA Runtime Version:                     %d.%d\n", version/1000, version%100);
    printf("  CUDA Capability version number:           %d.%d\n", device_prop.major, device_prop.minor);
	if (max_perf_device_cpm != 0)
		printf("  Multiprocessors x Cores/MP = Cores:       %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
			device_prop.multiProcessorCount,
			max_perf_device_cpm,
			max_perf_device_cpm * device_prop.multiProcessorCount);
	printf("  Total amount of global memory:            %lu MB\n", (unsigned long) device_prop.totalGlobalMem / (1024*1024));
	printf("  Total amount of constant memory:          %u bytes\n", device_prop.totalConstMem); 
	printf("  Total amount of shared memory per block:  %u bytes\n", device_prop.sharedMemPerBlock);
	printf("  Total number of registers per block:	    %d\n", device_prop.regsPerBlock);
	printf("  Clock rate:                               %.2f GHz\n", device_prop.clockRate * 1e-6f);
	printf("  Integrated:                               %s\n", device_prop.integrated ? "Yes" : "No");
	printf("\n");

	UI::set_gpu_name(device_prop.name);

	cuda_safe_call(cudaGLSetGLDevice(gpu_id));
}

void benchmark() {
	printf("Entering benchmark loop...\n");
	int sample_count[RENDERER_COUNT] = {0, 10, 10, 10, 10};
	for(renderer_id = 0; renderer_id < RENDERER_COUNT; renderer_id++) {
		// for configuration
		for (int k = 0; k < sample_count[renderer_id]; k++) {
			draw_volume();
		}
	}
	Profiler::dump("profiler.txt");
}

void cleanup_and_exit() {
	printf("Cleaning...\n");
	delete_PBO_texture();
	Profiler::destroy();
	for (int i = RENDERER_COUNT - 1; i >=0 ; i--)
		delete renderers[i];
	free(ModelBase::volume.data);
	free(RaycasterBase::raycaster.transfer_fn);
	free(RaycasterBase::raycaster.esl_min_max);
	free(RaycasterBase::raycaster.esl_volume);
	UI::destroy();
	printf("Bye!\n");
	exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {

	bool benchmark_mode = false;
	char file_name[256];
	strcpy(file_name, INIT_FILE_NAME);
	for (int i = 1; i < argc; i++) {
		char *arg = argv[i];
		if (strncmp(arg, "-h", 2) == 0) {
			printf("VolR - Volume rendering engine using CUDA, by Miroslav Beno, STU FIIT 2012\n\n");
			printf("Usage:\n");
			printf("VolR.exe [-h] [-f <filename>] [-r <id>] [-s <width> <height>] [-b] [-nosafe]\n");
			printf("  -h : Show this help\n");
			printf("  -f : Load specified file with volume data - either .pvm or .raw format\n");
			printf("  -r : Set specified renderer: 0 - CPU, 1-4 GPU renderers\n");
			printf("  -s : Set specified viewport size - must be in range <128; 2048>\n");
			printf("  -b : Run in benchmark mode - can take a few minutes\n");
			printf("  -nosafe : CUDA errors will not cause fatal exit - experimental\n");
			return 0;
		} else if (strncmp(arg, "-f", 2) == 0) {
			if (i + 1 >= argc) {
				printf("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			i++;
			printf("Setting volume data file '%s'...\n", argv[i]);
			strcpy(file_name, argv[i]);
		} else if (strncmp(arg, "-r", 2) == 0) {
			if (i + 1 >= argc) {
				printf("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			int r_id = atoi(argv[++i]);
			if (r_id < 0 || r_id >= RENDERER_COUNT) {
				printf("%s error: Wrong parameters. Use -h to help.\n", arg);
				continue;
			}
			printf("Setting initial renderer id %d...\n", r_id);
			renderer_id = r_id;
		} else if (strncmp(arg, "-s", 2) == 0) {
			if (i + 2 >= argc) {
				printf("%s error: Not enough parameters. Use -h to help.\n", arg);
				continue;
			}
			ushort2 viewport_dims;
			viewport_dims.x = atoi(argv[++i]);
			viewport_dims.y = atoi(argv[++i]);
			if (viewport_dims.x < 128 || viewport_dims.y < 128 || viewport_dims.x > 2048 || viewport_dims.y > 2048) {
				printf("%s error: Wrong parameters. Use -h to help.\n", arg);
				continue;
			}
			printf("Setting viewport size %dx%d...\n", viewport_dims.x, viewport_dims.y);
			ViewBase::set_viewport_dims(viewport_dims.x, viewport_dims.y);
		} else if (strncmp(arg, "-b", 2) == 0) {
			printf("Setting benchmark mode...\n");
			benchmark_mode = true;
		} else if (strncmp(arg, "-nosafe", 7) == 0) {
			printf("Supressing CUDA errors...\n");
			NO_SAFE = true;
		} else {
			printf("Warning: unknown argument: %s\n", arg);
		}
	}

	if (ModelBase::load_model(file_name) != 0) 
		exit(EXIT_FAILURE);

	RaycasterBase::set_view(ViewBase::view);
	RaycasterBase::reset_transfer_fn();
	RaycasterBase::set_volume(ModelBase::volume);

	UI::init(renderers, &renderer_id, draw_volume, cleanup_and_exit);
	glew_init();		
	cuda_init();
	Profiler::init();

	printf("Initializing renderers 0 - %d...\n", RENDERER_COUNT - 1);
	renderers[0] = new CPURenderer(RaycasterBase::raycaster);	
	renderers[1] = new GPURenderer1(RaycasterBase::raycaster);
	renderers[2] = new GPURenderer2(RaycasterBase::raycaster);
	renderers[3] = new GPURenderer3(RaycasterBase::raycaster);
	renderers[4] = new GPURenderer4(RaycasterBase::raycaster);
	
	if (benchmark_mode) {
		benchmark();
		cleanup_and_exit();
	}

	//printf("Raycaster data size: %i B\n", sizeof(Raycaster));
	UI::print_usage();
	UI::start();
	return EXIT_FAILURE;
}
