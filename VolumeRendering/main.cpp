#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**/#include <ctime>

#include "Model.h"
#include "View.h"
#include "Raycaster.h"
#include "Renderer.h"
#include "UI.h"

#include "cuda_utils.h"
#include "cuda_gl_interop.h"

extern const int RENDERERS_COUNT = 5;
//const char *FILE_NAME = "Bucky.pvm";						// 32x32x32 x 8bit
//const char *FILE_NAME = "Foot.pvm";						// 256x256x256 x 8bit
const char *FILE_NAME = "VisMale.pvm";					// 128x256x256 x 8bit
//const char *FILE_NAME = "Bonsai1-LO.pvm";					// 512x512x182 x 16 bit

static GLuint pbo_gl_id = NULL;
static GLuint tex_gl_id = NULL;

static int gpu_id;
static cudaGraphicsResource *pbo_cuda_id;

static cudaEvent_t start, stop, frame; 
/**/static clock_t start_time;
float2 elapsed_time = {0, 0};

int renderer_id = 1;
Renderer *renderers[RENDERERS_COUNT];

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

void reset_PBO_texture() {							// ! musi byt setnute main glut window, inak padne
	printf("Setting pixel buffer object...\n");
	delete_PBO_texture();
	glGenBuffersARB(1, &pbo_gl_id);	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_gl_id);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, UI::window_size.x * UI::window_size.y * 4, NULL, GL_STREAM_DRAW_ARB);		//GL_STREAM_DRAW_ARB|GL_DYNAMIC_DRAW_ARB ??  // int CHANNEL_COUNT = 4;
	cuda_safe_call(cudaGraphicsGLRegisterBuffer(&pbo_cuda_id, pbo_gl_id, cudaGraphicsMapFlagsWriteDiscard));
	glGenTextures(1, &tex_gl_id);
	glBindTexture(GL_TEXTURE_2D, tex_gl_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, UI::window_size.x, UI::window_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
	if (UI::window_resize_flag) {
		if (UI::window_size.x == 0 || UI::window_size.y == 0)
			return;
		reset_PBO_texture();
		ViewBase::set_window_size(UI::window_size);
		for (int i=0; i < RENDERERS_COUNT; i++)
			renderers[i]->set_window_buffer(ViewBase::view);
		UI::window_resize_flag = false;
	}
	RaycasterBase::raycaster.view = ViewBase::view;
	uchar4 *pbo_array = prepare_PBO();
	cuda_safe_call(cudaEventRecord(start, 0));
	if (renderer_id == 0) 
		start_time = clock();
	renderers[renderer_id]->render_volume(pbo_array, RaycasterBase::raycaster);
	cuda_safe_call(cudaEventRecord(stop, 0));
	cuda_safe_call(cudaEventSynchronize(stop));
	cuda_safe_call(cudaEventElapsedTime(&elapsed_time.x, start, stop));
	cuda_safe_call(cudaEventElapsedTime(&elapsed_time.y, frame, stop));
	if (renderer_id == 0) {
		elapsed_time.x = (clock() - start_time) / (CLOCKS_PER_SEC / 1000.0f);
		elapsed_time.y = 0;
	}
	finalize_PBO();
	cuda_safe_call(cudaEventRecord(frame, 0));
}

void cleanup_and_exit() {
	printf("Cleaning...\n");
	cuda_safe_call(cudaEventDestroy(start));
	cuda_safe_call(cudaEventDestroy(stop));
	cuda_safe_call(cudaEventDestroy(frame));
	delete_PBO_texture();
	for (int i = RENDERERS_COUNT - 1; i >=0 ; i--)
		delete renderers[i];
	free(ModelBase::volume.data);
	free(RaycasterBase::raycaster.transfer_fn);
	free(RaycasterBase::raycaster.esl_min_max);
	free(RaycasterBase::raycaster.esl_volume);
	UI::cleanup();
	printf("Bye!\n");
	exit(0);
}

void init_cuda() {
	printf("Initializing CUDA...\n");
    int device_count = 0;
	if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
		printf("Error: Update your display drivers - need at least CUDA driver version 3.2\n");
	}
	cuda_safe_check();
	if (device_count == 0) {
		printf("Error: No device supporting CUDA found\n");
		exit(EXIT_FAILURE);
	}
	printf("Number of CUDA devices found: %d\n", device_count);
  
	cudaDeviceProp device_prop;
	int max_compute_perf = 0, max_perf_device = -1, max_perf_device_cpm = 0, best_arch = 0;
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
	printf("  Total amount of global memory:            %llu bytes\n", (unsigned long long) device_prop.totalGlobalMem);
	printf("  Total amount of constant memory:          %u bytes\n", device_prop.totalConstMem); 
	printf("  Total amount of shared memory per block:  %u bytes\n", device_prop.sharedMemPerBlock);
	printf("  Total number of registers per block:	    %d\n", device_prop.regsPerBlock);
	printf("  Clock rate:                               %.2f GHz\n", device_prop.clockRate * 1e-6f);
	printf("  Integrated:                               %s\n", device_prop.integrated ? "Yes" : "No");
	printf("\n");

	cuda_safe_call(cudaGLSetGLDevice(gpu_id));
	cuda_safe_call(cudaEventCreate(&start));
	cuda_safe_call(cudaEventCreate(&stop));
	cuda_safe_call(cudaEventCreate(&frame));
	cuda_safe_call(cudaEventRecord(frame, 0));
}

int main(int argc, char **argv) {

	for (int i = 1; i < argc; i++) {
		char *arg = argv[i];
		if (strncmp(arg, "-h", 2) == 0) {
			printf("VolumeRendering.exe [-h]\n");
			return 0;
		} else if (strncmp(arg, "-size", 5) == 0) {
			if (i + 2 >= argc) {
				printf("Error: Not enough arguments.\n");
				return EXIT_FAILURE;
			}
			int width = atoi(argv[++i]);
			int height = atoi(argv[++i]);
			if (width <= 0 || height <= 0) {
				printf("Error: Non-positive size.\n");
				return EXIT_FAILURE;
			}
			UI::window_size.x = (unsigned short) width;
			UI::window_size.y = (unsigned short) height;
			ViewBase::set_window_size(UI::window_size);
		} else {
			printf("Warning: unknown argument: %s\n", arg);
		}
	}

	if (ModelBase::load_model(FILE_NAME) != 0) {
		exit(EXIT_FAILURE);
	}

	for (int i =0; i < TF_SIZE; i++) {
		RaycasterBase::base_transfer_fn[i] = make_float4(i <= TF_SIZE/3 ? (i*3)/(float)(TF_SIZE) : 0.0f, 
										(i > TF_SIZE/3) && (i <= TF_SIZE/3*2) ? ((i-TF_SIZE/3)*3)/(float)(TF_SIZE) : 0.0f, 
										i > TF_SIZE/3*2 ? ((i-TF_SIZE/3*2)*3)/(float)(TF_SIZE) : 0.0f, 
										i > (20/TF_RATIO) ? i/(float)(TF_SIZE) : 0.0f);
	}
	//RaycasterBase::base_transfer_fn[30] = make_float4(1,1,1,1);
	/*for (int i =0; i < TF_SIZE; i++) {
		RaycasterBase::base_transfer_fn[i] = make_float4(0.23f, 0.23f, 0.0f, i/(float)TF_SIZE);
	}*/
	RaycasterBase::raycaster.view = ViewBase::view;
	RaycasterBase::set_volume(ModelBase::volume);

	UI::init_gl(argc, argv);
	init_cuda();
	reset_PBO_texture();

	printf("Initializing renderers 0 - %d...\n", RENDERERS_COUNT);
	renderers[0] = new CPURenderer(RaycasterBase::raycaster);	
	renderers[1] = new GPURenderer1(RaycasterBase::raycaster);
	renderers[2] = new GPURenderer2(RaycasterBase::raycaster);
	renderers[3] = new GPURenderer3(RaycasterBase::raycaster);
	renderers[4] = new GPURenderer4(RaycasterBase::raycaster);
	
	printf("Initialization successful - entering main event loop...\n");
	//printf("Raycaster data size: %i B\n", sizeof(Raycaster));

	printf("\nUse '`1234' to change renderer\n    'wasd' and '7890' to manipulate camera position\n");
	printf("    'op' to change ray sampling rate\n");
	printf("    'nm' to change ray accumulation threshold\n    'r' to toggle autorotation\n");
	printf("    'l' to toggle empty space leaping\n");
	printf("    '[]' to change volume illumination intensity\n");
	printf("    '-' to toggle perspective and orthogonal projection\n");
	printf("    't' to toggle transfer function editor\n\n");

	UI::start();
	return EXIT_FAILURE;
}
