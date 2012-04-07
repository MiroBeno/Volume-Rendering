#ifndef _RENDERER_H_
#define _RENDERER_H_

#include "data_utils.h"
#include "View.h"
#include "Model.h"
#include "Raycaster.h"

class Renderer {
	public:
		virtual ~Renderer() {
			};
		virtual void set_window_buffer(int2 size) {
			};
		virtual void set_transfer_fn(float4 *tf) {
			};
		virtual void set_volume(Model volume, unsigned char *d) {
			};
		virtual int render_volume(uchar4 *buffer, Raycaster *r) = 0;
};

class CPURenderer: public Renderer {
	public:
		CPURenderer(int2 size, float4 *tf, Model volume, unsigned char *d);
		virtual void set_transfer_fn(float4 *tf);
		virtual void set_volume(Model volume, unsigned char *d);
		virtual int render_volume(uchar4 *buffer, Raycaster *r);
};

class GPURenderer: public Renderer {
	protected:
		static uchar4 *dev_buffer;
		static int dev_buffer_size;
		static unsigned char *dev_volume_data;
		static dim3 THREADS_PER_BLOCK;				
		static dim3 num_blocks;
};

class GPURenderer1: public GPURenderer {
	public:
		GPURenderer1(int2 size, float4 *tf, Model volume, unsigned char *d);
		virtual ~GPURenderer1();
		virtual void set_window_buffer(int2 size);
		virtual void set_transfer_fn(float4 *tf);
		virtual void set_volume(Model volume, unsigned char *d);
		virtual int render_volume(uchar4 *buffer, Raycaster *r);
};

class GPURenderer2: public GPURenderer {
	public:
		GPURenderer2(int2 size, float4 *tf, Model volume, unsigned char *d);
		virtual void set_transfer_fn(float4 *tf);
		virtual int render_volume(uchar4 *buffer, Raycaster *r);
};

class GPURenderer3: public GPURenderer {
	public:
		GPURenderer3(int2 size, float4 *tf, Model volume, unsigned char *d);
		virtual int render_volume(uchar4 *buffer, Raycaster *r);
};

class GPURenderer4: public GPURenderer {
	public:
		GPURenderer4(int2 size, float4 *tf, Model volume, unsigned char *d);
		virtual ~GPURenderer4();
		virtual void set_transfer_fn(float4 *tf);
		virtual void set_volume(Model volume, unsigned char *d);
		virtual int render_volume(uchar4 *buffer, Raycaster *r);
};

#endif