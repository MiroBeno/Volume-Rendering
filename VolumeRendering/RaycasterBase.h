/****************************************/
// Raycaster manager
/****************************************/

#ifndef _RAYCASTER_H_
#define _RAYCASTER_H_

#include "common.h"
#include "ModelBase.h"
#include "ViewBase.h"

#define ESL_MIN_BLOCK_SIZE 8
#define ESL_VOLUME_DIMS 32
#define ESL_VOLUME_SIZE 1024			// ESL_VOLUME_DIMS^3 / sizeof(esl_type) / 8 bits
#define TF_SIZE 128
#define TF_RATIO 2						// 256 / TF_SIZE

typedef unsigned int esl_type;

struct Raycaster {					// general rendering parameters and common functions
	Model volume;
	View view;
	float4 *transfer_fn;
	float ray_step;
	float ray_threshold;
	bool esl;
	esl_type *esl_volume;
	unsigned short esl_block_dims;
	float3 esl_block_size;
	float light_kd;

	inline __host__ __device__ bool intersect(float3 pt, float3 dir, float2 *k) {  // errors in computation results in values k outside of volume - need clamping
		if (dir.x == 0) dir.x = 0.00001f;					// todo: NaN if 0/0
		if (dir.y == 0) dir.y = 0.00001f;
		if (dir.z == 0) dir.z = 0.00001f;	
		float3 k1 = (volume.min_bound - pt) / dir;			// if vector component is parallel with axis and therefore with cube side (dir == 0), then
		float3 k2 = (-volume.min_bound - pt) / dir;				// if point is inside cube result is (-oo; +oo), otherwise (-oo;-oo) or (+oo;+oo) 
		k->x = flmax( flmax(flmin(k1.x, k2.x), flmin(k1.y, k2.y)), flmin(k1.z, k2.z) ); 
		k->y = flmin( flmin(flmax(k1.x, k2.x), flmax(k1.y, k2.y)), flmax(k1.z, k2.z) );
		k->x = flmax(k->x, 0);							// if x < 0 origin of ray is inside volume - not using intersection, but point of origin (k = 0)
		return ((k->x < k->y) && (k->y > 0));				// non-zero interval k (found intersection) AND exit point is on the ray
	}	

	inline __host__ __device__ void write_color(float4 color, short2 pos, uchar4 buffer[]) {
		buffer[pos.y * view.dims.x + pos.x] = 
			make_uchar4( map_float_int(color.x, 256), 
						map_float_int(color.y, 256), 
						map_float_int(color.z, 256), 
						map_float_int(color.w, 256));
	}

	inline __host__ __device__  bool sample_data_esl(esl_type esl_volume[], float3 pos) {
		/*unsigned short index = (	(map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims) * ESL_VOLUME_DIMS * ESL_VOLUME_DIMS +
									(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims) * ESL_VOLUME_DIMS +
									(map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims)
								);
		esl_type sample = esl_volume[index];
		return (sample == 0) ? false : true;*/
		unsigned short index = ((map_float_int((pos.z + 1)*0.5f, volume.dims.z) / esl_block_dims) * ESL_VOLUME_DIMS +
								(map_float_int((pos.y + 1)*0.5f, volume.dims.y) / esl_block_dims)
								);
		esl_type sample = esl_volume[index];
		index = map_float_int((pos.x + 1)*0.5f, volume.dims.x) / esl_block_dims;
		return ((sample & (1 << index)) != 0);
	}

	inline __host__ __device__ void leap_empty_space(float3 pt, float3 dir, float2 *k) {
		ushort3 index = make_ushort3(
			map_float_int((pt.x + 1)*0.5f, volume.dims.x) / esl_block_dims,
			map_float_int((pt.y + 1)*0.5f, volume.dims.y) / esl_block_dims,
			map_float_int((pt.z + 1)*0.5f, volume.dims.z) / esl_block_dims
		);
		if (dir.x > 0) index.x++;
		if (dir.y > 0) index.y++;
		if (dir.z > 0) index.z++;
		float3 kp = (volume.min_bound + (esl_block_size * index) - pt) / dir;
		if (dir.x == 0) kp.x = 100;	
		if (dir.y == 0) kp.y = 100;
		if (dir.z == 0) kp.z = 100;	
		float dk = flmin(kp.x, kp.y);
		dk = flmin(dk, kp.z);
		dk = flmax(dk, 0);
		dk = floor(dk / ray_step) * ray_step;
		k->x += dk;
	}

	inline __host__ __device__ void shade(float4 *color, float3 pos, unsigned char sample) {
		float3 light_dir = vector_normalize(view.light_pos - pos);
		float sample_l; 
		if (color->w > 0.05f && light_kd > 0.01f) 
			sample_l = volume.sample_data(pos + light_dir * 0.01f) / 255.0f;
		else
			sample_l = sample / 255.0f;
		float diffuse_light = (sample_l - sample / 255.0f) * light_kd;
		color->x += diffuse_light;
		color->y += diffuse_light;
		color->z += diffuse_light;
}

};

class RaycasterBase {
	public:
		static Raycaster raycaster;
		static float4 base_transfer_fn[TF_SIZE];
		static float2 ray_step_limits;
		static void change_ray_step(float step, bool reset);
		static void change_ray_threshold(float threshold, bool reset);
		static void change_light_intensity(float intensity, bool reset);
		static void toggle_esl();
		static void set_volume(Model volume);
		static void set_view(View view);
		static void update_transfer_fn();
		static void reset_transfer_fn();
		static void reset_ray_step();
	private:
		static uchar2 *esl_min_max;
};

#endif
