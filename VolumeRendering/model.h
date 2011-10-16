#ifndef _MODEL_H_
#define _MODEL_H_

#include "datatypes.h"

int load_model(const char* file_name);
float4 render_ray(float3 origin, float3 direction);
float4 render_ray_alt(float3 origin, float3 direction);

#endif