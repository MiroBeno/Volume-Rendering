#ifndef _COMMON_H_
#define _COMMON_H_

#include <math.h>

#include "host_defines.h"
#include "vector_types.h"
#include "vector_functions.h"

#define RENDERER_COUNT 5

#define PI 3.141592654f
#define MAXIMUM(a,b) ((a)>(b)?(a):(b))
#define MINIMUM(a,b) ((a)<(b)?(a):(b))
#define CLAMP(x,low,high) (MINIMUM((high),MAXIMUM((low),(x))))		//vyhodit
#define DEG_TO_RAD(a) ((a) * PI / 180)

// float3 ops

inline __host__ __device__ float3 operator-(float3 a)			
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float3 a, ushort3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 cross_product(float3 a, float3 b) {					// vektorovy sucin
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float vector_length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float3 vector_normalize(float3 v) {
	return (v * (1.0f / vector_length(v)));
}

// float4 ops

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline __host__ __device__ float4 operator+(float4 a, float3 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w);
}

inline __host__ __device__ float flmin(float a, float b)
{
  return a < b ? a : b;
}

inline __host__ __device__ float flmax(float a, float b)
{
  return a > b ? a : b;
}

// other

inline __host__ __device__ unsigned int map_float_int(float f, unsigned int n) {		// mapuje float<0,1> na int<0,n-1>
   long i = (long) (f * n);
   if (i >= (int) n) i = n - 1;
   if (i < 0) i = 0;
   return (unsigned int) i;
}

#endif