#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <float.h>
#include <math.h>

#define MAXIMUM(a,b) ((a)>(b)?(a):(b))
#define MINIMUM(a,b) ((a)<(b)?(a):(b))
#define PI 3.14159265f
#define DEG_TO_RAD(a) ((a) * PI / 180)

const float POS_INF = FLT_MAX, NEG_INF = FLT_MIN;

typedef struct {
	float a, b;
} float2;

typedef struct { 
	float x, y, z;
} float3;

typedef struct { 
	float r, g, b, a;
} float4;

inline float2 mk_float2(float a, float b) {
	float2 result;
	result.a = a;
	result.b = b;
	return result;
}

inline float3 mk_float3(float x, float y, float z) {
	float3 result;
	result.x = x;
	result.y = y;
	result.z = z;
	return result;
}

inline float4 mk_float4(float r, float g, float b, float a) {
	float4 result;
	result.r = r;
	result.g = g;
	result.b = b;
	result.a = a;
	return result;
}

// float3 ops

inline float3 add(const float3 a, const float3 b) {
	return mk_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 mul(const float3 a, float b) {
	return mk_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 cross_product(float3 a, float3 b) {					// vektorovy sucin
	return mk_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline float vector_length(float3 v) {
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline float3 normalize_vector(const float3 v) {
	return mul(v, 1.0f / vector_length(v));
}

// float4 ops

inline float4 add(const float4 a, const float4 b) {
	return mk_float4(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
}

inline float4 mul(const float4 a, float b) {
	return mk_float4(a.r * b, a.g * b, a.b * b, a.a * b);
}

// other

inline unsigned int map_interval(float f, unsigned int n) {			// mapuje float<0,1> na int<0,n-1>
   long i = (long) (f * n);
   if (i >= (int) n) i = n - 1;
   if (i < 0) i = 0;
   return (unsigned int) i;
}

#endif