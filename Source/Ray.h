#pragma once

#include "cuda_runtime.h"
#include "Vector3D.h"

struct Ray 
{
    __host__ __device__ inline Ray() {}

    __host__ __device__ inline Ray(const Vector3D& origin, const Vector3D& direction)
        : origin(origin), direction(direction) {}

    __host__ __device__ inline Vector3D At(float t) const {
        return origin + t * direction; 
    }

    ////////////////////////////////////////////////////////////////////////////////////
    Vector3D origin;
    Vector3D direction;
};

