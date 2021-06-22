#pragma once

#include "cuda_runtime.h"
#include "Hittable.h"
#include "Vector3D.h"


struct Sphere {

    __host__ __device__ inline Sphere() 
    : radius{ 0.0f }
    {}

    __host__ __device__ inline Sphere(Vector3D center, float r, Material material)
        : center{ center }, radius{ r }, material{ material } 
    {}

    __device__ inline bool Hit(const Ray& r, HitRecord& hitRecord) {

        Vector3D oc = r.origin - center;
        float a = Dot(r.direction, r.direction);
        float b = 2.0f * Dot(oc, r.direction);
        float c = Dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {     // no intersection
            return false;
        }

        float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        if (t < 0.0f) {
            return false;
        }
        
        hitRecord.t = t;
        hitRecord.hitPoint = r.At(t);
        Vector3D outwardNormal = MakeUnitVector((hitRecord.hitPoint - center) / radius);
        hitRecord.hasHitFrontSurface = Dot(r.direction, outwardNormal) < 0;
        hitRecord.hitNormal = hitRecord.hasHitFrontSurface ? outwardNormal : -outwardNormal;
        hitRecord.material = material;
        
        return true;

    }

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Vector3D center;
	float radius;
	Material material;
};