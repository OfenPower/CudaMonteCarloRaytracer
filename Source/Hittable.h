#pragma once

#include "cuda_runtime.h"
#include "Ray.h"
#include "curand.h"
#include "Color.h"
#include "Utils.h"

struct Material {
	Color albedoColor;
	bool isMetal;
	bool isDielectric;
	float metalFuzziness;
	float refractionIndex;
};

struct HitRecord {
	Vector3D hitPoint;
	Vector3D hitNormal;
	Material material;
	float t;
	bool hasHitFrontSurface;
};

// Code from https://raytracing.github.io/books/RayTracingInOneWeekend.html
__device__ inline void ScatterLambertian(
	const Ray& rayIn,
	const HitRecord& hitRecord,
	Color& attenuation,
	Ray& scatteredRay,
	Material& material,
	curandState* curandStates,
	int threadId)
{
	Vector3D scatterDirection = hitRecord.hitNormal + CudaGetRandomUnitVectorInHemisphere(hitRecord.hitNormal, curandStates, threadId);
	scatteredRay = Ray(hitRecord.hitPoint, MakeUnitVector(scatterDirection));
	attenuation = material.albedoColor;
}

// Code from https://raytracing.github.io/books/RayTracingInOneWeekend.html
 __device__ inline void ScatterMetal(
	const Ray& rayIn, 
	const HitRecord& hitRecord, 
	Color& attenuation,
	Ray& scatteredRay,
	Material& material,
	curandState* curandStates,
	int threadId)
{
	Vector3D reflected = Reflect(rayIn.direction, hitRecord.hitNormal);
	Vector3D scatterDirection = reflected + (material.metalFuzziness * CudaGetRandomUnitVectorInHemisphere(hitRecord.hitNormal, curandStates, threadId));
	scatteredRay = Ray(hitRecord.hitPoint, MakeUnitVector(scatterDirection));
	attenuation = material.albedoColor;
}


__device__ float CalculateSchlickReflectanceApproximation(float cosine, float reflectanceIndex);

// Code from https://raytracing.github.io/books/RayTracingInOneWeekend.html
__device__ inline void ScatterDielectric(
	const Ray& rayIn, 
	const HitRecord& hitRecord, 
	Color& attenuation, 
	Ray& scatteredRay,
	Material& material,
	curandState* curandStates,
	int threadId)
{
	// no color will be absorbed from glas surface
	attenuation = Color(1.0, 1.0, 1.0);	

	float refractionRatio;
	if (hitRecord.hasHitFrontSurface) {
		refractionRatio = 1.0f / material.refractionIndex;
	}
	else {
		refractionRatio = material.refractionIndex;
	}

	float cosTheta = fmin(Dot(-rayIn.direction, hitRecord.hitNormal), 1.0f);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	bool cannotRefract = refractionRatio * sinTheta > 1.0f;
	Vector3D direction;
	float reflectance = CalculateSchlickReflectanceApproximation(cosTheta, refractionRatio);
	if (cannotRefract || reflectance > CudaUtils::GetRandomFloat(curandStates, threadId))
		direction = Reflect(rayIn.direction, hitRecord.hitNormal);
	else
		direction = Refract(rayIn.direction, hitRecord.hitNormal, refractionRatio);

	scatteredRay = Ray(hitRecord.hitPoint, MakeUnitVector(direction));
}

// Code from https://raytracing.github.io/books/RayTracingInOneWeekend.html
__device__ inline float CalculateSchlickReflectanceApproximation(float cosine, float reflectanceIndex) 
{
	float r0 = (1.0f - reflectanceIndex) / (1.0f + reflectanceIndex);
	r0 = r0 * r0;
	float reflectance = r0 + (1 - r0) * pow((1 - cosine), 5);
	return reflectance;
}

