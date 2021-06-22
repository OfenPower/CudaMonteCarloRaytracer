#pragma once

#include "cuda_runtime.h"
#include "Vector3D.h"
#include "Ray.h"
#include "Utils.h"

struct Camera {

	__host__ inline void Initialize(
		Vector3D lookFrom,
		Vector3D lookAt,
		Vector3D up,
		float verticalFieldOfView,
		float aspectRatio,
		float aperture,
		float focusDist)
	{
		
		float thetaAngleInRadians = MathUtils::degreesToRadians(verticalFieldOfView);
		float h = tan(thetaAngleInRadians / 2.0f);
		float viewportHeight = 2.0f * h;
		float viewportWidth = aspectRatio * viewportHeight;

		// Gram-Schmidt Procedure
		zCameraAxis = MakeUnitVector(lookFrom - lookAt);		// Camera faces in -zCameraAxis direction!
		xCameraAxis = MakeUnitVector(Cross(up, zCameraAxis));
		yCameraAxis = Cross(zCameraAxis, xCameraAxis);

		origin = lookFrom;
		viewportHorizontalSize = focusDist * viewportWidth * xCameraAxis;
		viewportVerticalSize = focusDist * viewportHeight * yCameraAxis;
		lowerLeftCorner = origin - viewportHorizontalSize*0.5f - viewportVerticalSize*0.5f - focusDist*zCameraAxis;
		
		lensRadius = aperture * 0.5f;
	}

	__device__ inline Ray GetRay(float u, float v, curandState* curandStates, int threadId) const {
		Vector3D randomVectorOnLensDisk = lensRadius * GetRandomVectorInUnitDisk(curandStates, threadId);
		Vector3D offset = xCameraAxis * randomVectorOnLensDisk.X() + yCameraAxis * randomVectorOnLensDisk.Y();

		return Ray(
			origin + offset, 
			MakeUnitVector(lowerLeftCorner + u * viewportHorizontalSize + v * viewportVerticalSize - origin - offset));
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Vector3D origin;
	Vector3D lowerLeftCorner;
	Vector3D viewportHorizontalSize;
	Vector3D viewportVerticalSize;
	Vector3D zCameraAxis, xCameraAxis, yCameraAxis;
	float lensRadius;
};
