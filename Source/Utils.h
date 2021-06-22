#pragma once

#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

#include <limits>
#include <random>
#include <cstdlib>

#define PI 3.1415926535897932385f

namespace MathUtils {
	
	__host__ __device__ inline float degreesToRadians(float degrees) {
		return degrees * PI / 180.0f;
	}

	/* 
	// C++11
	__host__ inline float GetRandomFloat(float minValue = 0.0f, float maxValue = 1.0f) {
		std::random_device randDev;
		std::mt19937 generator(randDev());
		std::uniform_real_distribution<float> distribution(minValue, maxValue);
		return distribution(generator);

		//return minValue + ((float) rand()) / (((float) RAND_MAX) / (minValue - maxValue + 1.0f) + 1.0f);
		//return (float) rand() / (float) RAND_MAX;
	}
	*/

	// Returns a random float in [0,1).
	__host__ inline float GetRandomFloatInRangeZeroToOne() {
		return rand() / (RAND_MAX + 1.0f);
	}

	// Returns a random float in [min,max).
	__host__ inline float GetRandomFloatInRange(float min, float max) {
		return min + (max - min) * GetRandomFloatInRangeZeroToOne();
	}

	__host__ __device__ inline float Clamp(float value, float min, float max) {
		if (value < min) {
			return min;
		}
		if (value > max) {
			return max;
		}
		return value;
	}
}


namespace CudaUtils {

	// Returns a random float in range [-1,1], excluding 0.0f
	__device__ inline float GetRandomFloatInRangeMinusOneToOne(curandState* curandStates, int threadId) {
		float randomFloat = -1.0f + 2.0f * curand_uniform(&curandStates[threadId]);
		return randomFloat;
	}

	// Returns a random float in range (0,1]
	__device__ inline float GetRandomFloat(curandState* curandStates, int threadId) {
		float randomFloat = curand_uniform(&curandStates[threadId]);
		return randomFloat;
	}
}



