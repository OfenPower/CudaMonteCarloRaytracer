#pragma once

#include "cuda_runtime.h"
#include "Utils.h"

#include <cmath>


struct Vector3D
{
    __host__ __device__ inline Vector3D() 
        : e{0, 0, 0} 
    {}

    __host__ __device__ inline Vector3D(float e0, float e1, float e2)
        : e{e0, e1, e2}
    {}



    __host__ __device__ inline float X() const { 
        return e[0]; 
    }

    __host__ __device__ inline float Y() const { 
        return e[1]; 
    }

    __host__ __device__ inline float Z() const { 
        return e[2]; 
    }

    __host__ __device__ inline Vector3D operator-() const { 
        return Vector3D(-e[0], -e[1], -e[2]); 
    }

    __host__ __device__ inline float operator[](int i) const {
        return e[i]; 
    }

    __host__ __device__ inline float& operator[](int i) { 
        return e[i]; 
    
    }
    __host__ __device__ inline Vector3D& operator+=(const Vector3D& v) {
        e[0] += v[0];
        e[1] += v[1];
        e[2] += v[2];
        return *this;
    }

    __host__ __device__ inline Vector3D& operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline Vector3D& operator/=(const float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ inline float Size() const {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    __host__ __device__ inline float SizeSquared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ bool IsNearZero() const {
        const double epsilon = 1e-8;
        return (fabs(e[0]) < epsilon) && (fabs(e[1]) < epsilon) && (fabs(e[2]) < epsilon);
    }

private:
    float e[3];
};

__host__ __device__ inline Vector3D operator+(const Vector3D& u, const Vector3D& v) {
    return Vector3D(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

__host__ __device__ inline Vector3D operator-(const Vector3D& u, const Vector3D& v) {
    return Vector3D(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ inline Vector3D operator*(const Vector3D& u, const Vector3D& v) {
    return Vector3D(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

__host__ __device__ inline Vector3D operator*(float t, const Vector3D& v) {
    return Vector3D(t * v[0], t * v[1], t * v[2]);
}

__host__ __device__ inline Vector3D operator*(const Vector3D& v, float t) {
    return t * v;
}

__host__ __device__ inline Vector3D operator/(Vector3D v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline float Dot(const Vector3D& u, const Vector3D& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__host__ __device__ inline Vector3D Cross(const Vector3D& u, const Vector3D& v) {
    return Vector3D(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
}

__host__ __device__ inline Vector3D MakeUnitVector(Vector3D v) {
    return v / v.Size();
}

__host__ inline Vector3D GetRandomVector(float minValue = 0.0f, float maxValue = 1.0f) {
    return Vector3D(
        MathUtils::GetRandomFloatInRange(minValue, maxValue),
        MathUtils::GetRandomFloatInRange(minValue, maxValue),
        MathUtils::GetRandomFloatInRange(minValue, maxValue)
    );
}

__host__ inline Vector3D GetRandomPointInUnitSphere() {
    
    while (true) {   
        Vector3D point = GetRandomVector(-1.0f, 1.0f);
        if (point.SizeSquared() >= 1) {
            continue;
        }
        return point;
    }
}

__host__ inline Vector3D GetRandomUnitVectorInUnitSphere() {
    return MakeUnitVector(GetRandomPointInUnitSphere());
}

__host__ inline Vector3D GetRandomUnitVectorInHemisphere(const Vector3D& hemisphereNormal) {
    Vector3D randomUnitVectorInUnitSphere = GetRandomUnitVectorInUnitSphere();
    if (Dot(randomUnitVectorInUnitSphere, hemisphereNormal) > 0.0) { // In the same hemisphere as the normal
        return randomUnitVectorInUnitSphere;
    }
    else {
        return -randomUnitVectorInUnitSphere;
    }
}

__device__ inline Vector3D CudaGetRandomVectorInRangeMinusOneToOne(curandState* curandStates, int threadId) {
    return Vector3D(
        CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId),
        CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId),
        CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId)
    );
}

__device__ inline Vector3D CudaGetRandomPointInUnitSphere(curandState* curandStates, int threadId) {

    while (true) {
        Vector3D point = CudaGetRandomVectorInRangeMinusOneToOne(curandStates, threadId);
        if (point.SizeSquared() >= 1) {
            continue;
        }
        return point;
    }
}

__device__ inline Vector3D CudaGetRandomUnitVectorInUnitSphere(curandState* curandStates, int threadId) {
    return MakeUnitVector(CudaGetRandomPointInUnitSphere(curandStates, threadId));
}

__device__ inline Vector3D CudaGetRandomUnitVectorInHemisphere(const Vector3D& hemisphereNormal, curandState* curandStates, int threadId) {
    Vector3D randomUnitVectorInUnitSphere = CudaGetRandomUnitVectorInUnitSphere(curandStates, threadId);
    if (Dot(randomUnitVectorInUnitSphere, hemisphereNormal) > 0.0) { // In the same hemisphere as the normal
        return randomUnitVectorInUnitSphere;
    }
    else {
        return -randomUnitVectorInUnitSphere;
    }
}

__host__ __device__ inline Vector3D Reflect(const Vector3D v, const Vector3D& n) {
    return v - 2 * Dot(v, n) * n;
}

__device__ inline Vector3D Refract(const Vector3D& v, const Vector3D& n, float etaFractureValue) {
    float cosTheta = fmin(Dot(-v, n), 1.0f);
    Vector3D rOutPerpendicular = etaFractureValue * (v + cosTheta * n);
    Vector3D rOutParallel = -sqrt(fabs(1.0f - rOutPerpendicular.SizeSquared())) * n;
    return rOutPerpendicular + rOutParallel;
}

__device__ inline Vector3D GetRandomVectorInUnitDisk(curandState* curandStates, int threadId) {
    while (true) {
        Vector3D p = Vector3D(CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId), CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId), 0);
        if (p.SizeSquared() >= 1) {
            continue;
        }
        return p;
    }
}



