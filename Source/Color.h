#pragma once

#include "cuda_runtime.h"
#include "Vector3D.h"

struct Color
{
    __host__ __device__ inline Color() 
        : e{ 0, 0, 0 } 
    {}

    __host__ __device__ inline Color(float e0, float e1, float e2)
        : e{ e0, e1, e2 }
    {}

    __host__ __device__ inline float R() const {
        return e[0]; 
    }

    __host__ __device__ inline float G() const {
        return e[1]; 
    }

    __host__ __device__ inline float B() const {
        return e[2]; 
    }

    __host__ __device__ inline Color operator-() const {
        return Color(-e[0], -e[1], -e[2]);
    }

    __host__ __device__ inline float operator[](int i) const {
        return e[i];
    }

    __host__ __device__ inline float& operator[](int i) {
        return e[i];
    }

    __host__ __device__ inline Color& operator+=(const Color& c) {
        e[0] += c[0];
        e[1] += c[1];
        e[2] += c[2];
        return *this;
    }

    __host__ __device__ inline Color& operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline Color& operator*=(const Color& c) {
        e[0] *= c[0];
        e[1] *= c[1];
        e[2] *= c[2];
        return *this;
    }

    __host__ __device__ inline Color& operator/=(const float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ inline Color& operator/=(const Color& c) {
        e[0] /= c[0];
        e[1] /= c[1];
        e[2] /= c[2];
        return *this;
    }

private:
    float e[3];
};

__host__ __device__ inline Color operator+(const Color& u, const Color& v) {
    return Color(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

__host__ __device__ inline Color operator+(const Vector3D& u, const Color& v) {
    return Color(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

__host__ __device__ inline Color operator-(const Color& u, const Color& v) {
    return Color(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ inline Color operator*(const Color& u, const Color& v) {
    return Color(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

__host__ __device__ inline Color operator*(float t, const Color& v) {
    return Color(t * v[0], t * v[1], t * v[2]);
}

__host__ __device__ inline Color operator*(const Color& v, float t) {
    return t * v;
}

__host__ __device__ inline Color operator/(const Color& v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline Color Lerp(const Color& start, const Color& end, float alpha) {
    return (1.0f - alpha) * start + alpha * end;
}

__host__ inline Color GetRandomColor(float minValue = 0.0f, float maxValue = 1.0f) {
    return Color(
        MathUtils::GetRandomFloatInRange(minValue, maxValue),
        MathUtils::GetRandomFloatInRange(minValue, maxValue),
        MathUtils::GetRandomFloatInRange(minValue, maxValue)
    );
}





