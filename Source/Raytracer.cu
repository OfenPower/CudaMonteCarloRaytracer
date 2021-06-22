#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "Color.h"
#include "Vector3D.h"
#include "Ray.h"
#include "Utils.h"
#include "Hittable.h"
#include "Sphere.h"
#include "Camera.h"

#include "stb_image_write.h"

#include <vector>
#include <stdio.h>
#include <random>
#include <float.h>
#include <cstdlib>

#include <Windows.h>
#include <winbase.h>


struct ImageRenderingAttributes {
	float aspectRatio;
	int width;
	int height;
	int samplesPerPixel;
	int maxRayBounceLimit;
	float gammaCorrectionExponent;
};

struct CUDAAttributes {
	int blockSize;
	int threadSize;
	int numRays;
	int numResultPixelColors;
	Camera* deviceCameraMemory;
	Ray* deviceRayMemory;
	Color* deviceSamplePixelColorMemory;
	Color* deviceResultPixelColorMemory;
	Sphere* deviceWorldMemory;
	curandState* deviceRandomState;
};

ImageRenderingAttributes imageRenderingAttributes;
CUDAAttributes cudaAttributes;

// Scene
const size_t sceneSize = 4 + 22 * 22;
Sphere sphereScene[sceneSize];

// Camera
Camera camera;

// Host functions
__host__ void InitializeFromConfigFile();
__host__ void SetupScene();
__host__ void InitializeCUDA();
__host__ void LaunchCUDAKernels();
__host__ void WriteResultsToFile();
__host__ void FreeCUDAMemory();

// CUDA kernels
__global__ void InitializeCuRand(curandState* states);
__global__ void CreateRays(int imageWidth, int imageHeight, int samplesPerPixel, Camera* camera, Ray* rays, int numRays, curandState* curandStates);
__global__ void DoRaytracing(curandState* curandStates, Ray* rays, int numRays, Color* pixelColors, Sphere* world, size_t worldSize, int maxRayBounceLimit);
__global__ void ReducePixelColorSamplesToFinalValue(Color* pixelColors, int samplesPerPixel, Color* reducedPixelColors, int numReducedPixelColors);
__global__ void DoGammaCorrection(Color* resultPixelColors, int numResultPixelColors, float gammaExponent);

/// <summary>
/// Main Function
/// </summary>
int main() {

	InitializeFromConfigFile();
	SetupScene();
	InitializeCUDA();
	LaunchCUDAKernels();
	
	// Wait for GPU computation to finish
	cudaDeviceSynchronize();

	// Error check
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	printf("Raytracing finished!\n");

	WriteResultsToFile();
	
	return 0;
}

__host__ void InitializeFromConfigFile() {

	const char* configFilePath = ".\\config.ini";
	
	// Image Settings
	char isbuffer1[10];
	GetPrivateProfileString("ImageSettings", "AspectRatio", "", isbuffer1, 10, configFilePath);
	float aspectRatio = (float)atof(isbuffer1);
	int imageWidth = GetPrivateProfileInt("ImageSettings", "ImageWidth", -1, configFilePath);
	char isbuffer2[10];
	GetPrivateProfileString("ImageSettings", "GammaCorrectionExponent", "", isbuffer2, 10, configFilePath);
	float gammaCorrectionExponent = (float)atof(isbuffer2);
	int samplesPerPixel = GetPrivateProfileInt("ImageSettings", "SamplesPerPixel", -1, configFilePath);
	int maxRayBounceLimit = GetPrivateProfileInt("ImageSettings", "MaxRayBounceLimit", -1, configFilePath);
	
	imageRenderingAttributes.aspectRatio = aspectRatio;
	imageRenderingAttributes.width = imageWidth;
	imageRenderingAttributes.height = (int)(imageWidth / aspectRatio);
	imageRenderingAttributes.gammaCorrectionExponent = gammaCorrectionExponent;
	imageRenderingAttributes.samplesPerPixel = samplesPerPixel;
	imageRenderingAttributes.maxRayBounceLimit = maxRayBounceLimit;

	//printf("%f, %d, %d, %f, %d, %d\n", aspectRatio, imageWidth, imageRenderingAttributes.height, gammaCorrectionExponent, samplesPerPixel, maxRayBounceLimit);

	// Camera Settings
	char csbuffer1[10];
	GetPrivateProfileString("CameraSettings", "LookFromX", "", csbuffer1, 10, configFilePath);
	float lookFromX = (float)atof(csbuffer1);
	char csbuffer2[10];
	GetPrivateProfileString("CameraSettings", "LookFromY", "", csbuffer2, 10, configFilePath);
	float lookFromY = (float)atof(csbuffer2);
	char csbuffer3[10];
	GetPrivateProfileString("CameraSettings", "LookFromZ", "", csbuffer3, 10, configFilePath);
	float lookFromZ = (float)atof(csbuffer3);
	char csbuffer4[10];
	GetPrivateProfileString("CameraSettings", "LookAtX", "", csbuffer4, 10, configFilePath);
	float lookAtX = (float)atof(csbuffer4);
	char csbuffer5[10];
	GetPrivateProfileString("CameraSettings", "LookAtY", "", csbuffer5, 10, configFilePath);
	float lookAtY = (float)atof(csbuffer5);
	char csbuffer6[10];
	GetPrivateProfileString("CameraSettings", "LookAtZ", "", csbuffer6, 10, configFilePath);
	float lookAtZ = (float)atof(csbuffer6);
	char csbuffer7[10];
	GetPrivateProfileString("CameraSettings", "VerticalFieldOfView", "", csbuffer7, 10, configFilePath);
	float verticalFieldOfView = (float)atof(csbuffer7);
	int useDepthOfField = GetPrivateProfileInt("CameraSettings", "UseDepthOfField", -1, configFilePath);
	char csbuffer8[10];
	GetPrivateProfileString("CameraSettings", "DistanceToFocus", "", csbuffer8, 10, configFilePath);
	float distanceToFocus = (float)atof(csbuffer8);
	char csbuffer9[10];
	GetPrivateProfileString("CameraSettings", "ApertureSize", "", csbuffer9, 10, configFilePath);
	float apertureSize = (float)atof(csbuffer9);

	Vector3D lookFrom(lookFromX, lookFromY, lookFromZ);
	Vector3D lookAt(lookAtX, lookAtY, lookAtZ);
	Vector3D up(0.0f, 1.0f, 0.0f);
	if (useDepthOfField == 0) {
		distanceToFocus = 1.0f;
		apertureSize = 0.0f;
	}

	//printf("%f, %f, %f, %f, %f, %f, %f, %d, %f, %f\n", lookFromX, lookFromY, lookFromZ, lookAtX, lookAtY, lookAtZ, verticalFieldOfView, useDepthOfField, distanceToFocus, apertureSize);

	camera.Initialize(lookFrom, lookAt, up, verticalFieldOfView, aspectRatio, apertureSize, distanceToFocus);
}

__host__ void SetupScene() {
	int index = 0;

	Material groundMaterial;
	groundMaterial.albedoColor = Color(0.5f, 0.5f, 0.5f);
	groundMaterial.isMetal = false;
	groundMaterial.isDielectric = false;
	groundMaterial.refractionIndex = 0.0f;
	groundMaterial.metalFuzziness = 0.0f;
	Sphere groundSphere(Vector3D(0.0f, -1000.0f, 0.0f), 1000.0f, groundMaterial);
	sphereScene[index++] = groundSphere;

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float randomNumberForMaterialSelection = MathUtils::GetRandomFloatInRangeZeroToOne();
			Vector3D center(a + 0.9f * MathUtils::GetRandomFloatInRangeZeroToOne(), 0.2f, b + 0.9f * MathUtils::GetRandomFloatInRangeZeroToOne());
			if ((center - Vector3D(4.0f, 0.2f, 0.0f)).Size() > 0.9f) {
				if (randomNumberForMaterialSelection < 0.4f) {
					// Diffuse Material
					Color albedo = GetRandomColor() * GetRandomColor();
					Material sphereMaterial;
					sphereMaterial.albedoColor = albedo;
					sphereMaterial.isMetal = false;
					sphereMaterial.isDielectric = false;
					sphereMaterial.refractionIndex = 0.0f;
					sphereMaterial.metalFuzziness = 0.0f;
					Sphere sphere(center, 0.2f, sphereMaterial);
					sphereScene[index++] = sphere;
				}
				else if (randomNumberForMaterialSelection < 0.9f) {
					// Metal Material
					Color albedo = GetRandomColor(0.5f, 1.0f);
					float fuzziness = MathUtils::GetRandomFloatInRange(0.0f, 0.2f);
					Material sphereMaterial;
					sphereMaterial.albedoColor = albedo;
					sphereMaterial.isMetal = true;
					sphereMaterial.isDielectric = false;
					sphereMaterial.refractionIndex = 0.0f;
					sphereMaterial.metalFuzziness = fuzziness;
					Sphere sphere(center, 0.2f, sphereMaterial);
					sphereScene[index++] = sphere;
				}
				else {
					// Glass Material
					Material sphereMaterial;
					sphereMaterial.albedoColor = Color(1.0f, 1.0f, 1.0f);
					sphereMaterial.isMetal = false;
					sphereMaterial.isDielectric = true;
					sphereMaterial.refractionIndex = 1.5f;	
					sphereMaterial.metalFuzziness = 0.0f;
					Sphere sphere(center, 0.2f, sphereMaterial);
					sphereScene[index++] = sphere;
				}
			}
		}
	}

	Material material1;
	material1.albedoColor = Color(1.0f, 1.0f, 1.0f);
	material1.isMetal = false;
	material1.isDielectric = true;
	material1.refractionIndex = 1.5f;
	material1.metalFuzziness = 0.0f;
	Material material2;
	material2.albedoColor = Color(0.4f, 0.2f, 0.1f);
	material2.isMetal = false;
	material2.isDielectric = false;
	material2.refractionIndex = 0.0f;
	material2.metalFuzziness = 0.0f;
	Material material3;
	material3.albedoColor = Color(0.7f, 0.6f, 0.5f);
	material3.isMetal = true;
	material3.isDielectric = false;
	material3.refractionIndex = 0.0f;
	material3.metalFuzziness = 0.0f;
	Sphere sphere1(Vector3D(0.0f, 1.0f, 0.0f), 1.0f, material1);
	Sphere sphere2(Vector3D(-4.0f, 1.0f, 0.0f), 1.0f, material3);
	Sphere sphere3(Vector3D(4.0f, 1.0f, 0.0f), 1.0f, material2);
	sphereScene[index++] = sphere1;
	sphereScene[index++] = sphere2;
	sphereScene[index++] = sphere3;
}

__host__ void InitializeCUDA() {
	// Set GPU 
	cudaSetDevice(0);

	// Determine how many rays have to be generated and how many result pixels there are
	cudaAttributes.numRays = imageRenderingAttributes.width * imageRenderingAttributes.height * imageRenderingAttributes.samplesPerPixel;
	cudaAttributes.numResultPixelColors = cudaAttributes.numRays / imageRenderingAttributes.samplesPerPixel;

	// Allocate GPU memory
	cudaMalloc(&cudaAttributes.deviceRayMemory, cudaAttributes.numRays * sizeof(Ray));
	cudaMalloc(&cudaAttributes.deviceCameraMemory, sizeof(Camera));
	cudaMalloc(&cudaAttributes.deviceSamplePixelColorMemory, cudaAttributes.numRays * sizeof(Color));
	cudaMalloc(&cudaAttributes.deviceResultPixelColorMemory, cudaAttributes.numResultPixelColors * sizeof(Color));
	cudaMalloc(&cudaAttributes.deviceWorldMemory, sceneSize * sizeof(Sphere));

	// Copy data to GPU memory
	cudaMemcpy(cudaAttributes.deviceCameraMemory, &camera, sizeof(Camera), cudaMemcpyHostToDevice);		// copy camera
	cudaMemcpy(cudaAttributes.deviceWorldMemory, &sphereScene[0], sceneSize * sizeof(Sphere), cudaMemcpyHostToDevice);	// copy world

	// Set thread Size and determine block size
	cudaAttributes.threadSize = 1024;
	if (cudaAttributes.numRays % cudaAttributes.threadSize == 0) {
		cudaAttributes.blockSize = cudaAttributes.numRays / cudaAttributes.threadSize;
	}
	else {
		// Add one more block if needed. There will be idle threads in the last block, since the number of threads exceed the number of generated rays 
		cudaAttributes.blockSize = (int)(floor(cudaAttributes.numRays / cudaAttributes.threadSize)) + 1;	
	}

	// Allocate memory for random generators for every thread
	cudaMalloc(&cudaAttributes.deviceRandomState, cudaAttributes.blockSize * cudaAttributes.threadSize * sizeof(curandState));

	// Calculate memory usage in VRAM and print it for information 
	unsigned long long int VRAMmemoryInBytes = 2 * cudaAttributes.numRays * sizeof(Ray) + 
									sizeof(Camera) + cudaAttributes.numResultPixelColors * sizeof(Color) + 
									sceneSize * sizeof(Sphere) + 
									cudaAttributes.blockSize * cudaAttributes.threadSize * sizeof(curandState);
	float VRAMmemoryInGigabytes = VRAMmemoryInBytes / 1024.0f / 1024.0f / 1024.0f;
	printf("GPU VRAM: Allocated Gigabytes (GB): %f\n", VRAMmemoryInGigabytes);
}

__host__ void LaunchCUDAKernels() {

	// Setup the random generators, one per each Thread
	InitializeCuRand << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (cudaAttributes.deviceRandomState);

	// Create one ray per thread
	CreateRays << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (
		imageRenderingAttributes.width,
		imageRenderingAttributes.height,
		imageRenderingAttributes.samplesPerPixel,
		cudaAttributes.deviceCameraMemory,
		cudaAttributes.deviceRayMemory,
		cudaAttributes.numRays,
		cudaAttributes.deviceRandomState);

	// Render the scene 
	DoRaytracing << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (
		cudaAttributes.deviceRandomState,
		cudaAttributes.deviceRayMemory,
		cudaAttributes.numRays,
		cudaAttributes.deviceSamplePixelColorMemory,
		cudaAttributes.deviceWorldMemory,
		sceneSize,
		imageRenderingAttributes.maxRayBounceLimit);

	// The pixel color samples of one pixel need to be reduced to one final pixel color, if there are more than two samples per pixel 
	if (imageRenderingAttributes.samplesPerPixel >= 2) {
		ReducePixelColorSamplesToFinalValue << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (
			cudaAttributes.deviceSamplePixelColorMemory,
			imageRenderingAttributes.samplesPerPixel,
			cudaAttributes.deviceResultPixelColorMemory,
			cudaAttributes.numResultPixelColors);
		
		DoGammaCorrection << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (
			cudaAttributes.deviceResultPixelColorMemory,
			cudaAttributes.numResultPixelColors,
			imageRenderingAttributes.gammaCorrectionExponent);
	}
	else {
		DoGammaCorrection << <cudaAttributes.blockSize, cudaAttributes.threadSize >> > (
			cudaAttributes.deviceSamplePixelColorMemory,
			cudaAttributes.numResultPixelColors,
			imageRenderingAttributes.gammaCorrectionExponent);
	}
}

__global__ void InitializeCuRand(curandState* states) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int seed = id; // Different seed per thread
	curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND
}

__global__ void CreateRays(int imageWidth, int imageHeight, int samplesPerPixel, Camera* camera, Ray* rays, int numRays, curandState* curandStates) {
	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (threadId >= numRays) {
		return;
	}

	int pixelX = (int)(threadId / samplesPerPixel) % imageWidth;	// sample the image from left to right row-wise. One pixel is sampled by "samplesPerPixel"-many consecutive threads
	int pixelY = (imageHeight - 1) - (int)((int)(threadId / samplesPerPixel) / imageWidth);	// sample the image from top to bottom with the same sample-logic as above

	float u = (float)(pixelX + CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId)) / (float)(imageWidth - 1);
	float v = (float)(pixelY + CudaUtils::GetRandomFloatInRangeMinusOneToOne(curandStates, threadId)) / (float)(imageHeight - 1);

	Ray ray = camera->GetRay(u, v, curandStates, threadId);
	rays[threadId] = ray;
}

__global__ void DoRaytracing(curandState* curandStates, Ray* rays, int numRays, Color* pixelColors, Sphere* world, size_t worldSize, int maxRayBounceLimit) {

	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (threadId >= numRays) {
		return;
	}

	Ray rayForCurrentThread = rays[threadId];
	Color rayColor(1.0f, 1.0f, 1.0f);
	
	// Shoot ray to scene, check collisions and determine the ray color with the hit materials as long as the bounce limit of a ray is not reached or nothing in the scene is hit
	for (size_t i = 0; i < maxRayBounceLimit; ++i) {
		bool hasHitAnything = false;
		HitRecord closestHit;
		float tmpDistance = FLT_MAX;
		for (size_t i = 0; i < worldSize; ++i) {
			HitRecord tmpRecord;
			bool hit = world[i].Hit(rayForCurrentThread, tmpRecord);
			if (!hit) {
				continue;
			}
			hasHitAnything = true;

			// Check if ray length was shorter than the last length, resulting in the hit record of the nearest object
			if (tmpRecord.t < tmpDistance) {
				closestHit = tmpRecord;
				tmpDistance = tmpRecord.t;
			}
		}

		// If ray hit nothing in scene, stop raytrace
		if (!hasHitAnything) {
			break;
		}

		// Shade the ray with the material from the nearest hit object
		Color attenuation;
		Ray scatteredRay;
		if (closestHit.material.isMetal) {
			ScatterMetal(rayForCurrentThread, closestHit, attenuation, scatteredRay, closestHit.material, curandStates, threadId);
		}
		else if (closestHit.material.isDielectric) {
			ScatterDielectric(rayForCurrentThread, closestHit, attenuation, scatteredRay, closestHit.material, curandStates, threadId);
		}
		else {
			ScatterLambertian(rayForCurrentThread, closestHit, attenuation, scatteredRay, closestHit.material, curandStates, threadId);
		}
		rayForCurrentThread = scatteredRay;	// Ray for next iteration is the scattered ray
		rayColor *= attenuation;	// Attenuate the RayColor with the shading result

	}

	// Attenuate the ray with a blue-white gradient sky color as a light source
	Vector3D unitDirection = MakeUnitVector(rayForCurrentThread.direction);
	float t = 0.5f * (unitDirection.Y() + 1.0f);
	Color brightWhite(1.0f, 1.0f, 1.0f);
	Color blueish(0.5f, 0.7f, 1.0f);
	Color skyColor = Lerp(brightWhite, blueish, t);
	rayColor *= skyColor;

	pixelColors[threadId] = rayColor;
}

__global__ void ReducePixelColorSamplesToFinalValue(Color* pixelColors, int samplesPerPixel, Color* reducedPixelColors, int numReducedPixelColors) {

	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (threadId >= numReducedPixelColors) {
		return;
	}

	// Reduce the pixel color samples to one final pixel color. One thread is responsible for the reduction of all samples of one pixel
	int startIndex = threadId * samplesPerPixel;
	int endIndex = (startIndex + samplesPerPixel) - 1;
	Color resultColor(0.0f, 0.0f, 0.0f);
	for (int i = startIndex; i < endIndex; ++i) {
		resultColor += pixelColors[i];
	}

	// The final pixel color value is averaged through all samples and than saved 
	reducedPixelColors[threadId] = resultColor / samplesPerPixel;

}

__global__ void DoGammaCorrection(Color* resultPixelColors, int numResultPixelColors, float gammaExponent/*, unsigned char* resultColorValues*/) {

	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (threadId >= numResultPixelColors) {
		return;
	}

	// Apply gamma correction by exponentiating the color with the gamma value
	Color color = resultPixelColors[threadId];
	color[0] = pow(color[0], gammaExponent);
	color[1] = pow(color[1], gammaExponent);
	color[2] = pow(color[2], gammaExponent);
	resultPixelColors[threadId] = color;
}

__host__ void WriteResultsToFile() {
	// Copy results from device memory to host memory
	Color* hostColorMemory = new Color[cudaAttributes.numResultPixelColors];
	if (imageRenderingAttributes.samplesPerPixel >= 2) {
		cudaMemcpy(hostColorMemory, cudaAttributes.deviceResultPixelColorMemory, cudaAttributes.numResultPixelColors * sizeof(Color), cudaMemcpyDeviceToHost);
	}
	else {
		cudaMemcpy(hostColorMemory, cudaAttributes.deviceSamplePixelColorMemory, cudaAttributes.numResultPixelColors * sizeof(Color), cudaMemcpyDeviceToHost);
	}

	FreeCUDAMemory();

	unsigned char* pixelData = new unsigned char[imageRenderingAttributes.width * imageRenderingAttributes.height * 3];	// there are 3 RGB values for each pixel, so the size is multiplied by 3
	size_t colorIndex = 0;
	for (size_t i = 0; i < cudaAttributes.numResultPixelColors; ++i) {
		Color resultColor = hostColorMemory[i];
		pixelData[colorIndex++] = static_cast<int>(255.0f * MathUtils::Clamp(resultColor.R(), 0.0f, 1.0f));
		pixelData[colorIndex++] = static_cast<int>(255.0f * MathUtils::Clamp(resultColor.G(), 0.0f, 1.0f));
		pixelData[colorIndex++] = static_cast<int>(255.0f * MathUtils::Clamp(resultColor.B(), 0.0f, 1.0f));
	}

	// Write PNG image and open it immediately
	stbi_write_png("raytraced_image.png", imageRenderingAttributes.width, imageRenderingAttributes.height, 3, pixelData, 0);
	std::system("start raytraced_image.png");

	delete[] pixelData;
	delete[] hostColorMemory;
}

__host__ void FreeCUDAMemory() {
	cudaFree(cudaAttributes.deviceCameraMemory);
	cudaFree(cudaAttributes.deviceRayMemory);
	cudaFree(cudaAttributes.deviceSamplePixelColorMemory);
	cudaFree(cudaAttributes.deviceResultPixelColorMemory);
	cudaFree(cudaAttributes.deviceWorldMemory);
}