#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "device_launch_parameters.h"
#define THREAD 16



void checkCUDAError()
{
	cudaError_t error = cudaGetLastError();
	if (error != CUDA_SUCCESS) {
		printf("ERROR: %s: %s\n", cudaGetErrorString(error));
	}
	else {
		printf("Kernel launch successfully. \n");
	}
	//cout << cudaGetErrorString(error);
}

__global__ void rgba_to_greyscale(
	uchar4* rgbaImage,
	unsigned char*   greyImage,
	int		numRows,
	int		numCols)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	// map the two 2D indices to a single linear, 1D index
	if (index_x < numCols && index_y < numRows) {
		//int grid_width = gridDim.x * blockDim.x;
		int index = index_y * numCols + index_x;
		greyImage[index] = .299f * rgbaImage[index].x + .587f * rgbaImage[index].y + .114f * rgbaImage[index].z;
	}
}


__global__ void image_blur_avg(
	uchar4* rgbaImage,
	uchar4* blurImage,
	int		numRows,
	int		numCols
) {
	int index_x = blockDim.x * blockIdx.x + threadIdx.x;
	int index_y = blockDim.y * blockIdx.y + threadIdx.y;
	int linear_index = index_y * numCols + index_x;
	if (index_x < numCols && index_y < numRows) {
		float pixelR = 0;
		float pixelG = 0;
		float pixelB = 0;
		int pixels = 0;
		for (int blurRow = -3; blurRow < 4; blurRow++) {
			for (int blurCol = -3; blurCol < 4; blurCol++) {
				int curRow = index_y + blurRow;
				int curCol = index_x + blurCol;
				if (curRow > -1 && curRow < numRows && curCol > -1 && curCol < numCols) {
					int index = curRow * numCols + curCol;
					pixelR += rgbaImage[index].x;
					pixelG += rgbaImage[index].y;
					pixelB += rgbaImage[index].z;
					pixels++;
				}
			}
		}
		blurImage[linear_index].x = pixelR / pixels;
		blurImage[linear_index].y = pixelG / pixels;
		blurImage[linear_index].z = pixelB / pixels;
		blurImage[linear_index].w = rgbaImage[linear_index].w;
	}
}

__global__ void image_blur_avg_shared_memory(
	uchar4* rgbaImage,
	uchar4* blurImage,
	int		numRows,
	int		numCols
) {
	__shared__ uchar4 pixelVal[18][18];
	/*
		xxxxx
		xpppx
		xpppx
		xpppx
		xxxxx
	*/
	int index_x = blockDim.x * blockIdx.x + threadIdx.x;
	int index_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (index_x < numCols && index_y < numRows) {
		int linear_index = index_y * numCols + index_x;
		// laod to shared memory
		
		// load itself --- correct
		pixelVal[threadIdx.y + 1][threadIdx.x+1] = rgbaImage[linear_index];
		// load the first row --- correct
		if (threadIdx.y == 0 && blockIdx.y != 0) {
			pixelVal[threadIdx.y][threadIdx.x + 1] = rgbaImage[linear_index - numCols];
			// load the most upper left --- correct
			if (threadIdx.x == 0) {
				pixelVal[threadIdx.y][threadIdx.x] = rgbaImage[linear_index - numCols - 1];
			}
			// load the most upper right --- correct
			if (threadIdx.x == blockDim.x - 1) {
				pixelVal[threadIdx.y][threadIdx.x + 2] = rgbaImage[linear_index - numCols + 1];
			}
		}
		// loda the last row
		if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1) {
			pixelVal[threadIdx.y + 2][threadIdx.x + 1] = rgbaImage[linear_index + numCols];
			// load the most lower left
			if (threadIdx.x == 0) {
				pixelVal[threadIdx.y + 2][threadIdx.x] = rgbaImage[linear_index + numCols - 1];
			}
			// load the moost lower right
			if (threadIdx.x == blockDim.x - 1) {
				pixelVal[threadIdx.y + 2][threadIdx.x + 2] = rgbaImage[linear_index + numCols + 1];
			}
		}
		// load the first col
		if (threadIdx.x == 0 && blockIdx.x != 0) {
			pixelVal[threadIdx.y + 1][threadIdx.x] = rgbaImage[linear_index - 1];
		}
		// load the last col
		if (threadIdx.x == blockDim.x -1  && blockIdx.x != gridDim.x - 1) {
			pixelVal[threadIdx.y + 1][threadIdx.x + 2] = rgbaImage[linear_index + 1];
		}
		
		__syncthreads();
		// start calculation
		float pixelR = 0;
		float pixelG = 0;
		float pixelB = 0;
		for (int blurRow = -1; blurRow < 2; blurRow++) {
			for (int blurCol = -1; blurCol < 2; blurCol++) {
				pixelR += pixelVal[threadIdx.y + blurRow + 1 ][threadIdx.x + blurCol + 1].x;
				pixelG += pixelVal[threadIdx.y + blurRow + 1 ][threadIdx.x + blurCol + 1].y;
				pixelB += pixelVal[threadIdx.y + blurRow + 1 ][threadIdx.x + blurCol + 1].z;
			}
		}
		
		blurImage[linear_index].x = pixelR / 9.0;
		blurImage[linear_index].y = pixelG / 9.0;
		blurImage[linear_index].z = pixelB / 9.0;
		blurImage[linear_index].w = pixelVal[threadIdx.y + 1][threadIdx.x + 1].w;

	}
}



using namespace cv;
using namespace std;

Mat inputImageRGBA;
Mat outputImageRGBA;

/*This Function is used to read image input*/
int readImageData(uchar4 **h_inputImageRGBA, unsigned char **h_outputImageRGBA, string &fileName, int *row, int *col) {

	Mat image = imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		fprintf(stderr, "Cant read image  %s \n", fileName);
		return -1;
	}
	// Store image color data to inputImageRGBA
	cvtColor(image, inputImageRGBA, CV_BGR2RGBA);
	// Allocate output image color data to outputImageRGBA
	outputImageRGBA.create(image.rows, image.cols, CV_8UC1);
	if (!inputImageRGBA.isContinuous() || !outputImageRGBA.isContinuous()) {
		fprintf(stderr, "Image is not  continus. \n");
		return -1;
	}
	*h_inputImageRGBA = (uchar4 *)inputImageRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = outputImageRGBA.ptr<unsigned char>(0);
	*row = image.rows;
	*col = image.cols;
	return 0;
}

int readImageData_GB(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA, string &fileName, int *row, int *col) {

	Mat image = imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		fprintf(stderr, "Cant read image  %s \n", fileName);
		return -1;
	}
	// Store image color data to inputImageRGBA 
	cvtColor(image, inputImageRGBA, CV_BGR2RGBA);
	// Allocate output image color data to outputImageRGBA
	outputImageRGBA.create(image.rows, image.cols, CV_8UC4);
	if (!inputImageRGBA.isContinuous() || !outputImageRGBA.isContinuous()) {
		fprintf(stderr, "Image is not  continus. \n");
		return -1;
	}
	*h_inputImageRGBA	=	(uchar4 *)inputImageRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA	=	(uchar4 *)outputImageRGBA.ptr<unsigned char>(0);
	*row = image.rows;
	*col = image.cols;
	return 0;
}


int CUDA_convert2GreyScale(string inputImage, string outputImage) {
	int row, col, pixel_count;
	uchar4 *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
	readImageData(&h_rgbaImage, &h_greyImage, inputImage, &row, &col);
	pixel_count = row * col;
	cudaMalloc(&d_rgbaImage, sizeof(uchar4) * pixel_count);
	cudaMalloc(&d_greyImage, sizeof(unsigned char) * pixel_count);
	cudaMemset(d_greyImage, 0, pixel_count * sizeof(unsigned char));
	cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * pixel_count, cudaMemcpyHostToDevice);
	dim3 blockSize(THREAD, THREAD, 1);
	dim3 gridSize(ceil(col / (float)THREAD), ceil(row / (float)THREAD), 1);
	
	rgba_to_greyscale <<<gridSize, blockSize >> >(d_rgbaImage, d_greyImage, row, col);
	cudaDeviceSynchronize();
	cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char)*pixel_count, cudaMemcpyDeviceToHost);
	cudaFree(d_rgbaImage);
	cudaFree(d_greyImage);
	Mat output(row, col, CV_8UC1, (void*)h_greyImage);
	imwrite(outputImage.c_str(), output);
	return 0;
}

int CUDA_imageBlur(string inputImage, string outputImage) {
	int row, col, pixel_count;
	uchar4 *h_rgbaImage, *d_rgbaImage;
	uchar4 *h_blurImage, *d_blurImage;
	readImageData_GB(&h_rgbaImage, &h_blurImage, inputImage, &row, &col);
	pixel_count = row * col;
	cudaMalloc(&d_rgbaImage, sizeof(uchar4) * pixel_count);
	cudaMalloc(&d_blurImage, sizeof(uchar4) * pixel_count);
	cudaMemset(d_blurImage, 0, pixel_count * sizeof(uchar4));
	cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * pixel_count, cudaMemcpyHostToDevice);
	dim3 blockSize(THREAD, THREAD, 1);
	dim3 gridSize(ceil(col / (float)THREAD), ceil(row / (float)THREAD), 1);
	image_blur_avg_shared_memory << <gridSize, blockSize >> >(d_rgbaImage, d_blurImage, row, col);
	checkCUDAError();
	cudaMemcpy(h_blurImage, d_blurImage, sizeof(uchar4)*pixel_count, cudaMemcpyDeviceToHost);
	cudaFree(d_rgbaImage);
	cudaFree(d_blurImage);
	Mat output(row, col, CV_8UC4, (void*)h_blurImage);
	cvtColor(output, output, CV_RGBA2BGR);
	imwrite(outputImage.c_str(), output);
	return 0;
}
