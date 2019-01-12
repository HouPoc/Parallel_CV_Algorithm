#include "imageProcess.h"


int main(void)
{
	//CUDA_convert2GreyScale("tm.jpg", "tm_grey.jpg");
	CUDA_imageBlur("tm.jpg", "tm_blur.jpg");
	system("pause");
	return 0;
}
