#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;

class Hand
{
public:
	Mat img;
	int getLength(Mat img);
	float getFeature(Mat img);
	float getResult(Mat img);
};

int Hand::getLength(Mat img)
{
	return 10;
}

float Hand::getFeature(Mat img)
{
	return ;
}

float Hand::getResult(Mat img)
{
	;
}

void test()
{
	cout << "finfish include .h file" << endl;
}
