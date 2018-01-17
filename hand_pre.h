#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat hand_seg(Mat* src_u16)
{

	printf("rows=%d\n", (*src_u16).rows);
	Mat src_u8(240, 320, CV_8UC1, Scalar(255));
	Mat img(240, 320, CV_8UC1, Scalar(255));
	Mat bin_img(240, 320, CV_8UC1, Scalar(255));
	Mat mask, mask_img, rgb_img, roi_img;
	(*src_u16).convertTo(src_u8, CV_8UC1);		//ͼ��λ��ת��
	src_u8 = *src_u16;           //���ڵ���ֱ������8λͼ��
	//flip(src_u8, src_u8, 0);
	//��ֵ�˲�
	medianBlur(src_u8, img, 3);

	//����һά����
	int arr_img[76800];
	int arr[76800];
	int ind = 0;
	for (int i = 0; i < img.rows; i++)
	{
		//�õ���i�е��׵�ַ
		uchar* data = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (data[j] < 30)
				data[j] = 255;
			arr_img[ind] = data[j];
			ind = ind + 1;
		}
	}
	//������������ֵ����
	sort(arr_img, arr_img + 76800);
	//ȡǰ4000�������ƽ�������ֵ
	int sum = 0;
	for (int i = 0; i < 4000; i++)
		sum = sum + arr_img[i];
	int aver = sum / 3200;
	if (aver < 115)
		aver = 115;
	else
		aver = aver + 5;
	std::cout << "aver=" << aver << endl;
	////����и��˲�
	//for (int i = 0; i < img.rows; i++)
	//{
	//	//�õ���i�е��׵�ַ
	//	uchar* data = img.ptr<uchar>(i);
	//	{
	//	for (int j = 0; j < img.cols; j++)
	//		if (data[j]>aver)
	//			data[j] = 255;
	//	}
	//}
	//ͼ���ֵ��
	medianBlur(img, img, 3);
	threshold(img, bin_img, aver, 255, THRESH_BINARY);
	bitwise_not(bin_img, bin_img);
	//��ȡ��������ͨ���⼰����
	vector<vector<Point> > contours;
	vector<Point> max_cont;
	findContours(bin_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	double max_area = 0;
	int n = 0;
	for (int i = 0; i < int(contours.size()); i++)
	{
		if (contourArea(contours[i]) > max_area)
		{
			//cout << "if" << endl;
			max_area = contourArea(contours[i]);
			max_cont = contours[i];
			if (i > 0)
				drawContours(bin_img, contours, n, 0, -1);
			n = i;
		}
		else
		{
			//cout << "else" << endl;
			drawContours(bin_img, contours, i, 0, -1);
		}
	}
	if (max_area > 9000)      //ע����ֵ���趨
		drawContours(bin_img, contours, -1, 0, -1);
	//��ʴ������
	erode(bin_img, bin_img, Mat());
	dilate(bin_img, bin_img, Mat());
	cv::imshow("bin_src", bin_img);		//��ʾ��ֵ��ͼ��
	//������С��Ӿ���
	//vector<Rect> box;
	Rect box = boundingRect(Mat(max_cont));
	if (box.width > box.height)
	{
		rectangle(bin_img, Point(box.x + min(80,max(box.height, 70)), box.y), Point(320, 240), Scalar(0), -1);
		rectangle(bin_img, Point(box.x, box.y + min(80, max(box.height, 70))), Point(320, 240), Scalar(0), -1);
	}
	else
	{
		rectangle(bin_img, Point(box.x, box.y + min(80, max(box.width, 70))), Point(320, 240), Scalar(0), -1);
		rectangle(bin_img, Point(box.x + min(80, max(box.width, 70)), box.y), Point(320, 240), Scalar(0), -1);
	}
	cv::imshow("bin", bin_img);		//��ʾ��ֵ��ͼ��
	//����mask�������ֲ���ȡ
	bin_img.convertTo(mask, CV_8UC1, 1.0 / 255);
	mask_img = img.mul(mask);
	medianBlur(mask_img, mask_img, 3);
	//���������ͨ����
	vector<vector<Point> > contours_2;
	vector<Point> max_cont_2;
	findContours(bin_img, contours_2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	max_area = 0;
	n = 0;
	for (int i = 0; i < int(contours_2.size()); i++)
	{
		if (contourArea(contours_2[i]) > max_area)
		{
			//cout << "if" << endl;
			max_area = contourArea(contours_2[i]);
			max_cont_2 = contours_2[i];
			if (i > 0)
				drawContours(mask_img, contours_2, n, 0, -1);
			n = i;
		}
		else
		{
			//cout << "else" << endl;
			drawContours(mask_img, contours_2, i, 0, -1);
		}
	}
	//int roi_side = min(max(60, min(box.width, box.height)),100);
	//roi_img = mask_img(Rect(box.x, box.y, roi_side, roi_side));
	//resize(roi_img, roi_img, Size(80, 80));
	////����������������ԭͼ�Ϻ�ɫ��ע
	//vector<vector<Point>> fin_cont;
	//findContours(bin_img, fin_cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//cvtColor(img, rgb_img, COLOR_GRAY2BGR);
	//drawContours(rgb_img, fin_cont, -1, Scalar(0, 0, 255), -1);
	//cv::imshow("src", *src_u16);     //��ʾԭʼ16λͼ��
	//cv::imshow("src_u8", img);   //��ʾ8λ���ͼ
	//cv::imshow("bin", bin_img);		//��ʾ��ֵ��ͼ��
	cv::imshow("mask", mask_img);		//��ʾԭ���ͼmask
	//cv::imshow("rgb", rgb_img);		//��ʾ�ֲ�������ɫ���
	//cv::imshow("roi", roi_img);		//��ʾ�ֲ��и�ROI���ݼ�
	//imwrite("mask.png", mask);
	//waitKey(200);
	return mask_img;
}
