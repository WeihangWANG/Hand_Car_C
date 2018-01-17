#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <zmq.h>
//#include "test.h"
#include "hand_pre.h"
#include "HMHH.h"
#include "HON4D.h"
#include <time.h>
#include "dmcam.h"
using namespace std;
using namespace cv;

//Falseʵʱ,True����ģʽ
bool SIM = FALSE;//FALSE;		
//void hand_seg(string filename) 
//{
//	//Mat src_u16 = imread("save.png", -1);
//	std::cout << filename << endl;
//	Mat src_u16 = imread(filename, -1);
//	//namedWindow("test opencv", CV_WINDOW_AUTOSIZE);
//	printf("rows=%d\n",src_u16.rows);
//	Mat src_u8(240, 320, CV_8UC1, Scalar(255));
//	Mat img(240, 320, CV_8UC1, Scalar(255));
//	Mat bin_img(240, 320, CV_8UC1, Scalar(255));
//	Mat mask, mask_img, rgb_img, roi_img;
//	src_u16.convertTo(src_u8, CV_8UC1);		//ͼ��λ��ת��
//
//	//flip(src_u8, src_u8, 0);
//	//��ֵ�˲�
//	medianBlur(src_u8, img, 3);
//
//	//����һά����
//	int arr_img[76800];
//	int arr[76800];
//	int ind = 0;
//	for (int i = 0; i < img.rows; i++)
//	{
//		//�õ���i�е��׵�ַ
//		uchar* data = img.ptr<uchar>(i);
//		for (int j = 0; j < img.cols; j++)
//		{
//			if (data[j] < 30)
//				data[j] = 255;
//			arr_img[ind] = data[j];
//			ind = ind + 1;
//		}
//	}
//	//������������ֵ����
//	sort(arr_img, arr_img+76800);
//	//ȡǰ4000�������ƽ�������ֵ
//	int sum = 0;
//	for (int i = 0; i < 4000; i++)
//		sum = sum + arr_img[i];
//	int aver = sum / 3200;
//	if (aver < 115)
//		aver = 115;
//	else
//		aver = aver + 10;
//	std::cout << "aver=" << aver << endl;
//	////����и��˲�
//	//for (int i = 0; i < img.rows; i++)
//	//{
//	//	//�õ���i�е��׵�ַ
//	//	uchar* data = img.ptr<uchar>(i);
//	//	{
//	//	for (int j = 0; j < img.cols; j++)
//	//		if (data[j]>aver)
//	//			data[j] = 255;
//	//	}
//	//}
//	//ͼ���ֵ��
//	medianBlur(img, img, 3);
//	threshold(img, bin_img, aver, 255, THRESH_BINARY);
//	bitwise_not(bin_img, bin_img);
//	//��ȡ��������ͨ���⼰����
//	vector<vector<Point> > contours;
//	vector<Point> max_cont;
//	findContours(bin_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	
//	double max_area = 0;
//	int n = 0;
//	for (int i = 0; i < int(contours.size()); i++)
//	{
//		if(contourArea(contours[i]) > max_area)
//		{
//			//cout << "if" << endl;
//			max_area = contourArea(contours[i]);
//			max_cont = contours[i];
//			if(i > 0)
//				drawContours(bin_img, contours, n, 0, -1);
//			n = i;
//		}
//		else
//		{
//			//cout << "else" << endl;
//			drawContours(bin_img, contours, i, 0, -1);
//		}
//	}
//	if (max_area > 8000)
//		drawContours(bin_img, contours, -1, 0, -1);
//	//��ʴ������
//	erode(bin_img, bin_img, Mat());
//	dilate(bin_img, bin_img, Mat());
//	//������С��Ӿ���
//	//vector<Rect> box;
//	Rect box = boundingRect(Mat(max_cont));
//	if (box.width > box.height)
//		rectangle(bin_img, Point(box.x+max(box.height,60), box.y),Point(320,240), Scalar(0), -1);
//	else
//		rectangle(bin_img, Point(box.x, box.y+max(60, box.width)), Point(320, 240), Scalar(0), -1);
//	//����mask�������ֲ���ȡ
//	bin_img.convertTo(mask, CV_8UC1, 1.0 / 255);
//	mask_img = img.mul(mask);
//	medianBlur(mask_img, mask_img, 3);
//	int roi_side = max(60, min(box.width, box.height));
//	roi_img = mask_img(Rect(box.x, box.y, roi_side, roi_side));
//	resize(roi_img, roi_img, Size(80,80));
//	//����������������ԭͼ�Ϻ�ɫ��ע
//	vector<vector<Point>> fin_cont;
//	findContours(bin_img, fin_cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	cvtColor(img, rgb_img, COLOR_GRAY2BGR);
//	drawContours(rgb_img, fin_cont, -1, Scalar(0,0,255), -1);
//	cv::imshow("src", src_u16);     //��ʾԭʼ16λͼ��
//	cv::imshow("src_u8", img);   //��ʾ8λ���ͼ
//	cv::imshow("bin", bin_img);		//��ʾ��ֵ��ͼ��
//	//imshow("mask", mask_img);		//��ʾԭ���ͼmask
//	cv::imshow("rgb", rgb_img);		//��ʾ�ֲ�������ɫ���
//	cv::imshow("roi", roi_img);		//��ʾ�ֲ��и�ROI���ݼ�
//	imwrite("mask.png", mask);
//	waitKey(200);
//}

int main()
{
	double dur;
	clock_t start, end;		 //��������ʱ��
	std::cout << "Hand Recognition in Car Platform..." << endl;

	ofstream outfile;
	outfile.open("feature_HON4D.txt");		//д���ļ�����û�����½�

	/*****************���ڵ���*******************/
	/*
	Mat test = cv::imread("D:\\Documents\\Downloads\\data0110\\data0110\\nph\\src\\221.png",0);
	cv::imshow("src", test);
	Mat test_mask = hand_seg(&test);
	cv::imshow("test",test_mask);
	cv::waitKey(0);
	*/

	//����ģʽ
	if (SIM > 0)
	{
		std::cout << "sim mode NOW -_-";
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos("infofile.txt");//������ͼƬ���ļ����б�

		//���ʵ����
		//Hand *hand = new Hand();
		/*HMHH hand = HMHH();
		HON4D *cHON4D;
		cHON4D = new HON4D(40, 40, 320, 240, 30, "projector\\");*/
		//const int len = hand.get_feature_len();
		//const int len1 = cHON4D->get_feature_len();
		int cnt = 2424;
		for (int num = 0 ; num < 694 ; num++)
		{
			stringstream id;
			id << num;
			//string ImgName = "D:\\Documents\\PyCharm\\Hand_in_CAR\\data\\wwh\\src\\" + id.str() + ".png";
			string ImgName = "D:\\Documents\\Downloads\\data0110\\data0110\\whh\\src\\" + id.str() + ".png";
			cout << ImgName << endl;
			Mat src_img = cv::imread(ImgName, -1);		//��ȡͼƬ
//			cv::imshow("src", src_img);
			Mat roi_img = hand_seg(&src_img);		//�ֲ��ָ�
			//ͼƬ�����ļ���int2str
			stringstream ss;
			ss << cnt;
			string name = "./dataset/" + ss.str() + ".png";
			cv::imwrite(name, roi_img);
			cnt = cnt + 1;
			cv::waitKey(10);
		}
		//for (int num = 0; num < 611 && getline(finPos, ImgName) ; num++)
		//{
		//	//cout << "����" << ImgName << endl;
		//	ImgName = "D:\\Documents\\Downloads\\data0110\\data0110\\nph\\src\\" + ImgName;			//������·����
		//	cout << ImgName << endl;
		//	Mat src_img = cv::imread(ImgName,0);		//��ȡͼƬ
		//	cv::imshow("src", src_img);
		//	Mat roi_img = hand_seg(&src_img);		//�ֲ��ָ�
		//	//float* feature;
		//	//feature = cHON4D->get_feature(roi_img);
		//	////������������
		//	//outfile << cnt << " ";
		//	//for (int i = 0; i < len1; i++)
		//	//	if (i < len1 - 1)
		//	//		outfile << feature[i] << ",";
		//	//	else
		//	//		outfile << feature[i] << endl;
		//	//ͼƬ�����ļ���int2str
		//	stringstream ss;
		//	ss << cnt;
		//	string name = "./sav/" + ss.str() + ".png";
		//	//cout << name << endl;			//��ӡ�ļ���
		//	cv::imwrite(name, roi_img);
		//	//Mat img_u8;
		//	//img_u16.convertTo(img_u8, CV_8UC1);
		//	//cv::imshow("src", img_u8);     //��ʾ8λͼ��
		//	cv::waitKey(10);
		//	cnt = cnt + 1;
		//	
		//	//delete []feature;     //�ͷ��ڴ�
		//}
	}
	//ʵʱģʽ
	else
	{
		cout << "rt0" << endl;
		void* context = zmq_ctx_new();/// ����һ���µĻ���  
		assert(context != NULL);

		int ret = zmq_ctx_set(context, ZMQ_MAX_SOCKETS, 1);/// �û�����ֻ������һ��socket�Ĵ���  
		cout << "rt1" << endl;//cout << "2" << endl;
		assert(ret == 0);

		void* subscriber = zmq_socket(context, ZMQ_SUB);/// ����һ��������  
		assert(subscriber != NULL);
		cout << "rt2" << endl;
		//ret = zmq_connect(subscriber, "tcp://192.168.7.2:56789");/// ���ӵ�������  
		ret = zmq_connect(subscriber, "tcp://127.0.0.1:50660");/// ���ӵ������� 
		assert(ret == 0);
		cout << "rt3" << endl;
		ret = zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "DIST", 0);/// ������Ӹ�������Ϣ�˲���������ܲ�����Ϣ  
		assert(ret == 0);
		cout << "rt4" << endl;
		uchar buf[153612];/// ��Ϣ������ 
		uint16_t data[320 * 240];
		int cnt = 0;
		cout << "rt5" << endl;

		uint16_t dst[320][240];
		cout << zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT) << endl;
		while (zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT))
		{
			std::cout << zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT) << endl;
			//ret = /// ������Ϣ���Ƕ���ʽ
			//if (ret != -1)/// ��ӡ��Ϣ  
			//{
				uint16_t *ptr = (uint16_t *)(&buf[12]);		//ȥ��֡ͷ��ԭʼ����
				for (int i = 0; i<320 * 240; i++)
				{
					data[i] = (float)ptr[i];
					//cout << data[i];
				}
				for (int i = 0; i < 320; i++)	//column
				{
					for (int j = 0; j < 240; j++)	//row
					{
						dst[i][j] = data[320 * j + i];
						//std::cout << dst[j][i] << ',';
					}
					//std::cout << endl;
				}
				Mat img_u16;
				img_u16 = Mat(320, 240, CV_16UC1, &dst);
				cv::transpose(img_u16, img_u16);
				cv::flip(img_u16, img_u16, 1);		//�����ת90��
				cv::flip(img_u16, img_u16, 1);		//0�����·�ת��1��ˮƽ��ת��-1������ˮƽͬʱ��ת
				Mat img_src = img_u16.clone();
				(img_src).convertTo(img_src, CV_8UC1);		//ͼ��λ��ת��
				start = clock();
				Mat roi_img = hand_seg(&img_src);
				end = clock();
				std::cout << "preprocess time = " << (double)(end - start)/ CLOCKS_PER_SEC << endl;
				start = end;
				//���������ļ���
				//HMHH feature
				//float* fea = hand.get_feature(roi_img);
				//outfile << cnt << " ";
				//for (int i = 0; i < len; i++)
				//	if (i < len - 1)
				//		outfile << fea[i] << ",";
				//	else
				//		outfile << fea[i] << endl;
				//end = clock();
				//int res = hand.get_result(roi_img);
				//char cls[2];
				//_itoa(res, cls, 10);
				//Mat clr_img;
				//cv::applyColorMap(img_src, clr_img, COLORMAP_JET);
				//cv::putText(clr_img, cls, Point(50, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);//��ͼƬ��д����
				//cv::imshow("clr",clr_img);
				//std::cout << "frame result=" << res << endl;
				//std::cout << "HMHH time = " << (double)(end - start) / CLOCKS_PER_SEC << endl;
				//start = end;
				//////HON4D feature
				//float* feature;
				//feature = cHON4D->get_feature(roi_img);
				////������������
				//outfile << cnt << " ";
				//for (int i = 0; i < len1; i++)
				//	if (i < len1 - 1)
				//		outfile << feature[i] << ",";
				//	else
				//		outfile << feature[i] << endl;
				//end = clock();
				//cout << "HON4D time = " << (double)(end - start) / CLOCKS_PER_SEC << endl;
				//start = end;
				//ͼƬ�����ļ���int2str
				stringstream ss;
				ss << cnt;
				string name = "./sav/"+ ss.str() + ".png";
				string src_name = "./src/" + ss.str() + ".png";
				//cout << name << endl;			//��ӡ�ļ���
				//����ԭʼͼ���ֲ��ָ���ͼƬ
				//cv::imwrite(name, roi_img);
				//cv::imwrite(src_name, img_src);
				Mat img_u8;
				img_u16.convertTo(img_u8, CV_8UC1);
				cv::imshow("src", img_u8 * 20);     //��ʾ8λͼ��
				cv::waitKey(100);
				cnt = cnt + 1;
				//delete []fea;
				//delete []feature;     //�ͷ��ڴ�
			//}
			//std::cout << "finish single frame" << endl;
				memset(buf,0,sizeof(uchar)*153612);
				Sleep(1);
		}
	}	
	outfile.close();
}