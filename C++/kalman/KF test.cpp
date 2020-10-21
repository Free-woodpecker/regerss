// KF test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#include "Kalman.h"
#include <iostream>



int main() {
	Kalman kalman(1e-6, 0.01);


	int i = 100;
	while (i--) {
		float x = (rand() % 256 - 128) * 1.0f;
		std::cout << "A : " << x << std::endl;
		std::cout << "B : " << kalman.Update(x) << std::endl;
		std::cout << std::endl;
	}


	std::cout << "Hello World!\n";
}