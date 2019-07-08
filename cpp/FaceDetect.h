#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

#define LANDMARKNUMBER 5

typedef void* FaceDetectDriver;

struct FaceBox
{
	float score;
	cv::Rect box;
	cv::Point landmarks[LANDMARKNUMBER];

	FaceBox()
		: score(0)
	{}

	FaceBox(const FaceBox& other)
	{
		if (&other != this)
		{
			this->score = other.score;
			this->box = other.box;
			memcpy(this->landmarks, other.landmarks, sizeof this->landmarks);
		}
	}

	FaceBox& operator=(const FaceBox& other)
	{
		if (&other != this)
		{
			this->score = other.score;
			this->box = other.box;
			memcpy(this->landmarks, other.landmarks, sizeof this->landmarks);
		}

		return *this;
	}
};

FaceDetectDriver LoadMTCNNModel();
void ReleaseMTCNNDriver(FaceDetectDriver driver);
bool DetectFace(FaceDetectDriver driver,
								cv::Mat& image,
								std::vector<FaceBox>& faces);