/* Copyright 2019 Seedland Inc.
 * Author: Zeng Xianliang
 * Updated by: Sherk
 * This is originally derived for face detection using caffe model,
 *   then modified by Sherk to implement the torch model.
 */

// #include "mtcnn.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include "MTCNN.h"
#include "FaceDetect.h"
#include "torchutils.h"

// Using Caffe mtcnn model 
// FaceDetectDriver LoadMTCNNModel(const std::string& pnet_file,
// 																 const std::string& rnet_file,
// 																 const std::string& onet_file)
// {
// 	mtcnn* context = new mtcnn(pnet_file, rnet_file, onet_file);
// 	return context;
// }

// Using Torch mtcnn model
FaceDetectDriver LoadMTCNNModel()
{
	std::string pnet_weight_path = std::string(MODEL_PATH) + "pnet.pt";
	std::string rnet_weight_path = std::string(MODEL_PATH) + "rnet.pt";
	std::string onet_weight_path = std::string(MODEL_PATH) + "onet.pt";

	TAlgParam alg_param;
	alg_param.min_face = 40;
	alg_param.scale_factor = 0.79;
	alg_param.cls_thre[0] = 0.7;
	alg_param.cls_thre[1] = 0.8;
	alg_param.cls_thre[2] = 0.9;


	TModelParam modelParam;
	modelParam.alg_param = alg_param;
	modelParam.model_path = {pnet_weight_path,
													 rnet_weight_path,
													 onet_weight_path};
	modelParam.mean_value = {{127.5, 127.5, 127.5},
														{127.5, 127.5, 127.5},
														{127.5, 127.5, 127.5}};
	modelParam.scale_factor = {1.0f, 1.0f, 1.0f};
	modelParam.gpu_id = 0;
	modelParam.device_type = torch::DeviceType::CUDA;

	MTCNN* context = new MTCNN;
	context->InitDetector(&modelParam);
	return context;
}


void ReleaseMTCNNDriver(FaceDetectDriver driver)
{
	MTCNN* t = (MTCNN*)driver;
	delete t;
}

bool DetectFace(
	FaceDetectDriver driver,
	cv::Mat& image,
	std::vector<FaceBox>& faces)
{
	faces.clear();
	MTCNN* cnn = (MTCNN*)driver;
	std::vector<cv::Rect> outFaces;

	if (cnn)
	{
		// cnn->findFace(image, min_face);
		cnn->detect_face(image, outFaces);

		for (size_t i = 0; i < outFaces.size(); ++i)
		{
			auto it = outFaces.begin() + i;
			if (1)
			{
				FaceBox face;
				face.box.x = it->x;  // it->x;
				face.box.y = it->y;  // it->y;
				face.box.width = it->width;  // it->y2 - it->y1 + 1;
				face.box.height = it->height;  // it->x2 - it->x1 + 1;
				for (int k = 0; k < LANDMARKNUMBER; ++k)
				{
					face.landmarks[k].x = 0;  // (*it).ppoint[k];
					face.landmarks[k].y = 0;  // (*it).ppoint[k + LANDMARKNUMBER];
				}  // left eye, right eye, nose, mouth left, mouth right

				face.score = 0.99;  // it->score;

				// add face to the end of the vector faces
				faces.push_back(face);
			}
		}

		return true;
	}

	return false;
}