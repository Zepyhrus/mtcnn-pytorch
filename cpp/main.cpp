#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <spdhelper.hpp>
#include <opencv2/opencv.hpp>
#include <BTimer.hpp>
#include "MTCNN.h"
#include <boost/filesystem.hpp>
#include <unistd.h>
#include "FaceDetect.h"


int main(int argc, char* argv[])
{
    ENTER_FUNC;
    
    std::string img_path = std::string(MODEL_PATH) + "/../img/faces2.jpg";
    cv::Mat image = cv::imread(img_path);
    std::vector<FaceBox> faces;
    FaceDetectDriver m_faceDetector;
    int m_minFace = 40;
    
    LOGI("Here we go...");

    DetectFace(m_faceDetector, image, faces);

    LOGI("Nice to meet you...");

    LEAVE_FUNC;

    return 0;
}
