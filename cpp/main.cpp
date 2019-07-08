#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <spdhelper.hpp>
#include <opencv2/opencv.hpp>
#include <BTimer.hpp>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include "FaceDetect.h"
#include "MTCNN.h"


int main(int argc, char* argv[])
{
    ENTER_FUNC;
    BTimer timer;
    
    std::string img_path = std::string(MODEL_PATH) + "/../img/faces2.jpg";
    cv::Mat image = cv::imread(img_path);
    std::vector<FaceBox> faces;
    FaceDetectDriver m_faceDetector = LoadMTCNNModel();

    if (DetectFace(m_faceDetector, image, faces))
    {
        LOGI("Function called as expected.");
    }

    // cv::imshow("_", image);
    // cv::waitKey(0);

    LOGI("Warm up...");
    timer.reset();
    for (int i = 0; i < 5; ++i)
    {
        DetectFace(m_faceDetector, image, faces);
    }
    LOGI("Warm up over, time cost: {}", timer.elapsed());

    cv::VideoCapture cap;
    if (!cap.open(0))
    {
        LOGI("Failed to open camera.");
        return -1;
    }

    cv::Mat frame;
    while (1)
    {
        timer.reset();

        cap >> frame;
        if (frame.empty()) break;

        DetectFace(m_faceDetector, frame, faces);

        for (auto& i : faces)
        {
            cv::rectangle(frame, i.box, {0, 0, 255}, 2);
        }

        cv::imshow("_", frame);
        cv::waitKey(1);

        LOGI("Detect time cost: {}", timer.elapsed());
    }

    LEAVE_FUNC;

    ReleaseMTCNNDriver(m_faceDetector);

    return 0;
}
