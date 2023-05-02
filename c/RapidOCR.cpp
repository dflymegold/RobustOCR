#include <cstdio>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "version.h"
#include "OcrLite.h"
#include "OcrUtils.h"


int main() {
    std::cout << "INFO" << std::endl;
    std::string pathToVideo = "./uk.png";
    std::string modelDetPath, modelClsPath, modelRecPath, keysPath;
    int numThread = 8;
    int padding = 50;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.5f;
    float boxThresh = 0.3f;
    float unClipRatio = 1.6f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = true;
    int flagMostAngle = 1;
    int flagGpu = 0;
    int opt;
    int optionIndex = 0;
    cv::Mat frame;
    cv::VideoCapture cap;
    OcrLite ocrLite;

    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
        true,//isOutputConsole
        false,//isOutputPartImg
        false);//isOutputResultImg
    ocrLite.setGpuIndex(flagGpu);
    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
        "numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d),GPU(%d)\n",
        numThread, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle,
        flagGpu);
    modelClsPath = "./models/ch_ppocr_mobile_v2.0_cls_infer.meta.onnx";
    modelDetPath = "./models/cyrilic_det.onnx";
    modelRecPath = "./models/english_rec.onnx";
    keysPath = "./models/uk_dict.txt";
    ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);
    
   //cap.open(pathToVideo);

    std::ofstream out("output.txt");
    frame = cv::imread(pathToVideo);
    //while (true) {
    //    cap.read(frame);
    //    if (frame.empty()) {
    //        break;
    //    }
        //float stamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000;
    //    std::cout << stamp << std::endl;
    
    OcrResult result = ocrLite.detect(frame, padding, maxSideLen,
    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    std::cout << result.strRes << std::endl;
    out << "{" + result.strRes + "} \r";
    //}
    out.close();
    return 0;
}