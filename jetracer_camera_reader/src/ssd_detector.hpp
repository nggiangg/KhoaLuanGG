#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <string>
#include <iostream>

// Add TensorRT Logger implementation
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
        // Print message based on severity
        switch(severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cout << "WARNING: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cout << "INFO: " << msg << std::endl;
                break;
            default:
                break;
        }
    }
};

struct Object {
    cv::Rect rect;
    float prob;
    int label;
};

class SSDDetector {
private:
    // Add logger as a member
    Logger logger;

public:
    SSDDetector(const std::string& model_path, const std::string& labels_file);
    ~SSDDetector();
    void CopyFromMat(const cv::Mat& image);
    void Infer();
    void PostProcess(std::vector<Object>& objs, float conf_thres = 0.5f);
    void DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs);
    bool buildEngineFromOnnx(const std::string& onnx_file);

private:
    void loadLabels(const std::string& labels_file);

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    void* d_input = nullptr;
    void* d_output = nullptr;
    float* h_output = nullptr;

    int input_w = 300, input_h = 300, input_c = 3;
    int output_size = 100 * 7; // default for SSD MobileNet
    int img_w = 0, img_h = 0;

    std::vector<std::string> class_names;
};