#ifndef ONNX_DETECTOR_H
#define ONNX_DETECTOR_H

#include <string>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

// Định nghĩa cấu trúc cho kết quả phát hiện
struct Detection {
    int classId;
    float confidence;
    float xmin, ymin, xmax, ymax;
    std::string className;
};

// Logger được sử dụng bởi TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class ONNXDetector {
public:
    ONNXDetector(const std::string& modelPath, float confThreshold = 0.5f);
    ~ONNXDetector();

    // Phát hiện đối tượng từ frame OpenCV
    std::vector<Detection> detect(const cv::Mat& frame);

    // Vẽ kết quả lên frame
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

private:
    // Cấu trúc giải phóng bộ nhớ TRT
    struct InferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) obj->destroy();
        }
    };

    bool buildEngine();
    bool preprocessImage(const cv::Mat& frame);
    std::vector<Detection> postprocessResults();

    Logger mLogger;
    std::string mModelPath;
    float mConfThreshold;
    int mInputH, mInputW, mInputC;
    std::vector<std::string> mClassNames;

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;

    void* mCudaBuffer[2];  // Input và output buffers
    std::vector<float> mHostInputBuffer;
    std::vector<float> mHostOutputBuffer;
    size_t mInputSize;
    size_t mOutputSize;
    cudaStream_t mCudaStream;

    // Tải các class names
    void loadClassNames(const std::string& filename);
};

#endif // ONNX_DETECTOR_H
