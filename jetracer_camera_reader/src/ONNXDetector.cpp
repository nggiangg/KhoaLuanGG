#include "ONNXDetector.h"
#include <fstream>
#include <iostream>
#include <NvOnnxParser.h>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

ONNXDetector::ONNXDetector(const std::string& modelPath, float confThreshold)
    : mModelPath(modelPath)
    , mConfThreshold(confThreshold)
    , mInputH(300)
    , mInputW(300)
    , mInputC(3)
    , mCudaBuffer{nullptr, nullptr}
    , mCudaStream(nullptr) {
    
    // Tạo CUDA stream
    cudaError_t cudaErr = cudaStreamCreate(&mCudaStream);
    if (cudaErr != cudaSuccess) {
        std::cerr << "CUDA stream creation failed: " << cudaGetErrorString(cudaErr) << std::endl;
        return;
    }
    
    // Tải class names
    std::string labelsPath = mModelPath.substr(0, mModelPath.find_last_of("/")) + "/labels.txt";
    std::cout << "Looking for labels at: " << labelsPath << std::endl;
    loadClassNames(labelsPath);
    
    // Xây dựng engine
    if (!buildEngine()) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return;
    }
    
    // Cấp phát bộ nhớ cho buffers
    mInputSize = mInputH * mInputW * mInputC;
    mHostInputBuffer.resize(mInputSize);
    
    // Lấy kích thước output từ engine thay vì hard-code
    // Lấy binding dimensions cho output
    auto outputDims = mEngine->getBindingDimensions(1); // Index 1 là output
    std::cout << "Output dimensions: ";
    for (int i = 0; i < outputDims.nbDims; i++) {
        std::cout << outputDims.d[i] << " ";
    }
    std::cout << std::endl;
    
    // SSD MobileNet có 1 output với số lượng detections * 7 (image_id, label, confidence, xmin, ymin, xmax, ymax)
    mOutputSize = 100 * 7; // Mặc định
    
    // Nếu có thể lấy chính xác kích thước từ binding
    if (outputDims.nbDims > 0) {
        mOutputSize = 1;
        for (int i = 0; i < outputDims.nbDims; i++) {
            mOutputSize *= outputDims.d[i];
        }
    }
    
    std::cout << "Using output size: " << mOutputSize << std::endl;
    mHostOutputBuffer.resize(mOutputSize);
    
    // Cấp phát bộ nhớ CUDA
    cudaErr = cudaMalloc(&mCudaBuffer[0], mInputSize * sizeof(float));
    if (cudaErr != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input buffer: " << cudaGetErrorString(cudaErr) << std::endl;
        return;
    }
    
    cudaErr = cudaMalloc(&mCudaBuffer[1], mOutputSize * sizeof(float));
    if (cudaErr != cudaSuccess) {
        std::cerr << "CUDA malloc failed for output buffer: " << cudaGetErrorString(cudaErr) << std::endl;
        return;
    }
}

ONNXDetector::~ONNXDetector() {
    // Giải phóng bộ nhớ CUDA
    cudaFree(mCudaBuffer[0]);
    cudaFree(mCudaBuffer[1]);
    cudaStreamDestroy(mCudaStream);
}

bool ONNXDetector::buildEngine() {
    try {
        std::cout << "Building TensorRT engine from: " << mModelPath << std::endl;
        
        // Check if model file exists
        std::ifstream modelFile(mModelPath);
        if (!modelFile.good()) {
            std::cerr << "ONNX model file not found: " << mModelPath << std::endl;
            return false;
        }
        modelFile.close();
        
        // Tạo builder
        auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(
            nvinfer1::createInferBuilder(mLogger));
        if (!builder) {
            std::cerr << "Failed to create TensorRT builder" << std::endl;
            return false;
        }
        
        // Tạo network
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
            builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create TensorRT network" << std::endl;
            return false;
        }
        
        // Tạo config
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(
            builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create TensorRT builder config" << std::endl;
            return false;
        }
        
        // Tạo ONNX parser
        auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(
            nvonnxparser::createParser(*network, mLogger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        // Parse ONNX model
        std::cout << "Parsing ONNX model..." << std::endl;
        if (!parser->parseFromFile(mModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX model" << std::endl;
            return false;
        }
        
        // Get model information
        std::cout << "Network inputs: " << network->getNbInputs() << std::endl;
        std::cout << "Network outputs: " << network->getNbOutputs() << std::endl;
        
        for (int i = 0; i < network->getNbInputs(); i++) {
            auto input = network->getInput(i);
            auto dims = input->getDimensions();
            std::cout << "Input " << i << " name: " << input->getName() << ", dims: ";
            for (int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << " ";
            }
            std::cout << std::endl;
            
            // Update our input dimensions if available from the model
            if (dims.nbDims == 4) {  // NCHW format
                mInputC = dims.d[1];
                mInputH = dims.d[2]; 
                mInputW = dims.d[3];
                std::cout << "Using input dimensions from model: " << mInputC << "x" << mInputH << "x" << mInputW << std::endl;
            }
        }
        
        for (int i = 0; i < network->getNbOutputs(); i++) {
            auto output = network->getOutput(i);
            auto dims = output->getDimensions();
            std::cout << "Output " << i << " name: " << output->getName() << ", dims: ";
            for (int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Increase workspace size for better performance
        config->setMaxWorkspaceSize(1 << 20); // 1 MiB - extremely reduced for Jetson Nano
        
        // For Jetson Nano, use the most conservative settings
        std::cout << "Using default precision for maximum compatibility" << std::endl;
        
        // Don't use FP16 or INT8, default precision works better on Nano for some models
        // config->setFlag(nvinfer1::BuilderFlag::kFP16);
        
        // Set device properties and log information
        int device = 0;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        std::cout << "CUDA device: " << prop.name << " with " << (prop.totalGlobalMem / (1024*1024)) 
                 << "MB memory" << std::endl;
        
        // Log device compute capability
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Build engine trực tiếp (TensorRT < 8)
        std::cout << "Building engine..." << std::endl;
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), InferDeleter());
        if (!mEngine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        } 
        
        std::cout << "Engine built successfully" << std::endl;
        
        // Tạo execution context
        mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
            mEngine->createExecutionContext(), InferDeleter());
        if (!mContext) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Print engine details
        std::cout << "Engine created with " << mEngine->getNbBindings() << " bindings" << std::endl;
        for (int i = 0; i < mEngine->getNbBindings(); i++) {
            std::cout << "Binding " << i << ": " 
                    << (mEngine->bindingIsInput(i) ? "Input" : "Output") 
                    << " name: " << mEngine->getBindingName(i) << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in buildEngine: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool ONNXDetector::preprocessImage(const cv::Mat& frame) {
    try {
        static int skipped_frames = 0;
        cudaError_t cudaErr; // Declare here for later use
        // Additional safety - don't synchronize CUDA too often
        if (skipped_frames++ % 3 != 0) {
            // Skip CUDA sync on most frames
        } else {
            cudaGetLastError(); // Just clear errors
        }
        
        static int log_count = 0;
        bool log_this_frame = (log_count++ % 100 == 0); // Log much less frequently
        
        // First convert to a smaller image to reduce memory usage
        cv::Mat smaller;
        if (frame.cols > 320 || frame.rows > 320) {
            double scale = std::min(320.0 / frame.cols, 320.0 / frame.rows);
            cv::resize(frame, smaller, cv::Size(), scale, scale);
        } else {
            smaller = frame;
        }
        
        // Now resize to exact model input size
        cv::Mat resized;
        cv::resize(smaller, resized, cv::Size(mInputW, mInputH));
        
        // Print debugging info very occasionally
        if (log_this_frame) {
            std::cout << "Original image: " << frame.size() << " channels: " << frame.channels() 
                    << " type: " << frame.type() << std::endl;
            std::cout << "Resized image: " << resized.size() << " channels: " << resized.channels() 
                    << " type: " << resized.type() << std::endl;
        }
        
        // Chuyển đổi từ BGR sang RGB và normalize
        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32FC3);
        
        // Normalize to range [0,1] or [-1,1] depending on model requirements
        // SSD MobileNet typically expects [0,1] or [-1,1]
        if (true) { // Normalize to [-1,1] - common for models trained with TF
            floatImg = (floatImg / 127.5) - 1.0;
        } else { // Normalize to [0,1] - common for models trained with PyTorch
            floatImg = floatImg / 255.0;
        }
        
        // Reset host buffer
        std::fill(mHostInputBuffer.begin(), mHostInputBuffer.end(), 0.0f);
        
        // Check if we need NCHW (common in TensorRT) or NHWC format
        bool useNCHW = true; // Most TensorRT models use NCHW
        
        if (useNCHW) {
            // Convert to NCHW format (Num batches=1, Channels, Height, Width)
            std::vector<cv::Mat> channels;
            cv::split(floatImg, channels);
            
            size_t channelSize = mInputH * mInputW;
            for (int c = 0; c < mInputC; c++) {
                std::memcpy(mHostInputBuffer.data() + c * channelSize, 
                        channels[c].data, channelSize * sizeof(float));
            }
        } else {
            // Use NHWC format (Num batches=1, Height, Width, Channels)
            // This is default format from OpenCV
            std::memcpy(mHostInputBuffer.data(), floatImg.data, mInputSize * sizeof(float));
        }
        
        // Copy từ host sang device
        cudaErr = cudaMemcpyAsync(mCudaBuffer[0], mHostInputBuffer.data(), 
                        mInputSize * sizeof(float), cudaMemcpyHostToDevice, mCudaStream);
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA memory copy to device failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        
        // Ensure the memory transfer completes
        cudaErr = cudaStreamSynchronize(mCudaStream);
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA stream synchronize failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in preprocessing: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error in preprocessing: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

std::vector<Detection> ONNXDetector::postprocessResults() {
    std::vector<Detection> detections;
    
    // SSD MobileNet output format là [image_id, label, confidence, xmin, ymin, xmax, ymax]
    // cho mỗi detection
    for (size_t i = 0; i < mOutputSize; i += 7) {
        float confidence = mHostOutputBuffer[i+2];
        
        // Lọc theo ngưỡng confidence
        if (confidence > mConfThreshold) {
            Detection det;
            det.classId = static_cast<int>(mHostOutputBuffer[i+1]);
            det.confidence = confidence;
            
            // Tọa độ normalized (0-1)
            det.xmin = mHostOutputBuffer[i+3];
            det.ymin = mHostOutputBuffer[i+4];
            det.xmax = mHostOutputBuffer[i+5];
            det.ymax = mHostOutputBuffer[i+6];
            
            // Thêm tên class
            if (det.classId < mClassNames.size()) {
                det.className = mClassNames[det.classId];
            } else {
                det.className = "Unknown";
            }
            
            detections.push_back(det);
        }
    }
    
    return detections;
}

std::vector<Detection> ONNXDetector::detect(const cv::Mat& frame) {
    static int error_count = 0;
    static int frame_count = 0;
    static bool had_error = false;
    frame_count++;
    cudaError_t cudaErr; // Declare here for later use
    
    // Always skip frames for better performance on the Jetson Nano
    if (frame_count % 8 != 0) {  // Only process 1 in 8 frames (more aggressive skipping)
        return {};  // Skip this frame
    }
    
    try {
        // Skip more frames if we're having persistent errors
        if (had_error && frame_count % 10 != 0) {
            return {};  // Skip more frames during error recovery
        }
        had_error = false;
        
        // Simple error state clearing without synchronization
        cudaError_t lastError = cudaGetLastError();
        if (lastError != cudaSuccess) {
            error_count++;
            had_error = true;
            if (error_count % 20 == 0) { // Log even less frequently
                std::cerr << "CUDA state cleared: " << cudaGetErrorString(lastError) << std::endl;
            }
        }
        
        // More aggressive frame skipping during errors
        if (error_count > 20) {  // Lower threshold
            if (frame_count % 60 == 0) { // Process only 1 in 60 frames during heavy error states
                error_count -= 5; // Reduce error count for recovery
            } else {
                return {}; // Skip frames
            }
        }
        
        // Ensure model is loaded
        if (!mContext || !mEngine) {
            std::cerr << "Model not properly loaded" << std::endl;
            return {};
        }
        
        // Tiền xử lý hình ảnh
        if (!preprocessImage(frame)) {
            std::cerr << "Image preprocessing failed" << std::endl;
            return {};
        }
        
        // Thiết lập input và output
        void* bindings[] = {mCudaBuffer[0], mCudaBuffer[1]};
        
        // Kiểm tra bindings
        if (!mCudaBuffer[0] || !mCudaBuffer[1]) {
            std::cerr << "CUDA buffers not properly allocated" << std::endl;
            return {};
        }
        
        // Don't synchronize CUDA here, it slows things down
        // cudaDeviceSynchronize();
        
        // Thực hiện inference
        bool status = false;
        try {
            // Make sure context is valid
            if (!mContext) {
                std::cerr << "Context is invalid" << std::endl;
                return {};
            }
            
            // Log memory status only occasionally
            static int mem_log_counter = 0;
            if (mem_log_counter++ % 100 == 0) {
                size_t free_memory, total_memory;
                cudaMemGetInfo(&free_memory, &total_memory);
                std::cout << "CUDA memory: " << (free_memory/1024/1024) << "MB free" << std::endl;
            }
            
            // Execute inference
            status = mContext->executeV2(bindings);
        } catch (const std::exception& e) {
            // Don't log every exception
            static int error_log_counter = 0;
            if (error_log_counter++ % 5 == 0) {
                std::cerr << "Inference error" << std::endl;
            }
            return {};
        }
        
        if (!status) {
            std::cerr << "TensorRT inference failed" << std::endl;
            return {};
        }
        
        // Copy kết quả từ device về host
        cudaErr = cudaMemcpyAsync(mHostOutputBuffer.data(), mCudaBuffer[1], 
                            mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mCudaStream);
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA memory copy to host failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return {};
        }
        
        // Đồng bộ hóa
        cudaErr = cudaStreamSynchronize(mCudaStream);
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA stream synchronize failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return {};
        }
        
        // Xử lý kết quả
        return postprocessResults();
    } catch (const std::exception& e) {
        std::cerr << "Exception in detect function: " << e.what() << std::endl;
        return {};
    }
}

void ONNXDetector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Chuyển từ normalized coordinates (0-1) sang pixel coordinates
        int xmin = static_cast<int>(det.xmin * frame.cols);
        int ymin = static_cast<int>(det.ymin * frame.rows);
        int xmax = static_cast<int>(det.xmax * frame.cols);
        int ymax = static_cast<int>(det.ymax * frame.rows);
        
        // Vẽ bounding box
        cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);
        
        // Tạo nhãn
        std::string label = det.className + " " + std::to_string(det.confidence).substr(0, 4);
        
        // Vẽ nhãn
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(xmin, ymin - textSize.height - 5),
                     cv::Point(xmin + textSize.width, ymin),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(xmin, ymin - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void ONNXDetector::loadClassNames(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        mClassNames.push_back(line);
    }
}

