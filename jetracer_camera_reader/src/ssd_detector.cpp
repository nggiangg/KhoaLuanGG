#include "ssd_detector.hpp"
#include <fstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cassert>

SSDDetector::SSDDetector(const std::string& model_path, const std::string& labels_file) {
    loadLabels(labels_file);
    runtime = nvinfer1::createInferRuntime(logger);
    
    // Check if the model is an ONNX file or a TensorRT engine file
    if (model_path.find(".onnx") != std::string::npos) {
        // It's an ONNX model, build the engine from it
        if (!buildEngineFromOnnx(model_path)) {
            std::cerr << "Failed to build TensorRT engine from ONNX file" << std::endl;
            return;
        }
    } else {
        // Assume it's a TensorRT engine file
        std::ifstream file(model_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Engine file not found: " << model_path << std::endl;
            return;
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    }
    
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    auto input_dims = engine->getBindingDimensions(0);
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    auto output_dims = engine->getBindingDimensions(1);
    output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) output_size *= output_dims.d[i];
    cudaMalloc(&d_input, input_c * input_h * input_w * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    h_output = new float[output_size];
}

SSDDetector::~SSDDetector() {
    // Check and free CUDA memory safely
    if (d_input) {
        cudaError_t status = cudaFree(d_input);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error freeing d_input: " << cudaGetErrorString(status) << std::endl;
        }
        d_input = nullptr;
    }
    
    if (d_output) {
        cudaError_t status = cudaFree(d_output);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error freeing d_output: " << cudaGetErrorString(status) << std::endl;
        }
        d_output = nullptr;
    }
    
    // Free host memory
    if (h_output) {
        delete[] h_output;
        h_output = nullptr;
    }
    
    // Destroy TensorRT objects with checks
    if (context) {
        context->destroy();
        context = nullptr;
    }
    
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    
    // Destroy CUDA stream
    if (stream) {
        cudaError_t status = cudaStreamDestroy(stream);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error destroying stream: " << cudaGetErrorString(status) << std::endl;
        }
        stream = nullptr;
    }
    
    // Force CUDA to clean up to avoid memory leaks
    cudaDeviceSynchronize();
}

void SSDDetector::loadLabels(const std::string& labels_file) {
    std::ifstream file(labels_file);
    std::string line;
    while (std::getline(file, line)) class_names.push_back(line);
}

void SSDDetector::CopyFromMat(const cv::Mat& image) {
    try {
        img_w = image.cols;
        img_h = image.rows;
        
        // Make sure the image isn't empty
        if (image.empty() || img_w == 0 || img_h == 0) {
            throw std::runtime_error("Empty or invalid image");
        }
        
        // Prepare the input: resize -> convert to float -> normalize -> reorder
        cv::Mat resized, floatImg;
        cv::resize(image, resized, cv::Size(input_w, input_h));
        resized.convertTo(floatImg, CV_32FC3);
        
        // Normalize with mean=127.5, std=127.5
        floatImg = (floatImg / 127.5) - 1.0;
        
        // Split channels and prepare for transfer
        std::vector<cv::Mat> channels;
        cv::split(floatImg, channels);
        std::vector<float> input_data(input_c * input_h * input_w);
        
        // Copy each channel (CHW format for TensorRT)
        for (int c = 0; c < input_c; ++c) {
            memcpy(input_data.data() + c * input_h * input_w, 
                  channels[c].data, input_h * input_w * sizeof(float));
        }
        
        // Copy to GPU with error checking
        cudaError_t status = cudaMemcpyAsync(d_input, input_data.data(), 
                                           input_c * input_h * input_w * sizeof(float), 
                                           cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy failed: ") + 
                                   cudaGetErrorString(status));
        }
        
        // Synchronize to ensure the data is copied before inference
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA stream sync failed: ") + 
                                   cudaGetErrorString(status));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in CopyFromMat: " << e.what() << std::endl;
        throw;
    }
}

void SSDDetector::Infer() {
    try {
        // Setup the I/O bindings
        void* bindings[] = {d_input, d_output};
        
        // Execute inference with error checking
        bool status = context->enqueueV2(bindings, stream, nullptr);
        if (!status) {
            throw std::runtime_error("TensorRT inference failed");
        }
        
        // Copy results from device to host with error checking
        cudaError_t cudaStatus = cudaMemcpyAsync(h_output, d_output, 
                                               output_size * sizeof(float), 
                                               cudaMemcpyDeviceToHost, stream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy results failed: ") + 
                                   cudaGetErrorString(cudaStatus));
        }
        
        // Synchronize to ensure the results are ready
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA stream sync failed: ") + 
                                   cudaGetErrorString(cudaStatus));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in Infer: " << e.what() << std::endl;
        throw;
    }
}

void SSDDetector::PostProcess(std::vector<Object>& objs, float conf_thres) {
    objs.clear();
    for (int i = 0; i < output_size; i += 7) {
        float conf = h_output[i+2];
        if (conf < conf_thres) continue;
        Object obj;
        obj.label = static_cast<int>(h_output[i+1]);
        obj.prob = conf;
        float xmin = h_output[i+3], ymin = h_output[i+4], xmax = h_output[i+5], ymax = h_output[i+6];
        obj.rect = cv::Rect(
            int(xmin * img_w),
            int(ymin * img_h),
            int((xmax - xmin) * img_w),
            int((ymax - ymin) * img_h)
        );
        objs.push_back(obj);
    }
}

void SSDDetector::DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs) {
    for (const auto& obj : objs) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(0, 255, 0), 2);
        std::string label = class_names.size() > obj.label ? class_names[obj.label] : "Unknown";
        label += cv::format(" %.2f", obj.prob);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x, y = obj.rect.y - label_size.height - 5;
        if (y < 0) y = 0;
        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(0,255,0), cv::FILLED);
        cv::putText(bgr, label, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
}

bool SSDDetector::buildEngineFromOnnx(const std::string& onnx_file) {
    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // Parse ONNX model
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_file << std::endl;
        return false;
    }

    // Create optimization profile for Jetson Nano
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // Increase workspace size - 64MB instead of 1MB
    config->setMaxWorkspaceSize(64 << 20);  
    
    // Enable FP16 precision (crucial for Jetson Nano performance)
    if (builder->platformHasFastFp16()) {
        std::cout << "Enabling FP16 mode for faster inference" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // Build the engine - set DLA if available (for Jetson Xavier series)
    builder->setMaxBatchSize(1);  // Set batch size to 1 for real-time inference
    engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }

    // Clean up
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    
    std::cout << "TensorRT engine built successfully from ONNX model" << std::endl;
    return true;
}