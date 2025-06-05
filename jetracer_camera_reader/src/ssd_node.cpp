#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cuda_runtime_api.h>
#include "ssd_detector.hpp"

class SSDObjectDetector {
private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher image_pub_;
    bool use_compressed_;
    std::unique_ptr<SSDDetector> detector_;
    int frame_count_;

public:
    SSDObjectDetector() : use_compressed_(true), frame_count_(0) {
        // Lấy parameters từ ROS parameter server
        nh_.param<bool>("use_compressed", use_compressed_, true);
        float confidence_threshold;
        nh_.param<float>("confidence_threshold", confidence_threshold, 0.5f);
        
        // Đọc đường dẫn mô hình từ parameter
        std::string model_path;
        // Update default path to use .onnx extension
        nh_.param<std::string>("model_path", model_path, 
                              "/home/jetson/jetson-inference/python/training/detection/ssd/models/ViDu/mb2-ssd-lite.onnx");
        
        std::string labels_path;
        nh_.param<std::string>("labels_path", labels_path, 
                              "/home/jetson/jetson-inference/python/training/detection/ssd/models/ViDu/labels.txt");
        
        ROS_INFO("Using model path: %s", model_path.c_str());
        ROS_INFO("Using labels path: %s", labels_path.c_str());
        ROS_INFO("Using confidence threshold: %.2f", confidence_threshold);
        
        // Kiểm tra xem file model và labels có tồn tại không
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            ROS_ERROR("Model file not found: %s", model_path.c_str());
            return;
        }
        model_file.close();
        
        std::ifstream labels_file(labels_path);
        if (!labels_file.good()) {
            ROS_ERROR("Labels file not found: %s", labels_path.c_str());
            return;
        }
        labels_file.close();
        
        // Khởi tạo SSD detector
        try {
            detector_ = std::make_unique<SSDDetector>(model_path, labels_path);
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to initialize detector: %s", e.what());
            return;
        }
        
        // Subscribe to camera feed and advertise output
        std::string camera_topic;
        nh_.param<std::string>("camera_topic", camera_topic, "/csi_cam_0/image_raw");
        
        if (use_compressed_) {
            std::string compressed_topic = camera_topic + "/compressed";
            ROS_INFO("Subscribing to compressed topic: %s", compressed_topic.c_str());
            image_sub_ = nh_.subscribe(compressed_topic, 1, 
                                      &SSDObjectDetector::compressedImageCallback, this);
        } else {
            ROS_INFO("Subscribing to raw topic: %s", camera_topic.c_str());
            image_sub_ = nh_.subscribe(camera_topic, 1, 
                                      &SSDObjectDetector::imageCallback, this);
        }
        
        image_pub_ = nh_.advertise<sensor_msgs::Image>("/ssd_detected_objects", 1);
        
        ROS_INFO("SSD object detector initialized. Waiting for images...");
    }
    
    void compressedImageCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        try {
            // Tăng frame counter để bỏ qua frame nếu cần
            frame_count_++;
            if (frame_count_ % 10 != 0) { // Chỉ xử lý 1/10 số frame để giảm tải
                return;
            }
            
            // Giải nén ảnh
            cv::Mat image_np = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
            if (image_np.empty()) {
                ROS_WARN_THROTTLE(10.0, "Failed to decode compressed image");
                return;
            }
            
            // Xử lý ảnh
            processImage(image_np, msg->header);
        } catch (cv::Exception& e) {
            ROS_ERROR_THROTTLE(10.0, "OpenCV exception");
        }
    }
    
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // Tăng frame counter để bỏ qua frame nếu cần
            frame_count_++;
            if (frame_count_ % 10 != 0) { // Chỉ xử lý 1/10 số frame để giảm tải
                return;
            }
            
            // Chuyển từ ROS image sang OpenCV image
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            cv::Mat image_np = cv_ptr->image;
            
            // Xử lý ảnh
            processImage(image_np, msg->header);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }
    
    void processImage(const cv::Mat& image_np, const std_msgs::Header& header) {
        try {
            static int process_counter = 0;
            process_counter++;
            
            // Skip even more frames for better performance
            if (process_counter % 20 != 0) {  // Process fewer frames (1/20 instead of 1/10)
                return;
            }
            
            // Kiểm tra ảnh
            if (image_np.empty() || image_np.cols == 0 || image_np.rows == 0) {
                ROS_WARN_THROTTLE(10.0, "Received empty or invalid image");
                return;
            }
            
            // Less frequent debug info
            ROS_INFO_THROTTLE(10.0, "Processing image: %dx%d", image_np.cols, image_np.rows);
            
            // Giảm kích thước ảnh hiển thị để tiết kiệm bộ nhớ
            cv::Mat resized_input;
            double scale = 300.0 / std::max(image_np.cols, image_np.rows);  // Use 300x300 for SSD
            cv::resize(image_np, resized_input, cv::Size(), scale, scale);
            
            // Create reference only
            cv::Mat display = resized_input;
            
            // Thực hiện phát hiện đối tượng
            std::vector<Object> detections;
            static bool last_detection_failed = false;
            static int consecutive_failures = 0;
            
            try {
                // CUDA may need a reset after errors
                if (last_detection_failed) {
                    cudaError_t err = cudaDeviceSynchronize();
                    if (err != cudaSuccess) {
                        ROS_WARN("Synchronizing CUDA device after failure: %s", cudaGetErrorString(err));
                    }
                }
                
                // Execute detection pipeline
                detector_->CopyFromMat(resized_input);
                detector_->Infer();
                detector_->PostProcess(detections, 0.5f);
                
                // Reset error counters on success
                last_detection_failed = false;
                consecutive_failures = 0;
                
                // Only process further if we found something
                if (!detections.empty()) {
                    // Limit logging to avoid console spam
                    int max_objects_to_report = std::min(size_t(5), detections.size());
                    ROS_INFO_THROTTLE(5.0, "Detected %zu objects (showing top %d)", 
                                    detections.size(), max_objects_to_report);
                    
                    // Only now create a copy to draw on
                    display = resized_input.clone();
                    
                    // Vẽ các detections lên ảnh
                    detector_->DrawObjects(display, detections);
                    
                    // Only publish image when we have detections
                    cv_bridge::CvImage out_msg;
                    out_msg.header = header;
                    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                    out_msg.image = display;
                    image_pub_.publish(out_msg.toImageMsg());
                }
            } catch (const std::exception& e) {
                last_detection_failed = true;
                consecutive_failures++;
                
                // Log error details with decreasing frequency as failures persist
                if (consecutive_failures <= 3 || consecutive_failures % 10 == 0) {
                    ROS_ERROR("Detection failed (%d consecutive errors): %s", 
                            consecutive_failures, e.what());
                }
                
                // If too many consecutive failures, try to recover CUDA context
                if (consecutive_failures % 20 == 0) {
                    ROS_WARN("Too many consecutive errors, trying to recover CUDA context");
                    cudaDeviceReset();
                }
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in processImage: %s", e.what());
        }
    }
    
    void run() {
        ros::spin();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ssd_object_detector");
    
    // Print banner
    ROS_INFO("Starting SSD object detector");
    
    SSDObjectDetector detector;
    detector.run();
    
    return 0;
}