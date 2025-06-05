#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "ONNXDetector.h"

class ObjectDetector {
private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher image_pub_;
    bool use_compressed_;
    std::unique_ptr<ONNXDetector> detector_;
    int frame_count_;

public:
    ObjectDetector() : use_compressed_(true), frame_count_(0) {
        // Lấy parameters từ ROS parameter server
        nh_.param<bool>("use_compressed", use_compressed_, true);
        float confidence_threshold;
        nh_.param<float>("confidence_threshold", confidence_threshold, 0.3f);
        
        // Đọc đường dẫn mô hình từ parameter
        std::string model_path;
        nh_.param<std::string>("model_path", model_path, 
                              "/home/jetson/jetson-inference/python/training/detection/ssd/models/ViDu/mb2-ssd-lite.onnx");
        
        ROS_INFO("Using model path: %s", model_path.c_str());
        ROS_INFO("Using confidence threshold: %.2f", confidence_threshold);
        
        // Kiểm tra xem file model có tồn tại không
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            ROS_ERROR("Model file not found: %s", model_path.c_str());
            return;
        }
        model_file.close();
        
        // Khởi tạo ONNX detector
        try {
            detector_ = std::make_unique<ONNXDetector>(model_path, confidence_threshold);
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
                                      &ObjectDetector::compressedImageCallback, this);
        } else {
            ROS_INFO("Subscribing to raw topic: %s", camera_topic.c_str());
            image_sub_ = nh_.subscribe(camera_topic, 1, 
                                      &ObjectDetector::imageCallback, this);
        }
        
        image_pub_ = nh_.advertise<sensor_msgs::Image>("/detected_objects", 1);
        
        ROS_INFO("Object detector initialized. Waiting for images...");
    }
    
    void compressedImageCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        try {
            // Tăng frame counter để bỏ qua frame nếu cần
            frame_count_++;
            if (frame_count_ % 10 != 0) { // Chỉ xử lý 1/10 số frame để giảm tải - more aggressive skipping
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
            if (frame_count_ % 10 != 0) { // Chỉ xử lý 1/10 số frame để giảm tải - more aggressive skipping
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
            
            // Skip even more frames for better performance (9 out of 10 frames)
            if (process_counter % 10 != 0) {
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
            // Make the resize much smaller for Jetson Nano
            cv::Mat resized_input;
            double scale = 320.0 / std::max(image_np.cols, image_np.rows);
            cv::resize(image_np, resized_input, cv::Size(), scale, scale);
            
            // Don't clone the image until we know we have detections
            // Create reference only
            cv::Mat display = resized_input;
            
            // Don't use CUDA operations unnecessarily
            // cv::cuda::GpuMat temp;
            // temp.release();
            // cudaDeviceSynchronize();
            
            // Thực hiện phát hiện đối tượng
            std::vector<Detection> detections;
            try {
                detections = detector_->detect(resized_input);
                
                // Only process further if we found something
                if (!detections.empty()) {
                    // Log kết quả less frequently
                    ROS_INFO_THROTTLE(5.0, "Detected %zu objects", detections.size());
                    
                    // Only now create a copy to draw on
                    display = resized_input.clone();
                    
                    // Vẽ các detections lên ảnh
                    detector_->drawDetections(display, detections);
                    
                    // Enable image display to show bounding boxes
                    static bool display_available = true;
                    if (display_available) {
                        try {
                            cv::imshow("Object Detection", display);
                            cv::waitKey(1);
                        } catch (const std::exception& e) {
                            display_available = false;
                            ROS_WARN("Cannot display image: %s", e.what());
                        }
                    }
                    
                    // Only publish image when we have detections
                    cv_bridge::CvImage out_msg;
                    out_msg.header = header;
                    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                    out_msg.image = display;
                    image_pub_.publish(out_msg.toImageMsg());
                }
            } catch (const std::exception& e) {
                ROS_ERROR_THROTTLE(10.0, "Detection failed"); // Less detail in error logs
            }
            
            // Image publishing is now done only when detections exist
            // to save bandwidth and processing power
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in processImage: %s", e.what());
        }
    }
    
    void run() {
        ros::spin();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_detector");
    
    // Print banner
    ROS_INFO("Starting object detector with SSD-MobileNet-v2 ONNX");
    
    ObjectDetector detector;
    detector.run();
    
    return 0;
}

