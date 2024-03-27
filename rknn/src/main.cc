#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include <optional>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"

// FilterSubscriber 类定义，用于接收和处理过滤器类型的 ROS 消息
class FilterSubscriber : public rclcpp::Node {
public:
    FilterSubscriber(const std::string& name) : Node(name), new_filter_received(false) {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "filter_topic", 10,
            std::bind(&FilterSubscriber::filter_callback, this, std::placeholders::_1));
    }

    bool isNewFilterReceived() {
        bool temp = new_filter_received;
        new_filter_received = false;
        return temp;
    }

    std::string getFilterType() const {
        return filter_type;
    }

    void disableSubscription() {
        subscription_.reset(); // 关闭订阅
    }

private:
    void filter_callback(const std_msgs::msg::String::SharedPtr msg) {
        if (msg->data == "R" || msg->data == "B") {
            filter_type = msg->data;
            new_filter_received = true;
            RCLCPP_INFO(this->get_logger(), "New filter type received: '%s'", msg->data.c_str());
            disableSubscription(); // 接收到消息后关闭订阅
        }
    }

    std::string filter_type;
    bool new_filter_received;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};


int main(int argc, char **argv) {
    // 获取 FilterSubscriber 
    rclcpp::init(argc, argv);
    auto filter_subscriber = std::make_shared<FilterSubscriber>("filter_subscriber");

    // 初始化线程池
    int threadNum = 12;
    std::unique_ptr<rknnPool<rkYolov5s, cv::Mat, cv::Mat>> testPool;
    cv::namedWindow("Camera Left");
    cv::namedWindow("Camera Right");
    cv::VideoCapture captureLeft, captureRight;
std::string videoFilePath = "/home/orangepi/Desktop/output.mp4";
captureLeft.open(0);  // 继续使用摄像头作为左侧输入
captureRight.open(videoFilePath); // 使用视频文件作为右侧输入

    // 初始化时间和帧数计数器
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    auto beforeTime = startTime;
    //
    bool startInference = false;

    // 等待新的过滤类型消息主循环
while (rclcpp::ok()) {
        rclcpp::spin_some(filter_subscriber);

        if (filter_subscriber->isNewFilterReceived()) {
            std::string filterType = filter_subscriber->getFilterType();
            RCLCPP_INFO(filter_subscriber->get_logger(), "Setting filter type to: '%s'", filterType.c_str());

            if(filterType=="B"){//这里放B模型
                 testPool = std::make_unique<rknnPool<rkYolov5s, cv::Mat, cv::Mat>>("/home/orangepi/Desktop/320_B_M.rknn", threadNum,"sortpath");
                 
                 if(testPool->init()!=0){
                    printf("rknnPool init fail!\n");
                    return -1;
                }
                startInference = true;
            }
            if(filterType=="R"){
                testPool = std::make_unique<rknnPool<rkYolov5s, cv::Mat, cv::Mat>>("/home/orangepi/Desktop/auxiliary_camera_for_soldier/rknn/model/yolov5_L.rknn", threadNum,"sortpath");
                
                if(testPool->init()!=0){
                    printf("rknnPool init fail!\n");
                    return -1;
                }
                startInference = true;
                
            } 
            startInference = true;
            break;
        }
    }
    
    // 处理摄像头图像
    while (startInference) {
        cv::Mat imgLeft, imgRight;
        bool hasFrameLeft = captureLeft.read(imgLeft);
        bool hasFrameRight = captureRight.read(imgRight);

        if (!hasFrameLeft || !hasFrameRight) {
            std::cerr << "Failed to capture frame from camera." << std::endl;
            break;
        }

        if (imgLeft.empty() || imgRight.empty()) {
            std::cerr << "Empty frame received from camera." << std::endl;
            continue;
        }

        // 将图像放入线程池进行处理
        if (hasFrameLeft && testPool->put(std::make_pair(imgLeft, "left")) != 0)
            break;
        if (hasFrameRight && testPool->put(std::make_pair(imgRight, "right")) != 0)
            break;


        std::pair<cv::Mat, std::string> resultLeft;
        std::pair<cv::Mat, std::string> resultRight;
        if (frames >= threadNum) {
            if (testPool->get(resultLeft) != 0 || testPool->get(resultRight) != 0)
                break;

            cv::imshow("Camera Left", resultLeft.first);
            cv::imshow("Camera Right", resultRight.first);
        }


        if (cv::waitKey(1) == 'q')
            break;
        frames++;

        // 每120帧计算平均帧率
        if (frames % 120 == 0) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    // 计算总平均帧率并退出
    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    rclcpp::shutdown();
    return 0;
}
