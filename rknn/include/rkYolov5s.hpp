#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H
#include <vector>
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include "opencv2/core/core.hpp"
#include "postprocess.h"
#include "deepsort.h"
//重新定义结构体为后面的deepsort作准备；
//326改到另外头文件
#endif
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

class DetectorNode : public rclcpp::Node {
public:
    DetectorNode(const std::string& name) : Node(name) {
        publisher_ = this->create_publisher<std_msgs::msg::String>("detection_topic", 10);
    }
    void publish_detection(const std::string& message) {
        auto msg = std_msgs::msg::String();
        msg.data = message;
        publisher_->publish(msg);
    }

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};


class rkYolov5s
{

private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    DeepSort leftDeepSort;
    DeepSort rightDeepSort;


    float nms_threshold, box_conf_threshold;
    bool filter_type_set = false;
public:

    rkYolov5s(const std::string &model_path,const std::string &sort_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    std::string filter_type; 
    void setFilterType(const std::string& type);
    cv::Mat infer(cv::Mat &orig_img, const std::string& camera_side);
    ~rkYolov5s();
    std::shared_ptr<DetectorNode> detector_node_;
    const std::string& getFilterType() const {
        return filter_type;
    }

};
#ifdef CORENUM_H
#endif