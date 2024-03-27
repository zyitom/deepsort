#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>
#include <string>

// rknnModel模型类, inputType模型输入类型, outputType模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool {
private:
    int threadNum;
    std::string modelPath;

    long long id;
    std::mutex idMtx, queueMtx;
    std::unique_ptr<dpool::ThreadPool> pool;
    std::queue<std::pair<std::future<outputType>, std::string>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;

protected:
    int getModelId();

public:
    rknnPool(const std::string modelPath, int threadNum);
    int init();
    // 修改了put方法的签名，现在接受一个包含inputData和来源标识的pair
    int put(std::pair<inputType, std::string> inputData);
    // get方法现在也需要返回来源标识
    int get(std::pair<outputType, std::string> &outputData);
    ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string modelPath, int threadNum,const std::string sort_path) {
    this->modelPath = modelPath;
    this->threadNum = threadNum;
    this->sort_path = sort_path;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init() {
    try {
        this->pool = std::make_unique<dpool::ThreadPool>(this->threadNum);
        for (int i = 0; i < this->threadNum; i++) {
            models.push_back(std::make_shared<rknnModel>(this->modelPath.c_str()));
        }
    } catch (const std::bad_alloc &e) {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }
    // 初始化模型
    for (int i = 0, ret = 0; i < threadNum; i++) {
        ret = models[i]->init(models[0]->get_pctx(), i != 0);
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId() {
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(std::pair<inputType, std::string> inputData) {
    auto modelId = this->getModelId();
    // 提交任务到线程池
    auto fut = pool->submit([model = models[modelId], inputData]() mutable {
        // 在lambda内部创建副本
        cv::Mat inputCopy = inputData.first.clone();
        return model->infer(inputCopy, inputData.second);
    });
    // 存储future和来源标识
    futs.push(std::make_pair(std::move(fut), inputData.second));
    return 0;
}


template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(std::pair<outputType, std::string> &outputData) {
    std::lock_guard<std::mutex> lock(queueMtx);
    if (futs.empty()) {
        return 1;
    }
    auto &pair = futs.front();
    outputData.first = pair.first.get(); // 调用get()在正确的对象上
    outputData.second = pair.second;     // 获取摄像头标识
    futs.pop();
    return 0;
}

// 修改对future对象的处理
template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool() {
    while (!futs.empty()) {
        try {
            auto &pair = futs.front();
            pair.first.get(); // 仅调用get等待任务完成，不保存结果
        } catch (const std::exception& e) {
            std::cerr << "Exception caught from future: " << e.what() << std::endl;
        }
        futs.pop();
    }
}

#endif // RKNNPOOL_H
