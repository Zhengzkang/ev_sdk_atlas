/*
 * Copyright (c) 2021 ExtremeVision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
* 本demo采用YOLOv5目标检测器,检测行人
* 首次运行时采用加载onnx模型,atc转换生成.om模型,以便下次运行时直接加载.om模型,加快初始化
*
*/

#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H
#include <memory>
#include <map>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "opencv2/core.hpp"
#include "ji.h"
#include "ji_utils.h"
#define STATUS int
#define TRY_RELEASE_POINTER(val, func) if(val != nullptr){func(val); val = nullptr;}
#define TRY_DELETE_POINTER(val, func)  if(val != nullptr){func val; val = nullptr;}
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

static float IOU(const cv::Rect& b1, const cv::Rect& b2)
{
    auto intersec = b1 & b2;
    return static_cast<float>(intersec.area()) / ( b1.area() + b2.area() - intersec.area() );
}

class SampleDetector
{            
    public:
        SampleDetector();
        ~SampleDetector();
        STATUS Init(const std::string& modelpath, float thresh);
        STATUS UnInit();        
        STATUS ProcessImage(const cv::Mat& img, std::vector< BoxInfo >& DetObj, float thresh = 0.15);                   
        static void runNms(std::vector<BoxInfo>& objects, float iou_thresh);
    private:
        void decode_outputs(float* prob,  std::vector<BoxInfo>& objects, float scale, const int img_w, const int img_h);
    
    public:
        // 接口的返回值定义
        static const int ERROR_BASE = 0x0200;
        static const int ERROR_INPUT = 0x0201;
        static const int ERROR_INIT = 0x0202;
        static const int ERROR_PROCESS = 0x0203;
        static const int ERROR_INITACL = 0x0204;
        static const int ERROR_INITMODEL = 0x0205;
        static const int ERROR_INITDVPP = 0x0206;
        static const int STATUS_SUCCESS = 0x0000;   
        
    private:
        aclmdlIODims m_input_dims;
        void* m_input_buffer = nullptr;
        aclDataBuffer* m_input_data_buffer = nullptr;
        aclmdlDataset* m_input_dataset = nullptr;

        aclmdlIODims m_output_dims;
        void* m_output_buffer = nullptr;
        void* m_output_buffer_host = nullptr;
        aclDataBuffer* m_output_data_buffer = nullptr;
        aclmdlDataset* m_output_dataset = nullptr;

    private:
        aclrtRunMode m_acl_run_mode;
        aclrtContext m_acl_context = nullptr;
        aclrtStream  m_acl_stream;
        aclmdlDesc*  m_model_desc;
        uint32_t     m_model_id = 0;    

        size_t mModelMSize{0};
        size_t mModelWSize{0};
        void *mModelMptr{nullptr};
        void *mModelWptr{nullptr};

    private:
        cv::Size m_InputSize;
        cv::Mat m_Resized;
        cv::Mat m_Normalized;
        std::vector<cv::Mat> m_chw_wrappers{};     
        bool m_init_flag = false;
        float mThresh;
        int m_iClassNums;
        int m_iBoxNums;
};



#endif 