#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

#include "SampleDetector.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;

static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}

SampleDetector::SampleDetector()
{
    
}

STATUS SampleDetector::Init(const std::string& modelpath, float thresh)
{
    mThresh = thresh;
    std::string strModelName = "/usr/local/ev_sdk/model/model.om";
    size_t sep_pos = strModelName.find_last_of(".");
    string om_Name = strModelName.substr(0, sep_pos);
    string soc_version = aclrtGetSocName();
    SDKLOG(INFO) << "soc_version : " << soc_version;
    if(!ifFileExists(strModelName.c_str()))
    {        
        string cmd_str = "atc --framework=5 --model=" + modelpath + " --output=" + om_Name + "  --soc_version=" + soc_version;
        system(cmd_str.c_str());
    }
    
    SDKLOG(INFO) << "start to init model ";
    m_acl_context = nullptr;
    m_acl_stream = nullptr;
    uint32_t deviceCount;
    auto ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "No device found! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << deviceCount << " devices found";

    ret = aclrtSetDevice(m_model_id);      
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "Acl open device " << m_model_id << " failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "Open device " << m_model_id << " success";

    ret = aclrtCreateContext(&m_acl_context, m_model_id);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl create context failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "create context success";

    ret = aclrtSetCurrentContext(m_acl_context);  
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl set context failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "set context success";

    ret = aclrtCreateStream(&m_acl_stream);
    if(ret != 0)
    {
        SDKLOG(ERROR) << "failed to create  stream! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "create stream success";

    ret = aclrtGetRunMode(&m_acl_run_mode);
    if(ret != 0)
    {
        SDKLOG(ERROR) << "failed to get acl run mode! aclError= " << ret;
        return ERROR_INITACL;
    }
    if(m_acl_run_mode == ACL_DEVICE)
    {
        SDKLOG(INFO) << "run in device mode";
    }
    else
    {
        SDKLOG(INFO) << "run in host mode";
    }

    SDKLOG(INFO) << "load model " << strModelName;
    ret = aclmdlQuerySize(strModelName.c_str(), &mModelMSize, &mModelWSize);
    ret = aclrtMalloc(&mModelMptr, mModelMSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&mModelWptr, mModelWSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclmdlLoadFromFileWithMem(strModelName.c_str(), &m_model_id, mModelMptr,mModelMSize, mModelWptr, mModelWSize);
    m_model_desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(m_model_desc, m_model_id);  
    
    int input_num = aclmdlGetNumInputs(m_model_desc);
    if(input_num != 1)
    {
        SDKLOG(ERROR) << "this is not a standard yolov5 model";
    }

    aclmdlGetInputDims(m_model_desc, 0, &m_input_dims);
    if(m_input_dims.dimCount != 4)
    {
        SDKLOG(ERROR) << "this is not a standard yolov5 model";
    }
    SDKLOG(INFO) << "input dim is : "<< m_input_dims.dims[0] << " " << m_input_dims.dims[1] << " " << m_input_dims.dims[2] << " " << m_input_dims.dims[3];
    m_InputSize = cv::Size(m_input_dims.dims[3], m_input_dims.dims[2]);

    int output_num = aclmdlGetNumOutputs(m_model_desc);
    if(output_num != 1)
    {
        SDKLOG(ERROR) << "this is not a standard yolov5 model";
    }
    
    aclmdlGetOutputDims(m_model_desc, 0, &m_output_dims);
    if(m_output_dims.dimCount != 3)
    {
        SDKLOG(ERROR) << "this is not a standard yolov5 model";
    }
    SDKLOG(INFO) << "output dim is : "<< m_output_dims.dims[0] << " " << m_output_dims.dims[1] << " " << m_output_dims.dims[2];
    m_iClassNums = m_output_dims.dims[2] - 5;
    m_iBoxNums = m_output_dims.dims[1];
    
    m_input_dataset = aclmdlCreateDataset();
    ret = aclrtMalloc(&m_input_buffer, aclmdlGetInputSizeByIndex(m_model_desc, 0), ACL_MEM_MALLOC_NORMAL_ONLY);
    m_input_data_buffer = aclCreateDataBuffer(m_input_buffer, aclmdlGetInputSizeByIndex(m_model_desc, 0));
    aclmdlAddDatasetBuffer(m_input_dataset, m_input_data_buffer);
    
    if( m_acl_run_mode == ACL_DEVICE)
    {
        SDKLOG(INFO) << "atlas run in device mode";
        size_t single_chn_size = m_input_dims.dims[0] * m_input_dims.dims[2] * m_input_dims.dims[3] * sizeof(float);
        m_chw_wrappers.emplace_back(m_input_dims.dims[2], m_input_dims.dims[3], CV_32FC1, m_input_buffer);
        m_chw_wrappers.emplace_back(m_input_dims.dims[2], m_input_dims.dims[3], CV_32FC1, m_input_buffer + single_chn_size);
        m_chw_wrappers.emplace_back(m_input_dims.dims[2], m_input_dims.dims[3], CV_32FC1, m_input_buffer + 2 * single_chn_size);
    }

    m_output_dataset = aclmdlCreateDataset();
    ret = aclrtMalloc(&m_output_buffer, aclmdlGetOutputSizeByIndex(m_model_desc, 0), ACL_MEM_MALLOC_NORMAL_ONLY);
    if(m_acl_run_mode == ACL_HOST) m_output_buffer_host= new char[aclmdlGetOutputSizeByIndex(m_model_desc, 0)]();
    m_output_data_buffer = aclCreateDataBuffer(m_output_buffer, aclmdlGetOutputSizeByIndex(m_model_desc, 0));
    aclmdlAddDatasetBuffer(m_output_dataset, m_output_data_buffer);

    m_init_flag = true;
    return STATUS_SUCCESS;
}

STATUS SampleDetector::UnInit()
{
    if(m_init_flag == false)
    {
        return false;
    }
    SDKLOG(INFO) << "in uninit func";
    TRY_RELEASE_POINTER(m_input_dataset, aclmdlDestroyDataset);
    TRY_RELEASE_POINTER(m_input_data_buffer, aclDestroyDataBuffer);
    TRY_RELEASE_POINTER(m_input_buffer, aclrtFree);

    TRY_RELEASE_POINTER(m_output_dataset, aclmdlDestroyDataset);
    TRY_RELEASE_POINTER(m_output_data_buffer, aclDestroyDataBuffer);
    TRY_RELEASE_POINTER(mModelMptr, aclrtFree);
    TRY_RELEASE_POINTER(mModelWptr, aclrtFree);

    TRY_DELETE_POINTER(m_output_buffer_host, delete[]);

    aclmdlDestroyDesc(m_model_desc);
    if(m_model_id > 0)
    {
        aclmdlUnload(m_model_id);
    }
    aclrtDestroyStream(m_acl_stream);
    aclrtDestroyContext(m_acl_context); 
    m_chw_wrappers.clear();
    m_init_flag = false;
}

SampleDetector::~SampleDetector()
{
    UnInit();   
}

STATUS SampleDetector::ProcessImage(const cv::Mat& img, std::vector<BoxInfo>& DetObjs, float thresh)
{
    auto ret = aclrtSetCurrentContext(m_acl_context);  
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl set context failed! aclError= " << ret;
        return ERROR_INITACL;
    }

    mThresh = thresh;
    DetObjs.clear();  
    float r = std::min(m_InputSize.height / static_cast<float>(img.rows), m_InputSize.width / static_cast<float>(img.cols));
    cv::Size new_size = cv::Size{img.cols * r, img.rows * r};    
    cv::Mat tmp_resized;    
    
    cv::resize(img, tmp_resized, new_size);
    cv::cvtColor(tmp_resized, tmp_resized, cv::COLOR_BGR2RGB);
    m_Resized = cv::Mat(cv::Size(m_InputSize.width, m_InputSize.height), CV_8UC3, cv::Scalar(114, 114, 114));    
    tmp_resized.copyTo(m_Resized(cv::Rect{0, 0, tmp_resized.cols, tmp_resized.rows}));

    m_Resized.convertTo(m_Normalized, CV_32FC3, 1/255.);
    cv::split(m_Normalized, m_chw_wrappers); 

    if(m_acl_run_mode == ACL_HOST)
    {
        size_t single_chn_size = m_chw_wrappers[0].rows * m_chw_wrappers[0].cols * sizeof(float);    
        aclrtMemcpy(m_input_buffer, single_chn_size, m_chw_wrappers[0].data, single_chn_size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy((char*)m_input_buffer + single_chn_size, single_chn_size, m_chw_wrappers[1].data, single_chn_size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy((char*)m_input_buffer + 2 * single_chn_size, single_chn_size, m_chw_wrappers[2].data, single_chn_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    
    ret = aclmdlExecute(m_model_id, m_input_dataset, m_output_dataset);

    float scale = std::min(m_InputSize.width / (img.cols * 1.0), m_InputSize.height / (img.rows * 1.0));
    void* obuf = nullptr;
    if(m_acl_run_mode == ACL_HOST) 
    {
        size_t cp_size = m_output_dims.dims[0] * m_output_dims.dims[1] * m_output_dims.dims[2] * sizeof(float);
        aclrtMemcpy(m_output_buffer_host, cp_size, m_output_buffer,  cp_size, ACL_MEMCPY_DEVICE_TO_HOST);
        obuf = m_output_buffer_host;
    }
    else
    {
        obuf = m_output_buffer;
    }
    decode_outputs((float*)obuf, DetObjs, scale, img.cols, img.rows);
    runNms(DetObjs, 0.45);
}

void SampleDetector::runNms(std::vector<BoxInfo>& objects, float iou_thresh) 
{
    auto cmp_lammda = [](const BoxInfo& b1, const BoxInfo& b2){return b1.score < b2.score;};
    std::sort(objects.begin(), objects.end(), cmp_lammda);
    for(int i = 0; i < objects.size(); ++i)
    {
        if( objects[i].score < 0.1 )
        {
            continue;
        }
        for(int j = i + 1; j < objects.size(); ++j)
        {
            cv::Rect rect1 = cv::Rect{objects[i].x1, objects[i].y1, objects[i].x2 - objects[i].x1, objects[i].y2 - objects[i].y1};
            cv::Rect rect2 = cv::Rect{objects[j].x1, objects[j].y1, objects[j].x2 - objects[j].x1, objects[j].y2 - objects[j].y1};
            if(IOU(rect1, rect2) > iou_thresh)   
            {
                objects[i].score = 0.f;
            }
        }
    }
    auto iter = objects.begin();
    while( iter != objects.end() )
    {
        if(iter->score < 0.1)
        {
            iter = objects.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}

void SampleDetector::decode_outputs(float* prob, std::vector<BoxInfo>& objects, float scale, const int img_w, const int img_h) 
{
    SDKLOG(INFO) << "---------" << scale;
    std::vector<BoxInfo> proposals;
    for(int i = 0; i < m_iBoxNums; ++i)
    {
        int index = i * (m_iClassNums + 5);        
        if(prob[index + 4] > mThresh)
        {            
            float x = prob[index] / scale;
            float y = prob[index + 1] / scale;
            float w = prob[index + 2] / scale;
            float h = prob[index + 3] / scale;
            float* max_cls_pos = std::max_element(prob + index + 5, prob + index + 4 + m_iClassNums);
            if((*max_cls_pos) * prob[index+4] > mThresh)
            {
                cv::Rect box{x- w * 0.5, y - h * 0.5, w, h};
                box = box & cv::Rect(0, 0, img_w, img_h);
                if( box.area() > 0)
                {
                    BoxInfo box_info = { box.x, box.y, box.x + box.width, box.y + box.height, (*max_cls_pos) * prob[index+4], max_cls_pos-(prob + index + 5)};
                    objects.push_back(box_info);
                }
            }
        }
    }
}