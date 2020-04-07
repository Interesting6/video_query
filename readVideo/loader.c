// CoViAR python data loader.
// Part of this implementation is modified from the tutorial at
// https://blog.csdn.net/leixiaohua1020/article/details/50618190
// and FFmpeg extract_mv example.


#include <Python.h>
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdio.h>
#include <omp.h>

#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>

#define FF_INPUT_BUFFER_PADDING_SIZE 32
#define MV 1
#define RESIDUAL 2

static const char *filename = NULL;


static PyObject *CoviarError; // 静态全局变量


void create_and_load_bgr(AVFrame *pFrame, AVFrame *pFrameBGR, uint8_t *buffer,
    PyArrayObject ** arr, int cur_pos, int pos_target) {

    int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height); // 算出某格式和分辨率下一帧图像的数据大小 
    buffer = (uint8_t*) av_malloc(numBytes * sizeof(uint8_t)); // 分配保存图像的内存
    avpicture_fill((AVPicture*) pFrameBGR, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height); // 将自己分配的内存绑定到AVFrame帧的data数据区 

    struct SwsContext *img_convert_ctx;
    // 转换上下文?  一路码流转换
    img_convert_ctx = sws_getCachedContext(NULL, 
        pFrame->width, pFrame->height, AV_PIX_FMT_YUV420P, 
        pFrame->width, pFrame->height, AV_PIX_FMT_BGR24, 
        SWS_BICUBIC, NULL, NULL, NULL);

    // 像素格式和分辨率的转换
    sws_scale(img_convert_ctx,  // 转换格式的上下文
        pFrame->data,           // 输入图像的每个颜色通道的数据指针
        pFrame->linesize, 0, pFrame->height, // 输入图像的每个颜色通道的跨度; 在输入图像上处理区域, Y=0是起始，H为处理多少行
        pFrameBGR->data,        // 定义输出图像信息
        pFrameBGR->linesize);
    sws_freeContext(img_convert_ctx); // 释放sws_scale

    int linesize = pFrame->width * 3; // width*channel
    int height = pFrame->height;

    int stride_0 = height * linesize; // h*w*3
    int stride_1 = linesize;          // w*3
    int stride_2 = 3;                 // 3

    uint8_t *src  = (uint8_t*) pFrameBGR->data[0];
    uint8_t *dest = (uint8_t*) (*arr)->data;

    int array_idx;
    if (cur_pos == pos_target) {
        array_idx = 1;
    } else {
        array_idx = 0;
    }
    memcpy(dest + array_idx * stride_0, src, height * linesize * sizeof(uint8_t));
    av_free(buffer);
}


void create_and_load_mv_residual(
    AVFrameSideData *sd, 
    PyArrayObject * bgr_arr,
    PyArrayObject * mv_arr,
    PyArrayObject * res_arr,
    int cur_pos,
    int accumulate,
    int representation,
    int *accu_src, 
    int *accu_src_old,
    int width,
    int height,
    int pos_target) {

    int p_dst_x, p_dst_y, p_src_x, p_src_y, val_x, val_y;
    const AVMotionVector *mvs = (const AVMotionVector *)sd->data; // 转换为运动矢量数据，所指的数据不能修改

    for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
        const AVMotionVector *mv = &mvs[i];
        assert(mv->source == -1); // 过去的参考帧 

        if (mv->dst_x - mv->src_x != 0 || mv->dst_y - mv->src_y != 0) {

            val_x = mv->dst_x - mv->src_x;  // x的移动量
            val_y = mv->dst_y - mv->src_y;  // y的移动量

            for (int x_start = (-1 * mv->w / 2); x_start < mv->w / 2; ++x_start) {
                for (int y_start = (-1 * mv->h / 2); y_start < mv->h / 2; ++y_start) { // 中心原点在中间？
                    p_dst_x = mv->dst_x + x_start;
                    p_dst_y = mv->dst_y + y_start;

                    p_src_x = mv->src_x + x_start;
                    p_src_y = mv->src_y + y_start; // 移原点到左上角？

                    if (p_dst_y >= 0 && p_dst_y < height && 
                        p_dst_x >= 0 && p_dst_x < width &&
                        p_src_y >= 0 && p_src_y < height &&  
                        p_src_x >= 0 && p_src_x < width) {

                        // Write MV. 
                        if (accumulate) {
                            for (int c = 0; c < 2; ++c) {
                                accu_src       [p_dst_x * height * 2 + p_dst_y * 2 + c]
                                 = accu_src_old[p_src_x * height * 2 + p_src_y * 2 + c];
                            }
                        } else {
                            *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = val_x;
                            *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = val_y;
                        }
                    }
                }
            }
        }
    }
    if (accumulate) {
        memcpy(accu_src_old, accu_src, width * height * 2 * sizeof(int));
    }
    if (cur_pos > 0){
        if (accumulate) {
            if (representation == MV && cur_pos == pos_target) {
                for (int x = 0; x < width; ++x) {
                    for (int y = 0; y < height; ++y) {
                        *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0))
                         = x - accu_src[x * height * 2 + y * 2];
                        *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1))
                         = y - accu_src[x * height * 2 + y * 2 + 1];
                    }
                }
            }
        }
        if (representation == RESIDUAL && cur_pos == pos_target) {

            uint8_t *bgr_data = (uint8_t*) bgr_arr->data;
            int32_t *res_data = (int32_t*) res_arr->data;

            
            int stride_0 = height * width * 3;
            int stride_1 = width * 3;
            int stride_2 = 3;
            
            int y;

            for (y = 0; y < height; ++y) {
                int c, x, src_x, src_y, location, location2, location_src;
                int32_t tmp;
                for (x = 0; x < width; ++x) {
                    tmp = x * height * 2 + y * 2;
                    if (accumulate) {
                        src_x = accu_src[tmp];
                        src_y = accu_src[tmp + 1];
                    } else {
                        src_x = x - (*((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0)));
                        src_y = y - (*((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1)));
                    }
                    location_src = src_y * stride_1 + src_x * stride_2;

                    location = y * stride_1 + x * stride_2; 
                    for (c = 0; c < 3; ++c) {
                        location2 = stride_0 + location;
                        res_data[location] =  (int32_t) bgr_data[location2]
                                            - (int32_t) bgr_data[location_src + c];
                        location += 1;
                    }
                }
            }
        }
    }
}


int decode_video(
    int gop_target,
    int pos_target,
    PyArrayObject ** bgr_arr, 
    PyArrayObject ** mv_arr, 
    PyArrayObject ** res_arr, 
    int representation,
    int accumulate) {

    AVCodec *pCodec; // 采用的解码器AVCodec（H.264,MPEG2...）
    AVCodecContext *pCodecCtx= NULL;   // 包含了众多编解码器需要的参数信息
    AVCodecParserContext *pCodecParserCtx=NULL;  

    FILE *fp_in;
    AVFrame *pFrame;
    AVFrame *pFrameBGR;
    
    const int in_buffer_size=4096;  
    uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];  // FF_INPUT_BUFFER_PADDING_SIZE=32
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);  // 从上面第一个到第二个填充的部分设为0？

    uint8_t *cur_ptr;  
    int cur_size;
    int cur_gop = -1;
    AVPacket packet;  
    int ret, got_picture;
      
    avcodec_register_all();  // 注册编解码器等，只有调用了该函数，才能使用编解码器
  
    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);   // 获取解码器，AV_CODEC_ID_MPEG4=12，ie MPEG4的CodecID
    // pCodec = avcodec_find_decoder(AV_CODEC_ID_H264);  
    if (!pCodec) {  
        printf("Codec not found\n");  
        return -1;  
    }  
    pCodecCtx = avcodec_alloc_context3(pCodec);  // 空间申请
    if (!pCodecCtx){  
        printf("Could not allocate video codec context\n");  
        return -1;  
    }  

    pCodecParserCtx=av_parser_init(AV_CODEC_ID_MPEG4);  // 初始化AVCodecParserContext
    // pCodecParserCtx=av_parser_init(AV_CODEC_ID_H264);  
    if (!pCodecParserCtx){  
        printf("Could not allocate video parser context\n");  
        return -1;  
    }  
  
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "flags2", "+export_mvs", 0);      
    if (avcodec_open2(pCodecCtx, pCodec, &opts) < 0) {  // 初始化一个视音频编解码器的AVCodecContext
        printf("Could not open codec\n");  
        return -1;  
    }  
    //Input File  
    fp_in = fopen(filename, "rb");  
    if (!fp_in) {  
        printf("Could not open input stream\n");  
        return -1;  
    }  

    int cur_pos = 0;

    pFrame = av_frame_alloc();  // 分配内存
    pFrameBGR = av_frame_alloc();

    uint8_t *buffer;

    av_init_packet(&packet);

    int *accu_src = NULL;
    int *accu_src_old = NULL;

    while (1) {

        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);  // 从给定流fp_in读取数据到in_buffer所指向的数组中
        // 每次读in_buffer_size=4096 bite，要读取的每个元素大小为1。
        if (cur_size == 0)  
            break;  
        cur_ptr=in_buffer;  
  
        while (cur_size>0){  
  
            int len = av_parser_parse2(  // 拿到AVPaket数据，从输入的数据流中分离出一帧一帧的压缩编码数据，输入为H.264、HEVC码流文件
                pCodecParserCtx, pCodecCtx,  
                &packet.data, &packet.size,  
                cur_ptr , cur_size ,  // 一次接收的数据包 和 本次接收数据包的长度
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);  

            cur_ptr += len;  
            cur_size -= len;

            if(packet.size==0)  // 包里没有数据？
                continue;  

            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {  // 读到I帧的时候
                ++cur_gop;
            }

            if (cur_gop == gop_target && cur_pos <= pos_target) {  //当前gop是目标的gop，且位置比目标位置低
      
                ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet); // 解码一帧视频数据 
                // 一个压缩编码的结构体AVPacket，输出一个解码后的结构体AVFrame。
                if (ret < 0) {  
                    printf("Decode Error.\n");  
                    return -1;  
                }
                int h = pFrame->height;
                int w = pFrame->width;

                // Initialize arrays. 
                if (! (*bgr_arr)) {
                    npy_intp dims[4];
                    dims[0] = 2;
                    dims[1] = h;
                    dims[2] = w;
                    dims[3] = 3;
                    *bgr_arr = PyArray_ZEROS(4, dims, NPY_UINT8, 0);
                }

                if (representation == MV && ! (*mv_arr)) {
                    npy_intp dims[3];
                    dims[0] = h;
                    dims[1] = w;
                    dims[2] = 2;
                    *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
                }

                if (representation == RESIDUAL && ! (*res_arr)) {
                    npy_intp dims[3];
                    dims[0] = h;
                    dims[1] = w;
                    dims[2] = 3;

                    *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
                    *res_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
                }

                if ((representation == MV ||
                     representation == RESIDUAL) && accumulate && 
                    !accu_src && !accu_src_old) {
                    accu_src     = (int*) malloc(w * h * 2 * sizeof(int));
                    accu_src_old = (int*) malloc(w * h * 2 * sizeof(int));

                    for (size_t x = 0; x < w; ++x) {
                        for (size_t y = 0; y < h; ++y) {
                            accu_src_old[x * h * 2 + y * 2    ]  = x;
                            accu_src_old[x * h * 2 + y * 2 + 1]  = y;
                        }
                    }
                    memcpy(accu_src, accu_src_old, h * w * 2 * sizeof(int));
                }

                if (got_picture) {

                    if ((cur_pos == 0              && accumulate  && representation == RESIDUAL) ||
                        (cur_pos == pos_target - 1 && !accumulate && representation == RESIDUAL) ||
                        cur_pos == pos_target) {
                        create_and_load_bgr(
                            pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, pos_target);
                    }

                    if (representation == MV || 
                        representation == RESIDUAL) {
                        AVFrameSideData *sd;
                        sd = av_frame_get_side_data(pFrame, AV_FRAME_DATA_MOTION_VECTORS);
                        if (sd) {
                            if (accumulate || cur_pos == pos_target) {
                                create_and_load_mv_residual(
                                    sd, 
                                    *bgr_arr, *mv_arr, *res_arr,
                                    cur_pos,
                                    accumulate,
                                    representation,
                                    accu_src,
                                    accu_src_old,
                                    w,
                                    h,
                                    pos_target);
                            }
                        }
                    }
                    cur_pos ++;
                }
            }
        }
    }
  
    //Flush Decoder  
    packet.data = NULL;  
    packet.size = 0;  
    while(1){  
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);  
        if (ret < 0) {  
            printf("Decode Error.\n");  
            return -1;  
        }  
        if (!got_picture) {
            break;  
        } else if (cur_gop == gop_target) {
            if ((cur_pos == 0 && accumulate) ||
                (cur_pos == pos_target - 1 && !accumulate) ||
                cur_pos == pos_target) {
                create_and_load_bgr(
                    pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, pos_target);
            }
        }  
    }  

    fclose(fp_in);

    av_parser_close(pCodecParserCtx);  
  
    av_frame_free(&pFrame);  
    av_frame_free(&pFrameBGR);  
    avcodec_close(pCodecCtx);  
    av_free(pCodecCtx);
    if ((representation == MV || 
         representation == RESIDUAL) && accumulate) {
        if (accu_src) {
            free(accu_src);
        }
        if (accu_src_old) {
            free(accu_src_old);
        }
    }
  
    return 0;  
}  


void count_frames(int* gop_count, int* frame_count) {

    AVCodec *pCodec;
    AVCodecContext *pCodecCtx= NULL;  
    AVCodecParserContext *pCodecParserCtx=NULL;  

    FILE *fp_in;
    
    const int in_buffer_size=4096;  
    uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    uint8_t *cur_ptr;  
    int cur_size;  
    AVPacket packet;  

    avcodec_register_all();  
  
    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);  
    // pCodec = avcodec_find_decoder(AV_CODEC_ID_H264);  
    if (!pCodec) {  
        printf("Codec not found\n");  
        return -1;  
    }  
    pCodecCtx = avcodec_alloc_context3(pCodec);  
    if (!pCodecCtx){  
        printf("Could not allocate video codec context\n");  
        return -1;  
    }  

    pCodecParserCtx=av_parser_init(AV_CODEC_ID_MPEG4);  
    // pCodecParserCtx=av_parser_init(AV_CODEC_ID_H264);  
    if (!pCodecParserCtx){  
        printf("Could not allocate video parser context\n");  
        return -1;  
    }  

    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {  
        printf("Could not open codec\n");  
        return -1;  
    }  

    //Input File  
    fp_in = fopen(filename, "rb");  
    if (!fp_in) {  
        printf("Could not open input stream\n");  
        return -1;  
    }  

    *gop_count = 0;
    *frame_count = 0;

    av_init_packet(&packet);  

    while (1) {

        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);  
        if (cur_size == 0)  
            break;  
        cur_ptr=in_buffer;  
  
        while (cur_size>0){  
  
            int len = av_parser_parse2(  
                pCodecParserCtx, pCodecCtx,  
                &packet.data, &packet.size,  
                cur_ptr , cur_size ,  
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);  

            cur_ptr += len;  
            cur_size -= len;  

            if(packet.size==0)  
                continue;  
            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {
                ++(*gop_count);
            }
            ++(*frame_count);
        }
    }

    fclose(fp_in);  
    av_parser_close(pCodecParserCtx);  

    avcodec_close(pCodecCtx);  
    av_free(pCodecCtx);

    return 0;
}


static PyObject *get_num_gops(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    count_frames(&gop_count, &frame_count);
    return Py_BuildValue("i", gop_count);
}


static PyObject *get_num_frames(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    count_frames(&gop_count, &frame_count);
    return Py_BuildValue("i", frame_count);
}


static PyObject *load(PyObject *self, PyObject *args)
{

    PyObject *arg1 = NULL; // filename.
    int gop_target, pos_target, representation, accumulate;

    if (!PyArg_ParseTuple(args, "siiii", &filename,
        &gop_target, &pos_target, &representation, &accumulate)) return NULL;

    PyArrayObject *bgr_arr = NULL;
    PyArrayObject *final_bgr_arr = NULL;
    PyArrayObject *mv_arr = NULL;
    PyArrayObject *res_arr = NULL;

    if(decode_video(gop_target, pos_target,
                    &bgr_arr, &mv_arr, &res_arr, 
                    representation,
                    accumulate) < 0) {
        printf("Decoding video failed.\n");

        Py_XDECREF(bgr_arr);
        Py_XDECREF(mv_arr);
        Py_XDECREF(res_arr);
        return Py_None;
    }
    if(representation == MV) {
        Py_XDECREF(bgr_arr);
        Py_XDECREF(res_arr);
        return mv_arr;

    } else if(representation == RESIDUAL) {
        Py_XDECREF(bgr_arr);
        Py_XDECREF(mv_arr);
        return res_arr;

    } else {
        Py_XDECREF(mv_arr);
        Py_XDECREF(res_arr);

        npy_intp *dims_bgr = PyArray_SHAPE(bgr_arr);
        int h = dims_bgr[1];
        int w = dims_bgr[2];

        npy_intp dims[3];
        dims[0] = h;
        dims[1] = w;
        dims[2] = 3;
        PyArrayObject *final_bgr_arr = PyArray_ZEROS(3, dims, NPY_UINT8, 0);

        int size = h * w * 3 * sizeof(uint8_t);
        memcpy(final_bgr_arr->data, bgr_arr->data + size, size);

        Py_XDECREF(bgr_arr);
        return final_bgr_arr;
    }
}


static PyMethodDef CoviarMethods[] = {
    {"load",  load, METH_VARARGS, "Load a frame."},
    {"get_num_gops",  get_num_gops, METH_VARARGS, "Getting number of GOPs in a video."},
    {"get_num_frames",  get_num_frames, METH_VARARGS, "Getting number of frames in a video."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef coviarmodule = {
    PyModuleDef_HEAD_INIT,
    "coviar",   /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CoviarMethods
};


PyMODINIT_FUNC PyInit_coviar(void)
{
    PyObject *m;

    m = PyModule_Create(&coviarmodule);
    if (m == NULL)
        return NULL;

    /* IMPORTANT: this must be called */
    import_array();

    CoviarError = PyErr_NewException("coviar.error", NULL, NULL);
    Py_INCREF(CoviarError);
    PyModule_AddObject(m, "error", CoviarError);
    return m;
}


int main(int argc, char *argv[])
{
    av_log_set_level(AV_LOG_QUIET);

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("coviar", PyInit_coviar);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}
