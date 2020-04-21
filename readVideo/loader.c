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
    /* pFrame为解码后的原始数据，pFrameBGR为一块空内存，buffer指针指向一块内存空间
    arr为解码的数据，cur_pos为gop里的当前位置，pos_target为gop里的目标位置
    流程：pFrame -转换-> pFrameBGR -复制内存到-> arr
    */

    int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height); // 算出某格式和分辨率下一帧图像的数据大小：3*w*h 
    buffer = (uint8_t*) av_malloc(numBytes * sizeof(uint8_t)); // 分配保存图像的内存。buffer指向该内存空间。
    avpicture_fill((AVPicture*) pFrameBGR, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);  // 会将pFrameRGB的数据按RGB24格式自动"关联"到buffer
    // pFrameRGB和buffer都是已经申请到的一段内存, 下面sws_scale调用时，pFrameRGB为转换完的数据，也自动到了buffer里面。

    struct SwsContext *img_convert_ctx;
    // 上下文转换器：YUV格式转BGR格式，大小不变，缩放算法为SWS_BICUBIC
    img_convert_ctx = sws_getCachedContext(NULL, 
        pFrame->width, pFrame->height, AV_PIX_FMT_YUV420P, 
        pFrame->width, pFrame->height, AV_PIX_FMT_BGR24, 
        SWS_BICUBIC, NULL, NULL, NULL);

    // 将pFrame转换为pFrameRBG
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

    int array_idx; // 这里就是下面bgr_arr的第一维为什么设为2的原因了。
    if (cur_pos == pos_target) { // 如果在gop里的   当前位置等于的目标位置
        array_idx = 1;  // arr的后一半，只有当当前位置等于目标位置时，才是当前帧的BGR图像，放在arr的后一半
    } else {
        array_idx = 0;  // arr的前一半，不是要取的那帧，所以放在arr的前一半。
    }
    // 在后面就是用arr后一半的frame 减 前一半的frame，从而得到residual。
    memcpy(dest + array_idx * stride_0, src, height * linesize * sizeof(uint8_t));
    av_free(buffer);  // 好像buffer也没有用到呀。。。
}


void create_and_load_mv_residual(
    AVFrameSideData *sd,      // 解码frame中的运动矢量
    PyArrayObject * bgr_arr,  // I帧的指针
    PyArrayObject * mv_arr,   // 运动矢量的指针
    PyArrayObject * res_arr,  // 帧差图的指针
    int cur_pos,              // 当前在GOP中的位置
    int accumulate,           // 累计
    int representation,       
    int *accu_src,            // 当前帧的参考帧的各个像素点在二维平面中位置的一维数组
    int *accu_src_old,        // 旧的参考帧的各个像素点在二维平面中位置的一维数组
    int width,
    int height,
    int pos_target) {

    int p_dst_x, p_dst_y, p_src_x, p_src_y, val_x, val_y;
    const AVMotionVector *mvs = (const AVMotionVector *)sd->data; // 转换为运动矢量数据，所指的数据不能修改

    // 遍历sd(mvs)里面的每一个mv
    for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
        const AVMotionVector *mv = &mvs[i]; 
        assert(mv->source == -1); // 断言为：由过去的参考帧得到的mv

        // mv的坐标有变化：
        if (mv->dst_x - mv->src_x != 0 || mv->dst_y - mv->src_y != 0) { 

            val_x = mv->dst_x - mv->src_x;  // x的移动量
            val_y = mv->dst_y - mv->src_y;  // y的移动量

            // 遍历宏块，从-w/2到w/2；-h/2到h/2
            for (int x_start = (-1 * mv->w / 2); x_start < mv->w / 2; ++x_start) {
                for (int y_start = (-1 * mv->h / 2); y_start < mv->h / 2; ++y_start) { // 零点在图像中心？
                    p_dst_x = mv->dst_x + x_start;  // 绝对位置 变为 相对位置？
                    p_dst_y = mv->dst_y + y_start;

                    p_src_x = mv->src_x + x_start;
                    p_src_y = mv->src_y + y_start; // 移原点到左上角？

                    // 如果在图像里面
                    if (p_dst_y >= 0 && p_dst_y < height && 
                        p_dst_x >= 0 && p_dst_x < width &&
                        p_src_y >= 0 && p_src_y < height &&  
                        p_src_x >= 0 && p_src_x < width) {

                        // Write MV. 
                        if (accumulate) { // 累计情况，通过指针累计；但这里只是赋值，也没看到累加呀？
                            accu_src[p_dst_x*height*2 + p_dst_y*2] = accu_src_old[p_src_x*height*2 + p_src_y*2];
                            accu_src[p_dst_x*height*2 + p_dst_y*2 + 1] = accu_src_old[p_src_x*height*2 + p_src_y*2 + 1];
                            // 相当于a[i] = old_a[i]，不就是a = old_a吗？
                        } else { //非累计情况
                            *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = val_x; // 获得mv_arr在(y,x,0)处的指针，的值赋值
                            *((int32_t*)PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = val_y;
                            // 意思是(不考虑指针)：mv_arr[y, x, 0] = 点(x, y)在x上的变化量
                        }
                    }
                }
            }
        }
    }
    if (accumulate) {
        memcpy(accu_src_old, accu_src, width * height * 2 * sizeof(int)); // 新的替换掉老的
    }
    if (cur_pos > 0){
        if (accumulate) { //累计情况
            if (representation == MV && cur_pos == pos_target) { // 当前GOP里帧位置为目标帧
                for (int x = 0; x < width; ++x) {
                    for (int y = 0; y < height; ++y) {
                        *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0)) = x - accu_src[x * height * 2 + y * 2];
                        *((int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1)) = y - accu_src[x * height * 2 + y * 2 + 1];
                        // accu_src 为上一帧向量所在的位置？用当前位置x-上一帧x的位置得到变化量？
                    }
                }
            }
        }
        if (representation == RESIDUAL && cur_pos == pos_target) {

            uint8_t *bgr_data = (uint8_t*) bgr_arr->data;
            int32_t *res_data = (int32_t*) res_arr->data; // int32位？

            
            int stride_0 = height * width * 3;
            int stride_1 = width * 3;
            int stride_2 = 3;
            
            int y;

            for (y = 0; y < height; ++y) {
                int c, x, src_x, src_y, location, location2, location_src;
                int32_t tmp;
                for (x = 0; x < width; ++x) {
                    tmp = x * height * 2 + y * 2;  // (x,y)的一维索引
                    // 由运动矢量mv，求(x,y)对应的上一帧的位置(src_x, src_y)，
                    if (accumulate) {
                        src_x = accu_src[tmp];
                        src_y = accu_src[tmp + 1];
                    } else {
                        src_x = x - (*( (int32_t*)PyArray_GETPTR3(mv_arr, y, x, 0) ));
                        src_y = y - (*( (int32_t*)PyArray_GETPTR3(mv_arr, y, x, 1) ));
                    }
                    location_src = src_y * stride_1 + src_x * stride_2; // 上一帧的(src_x, src_y)像素点在residual中的一维索引

                    location = y * stride_1 + x * stride_2; // 当前帧的(x, y)像素点在residual中的一维索引

                    for (c = 0; c < 3; ++c) { // BGR三个像素
                        location2 = stride_0 + location; // BGR的索引，加stride_0是因为(2, w,h,3)中第二个(w,h,3)
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
    
    const int in_buffer_size=4096;     // FF_INPUT_BUFFER_PADDING_SIZE=32
    uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];  // 定长数组in_buffer[4096+32]，
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);  // in_buffer[4096:4096+32] = 0

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

    pFrame = av_frame_alloc();  // 分配内存，存放从AVPacket中解码出来的原始数据
    pFrameBGR = av_frame_alloc(); // 这个不放到create_and_load_bgr里面去创建，原因应该是最后返回这个，之后得free掉，否者在函数里面free掉就没法返回了

    uint8_t *buffer;

    av_init_packet(&packet);

    int *accu_src = NULL;
    int *accu_src_old = NULL;

    while (1) {

        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);  // 从给定流fp_in读取数据到in_buffer所指向的数组中
        // 每次读in_buffer_size=4096 bite，要读取的每个元素大小为1。
        if (cur_size == 0)  
            break;  
        cur_ptr=in_buffer;  // 这个设不设都没关系吧？为了保护in_buffer不被变化？
  
        while (cur_size>0){   // 缓存中还剩余数据，否者缓存中的数据全部解析后依然未能找到一个完整的包，那么继续从输入文件中读取数据到缓存
            // 将一个个AVPaket数据解析组成完整的一帧未解码的压缩数据
            int len = av_parser_parse2(  // 拿到AVPaket数据，从输入的数据流中分离出一帧一帧的压缩编码数据，输入为H.264、HEVC码流文件
                pCodecParserCtx, pCodecCtx,  
                &packet.data, &packet.size,  // packet.size维0，不断读取数据流，直到读取到一个完整帧，才置为1
                cur_ptr , cur_size ,  // 一次接收的数据包 和 本次接收数据包的长度
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);  

            cur_ptr += len;  
            cur_size -= len;

            if(packet.size==0)  // 还未解析到一个完整帧，继续继续解析缓存中剩余的码流。
                continue;  

            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {  // 读到I帧的时候
                ++cur_gop;
            }

            if (cur_gop == gop_target && cur_pos <= pos_target) {  //当前gop是要取的gop，且当前GOP里的帧位置比要取的帧位置低

                ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet); // 从packet中解码一帧视频数据到pFrame 
                // 解码前数据：AVPacket；解码后数据：AVFrame，解码成功得到一张图片got_picture归为1
                if (ret < 0) {  
                    printf("Decode Error.\n");  
                    return -1;  
                }
                int h = pFrame->height;
                int w = pFrame->width;

                // Initialize arrays. 
                if (! (*bgr_arr)) {
                    npy_intp dims[4];
                    dims[0] = 2; // 这个2什么意思？
                    dims[1] = h;
                    dims[2] = w;
                    dims[3] = 3;
                    *bgr_arr = PyArray_ZEROS(4, dims, NPY_UINT8, 0);  // 初始化I帧，4阶
                }

                if (representation == MV && ! (*mv_arr)) {
                    npy_intp dims[3];
                    dims[0] = h;
                    dims[1] = w;
                    dims[2] = 2;
                    *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0); // 初始化运动矢量
                }

                if (representation == RESIDUAL && ! (*res_arr)) {
                    npy_intp dims[3];
                    dims[0] = h;
                    dims[1] = w;
                    dims[2] = 3;

                    *mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
                    *res_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0); // 初始化运动矢量和帧差
                }

                if ((representation == MV ||
                     representation == RESIDUAL) && accumulate &&  // 累计模式
                    !accu_src && !accu_src_old) { // accu_src和accu_src_old均为空时
                    accu_src     = (int*) malloc(w * h * 2 * sizeof(int));
                    accu_src_old = (int*) malloc(w * h * 2 * sizeof(int));

                    for (size_t x = 0; x < w; ++x) {
                        for (size_t y = 0; y < h; ++y) {
                            accu_src_old[x * h * 2 + y * 2    ]  = x; // [0]=0, [2]=0, ..., [h*2 + (h-1)*2] = 1
                            accu_src_old[x * h * 2 + y * 2 + 1]  = y; // [1]=0, [3]=1, ..., [h*2 + (h-1)*2 + 1]=h-1
                            /* 第三维上是向量(x, y)。相当于是(w,h)的mv展开(x,y)坐标给平坦flat之后的一维数组：
                               [[[(0,0), (0,1), ... , (0,h-1)]
                                [[(1,0), (1,1), ... , (1,h-1)]
                                   ...
                                [[(w-1,0), (0,1), ... , (w-1,h-1)]]  将其flat后即为accu_src_old。
                            */
                        }
                    }
                    memcpy(accu_src, accu_src_old, h * w * 2 * sizeof(int));  // 这个有什么意义？
                }

                if (got_picture) { // 得到图片的时候

                    // 累计情形下的第一张帧差 or 非累计情形的要取的前一帧帧差 or 就是要取的
                    if ((cur_pos == 0              && accumulate  && representation == RESIDUAL) ||
                        (cur_pos == pos_target - 1 && !accumulate && representation == RESIDUAL) ||
                        cur_pos == pos_target) { 
                        create_and_load_bgr(  // 获得一帧BGR图像
                            pFrame, pFrameBGR, buffer, bgr_arr, cur_pos, pos_target); // buffer为空指针？
                            // pFrame为解码后的原始数据，pFrameBGR为空内存，bgr_arr也是分配好的空间内存
                    }

                    if (representation == MV ||  // 表示为运动矢量
                        representation == RESIDUAL) {  // 或者表示为帧差
                        AVFrameSideData *sd;
                        sd = av_frame_get_side_data(pFrame, AV_FRAME_DATA_MOTION_VECTORS); // 获取解码frame中的运动矢量
                        if (sd) {  // 提取到运动矢量
                            if (accumulate || cur_pos == pos_target) { // 累计 or 当前位置是目标位置
                                create_and_load_mv_residual(  // ie 在累计模式下，每帧都要算运动矢量和帧差；非累计模式下只计算目标位置的。
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
    while(1){  // 这个循环体的作用是什么？
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);  
        if (ret < 0) {  
            printf("Decode Error.\n");  
            return -1;  
        }  
        if (!got_picture) {
            break;  
        } else if (cur_gop == gop_target) { // 解码到图片，且当前gop为目标gop
            if ((cur_pos == 0 && accumulate) ||  // 累计且当前为I帧位置，条件与上面的循环一样
                (cur_pos == pos_target - 1 && !accumulate) ||
                cur_pos == pos_target) { 
                create_and_load_bgr(     // 为什么再次重复load bgr？
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
    uint8_t  in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE]; // 用来存从文件一次读取4096字节的数据缓存区数组，比要读取的多32字节
    // in_buffer[4096+32] 下面将in_buffer的4096位到4096+32位这32个地方置为零
    memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);  // 清空缓存区？

    uint8_t *cur_ptr;  
    int cur_size;  
    AVPacket packet;  

    avcodec_register_all();  // 注册了和编解码器有关的组件
  
    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG4);  // 获取解码器
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

        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);  // 每次从文件中读取4096字节到in_buffer中
        // cur_size为读取到的大小吧？
        if (cur_size == 0)  
            break;  
        cur_ptr=in_buffer;  // 当前指针指向从文件读取的缓存数据
  
        while (cur_size>0){  
  
            int len = av_parser_parse2(  // 解析数据获得一个Packet， 从输入的数据流中分离出一帧一帧的压缩编码数据
                pCodecParserCtx, pCodecCtx,  
                &packet.data, &packet.size,  
                cur_ptr , cur_size ,  
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);  

            cur_ptr += len;  // 
            cur_size -= len;  

            if(packet.size==0)  
                continue;  
            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I) {  // 读到的帧类型为I帧的时候，gop的数量+1。
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

    // load函数的输入的参数解析：filename:str, gop_target:int, pos_target:int, representation:int, accumulate:int,
    if (!PyArg_ParseTuple(args, "siiii", &filename,
        &gop_target, &pos_target, &representation, &accumulate)) return NULL;

    PyArrayObject *bgr_arr = NULL;   // BGR的arr的指针
    PyArrayObject *final_bgr_arr = NULL;  // 要返回的arr的指针
    PyArrayObject *mv_arr = NULL;  // motion vector arr的指针
    PyArrayObject *res_arr = NULL; // residual arr的指针

    if(decode_video(gop_target, pos_target,
                    &bgr_arr, &mv_arr, &res_arr, 
                    representation,
                    accumulate) < 0) {
        printf("Decoding video failed.\n");

        Py_XDECREF(bgr_arr); // 释放内存
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

    } else { // I_frame
        Py_XDECREF(mv_arr);
        Py_XDECREF(res_arr);

        npy_intp *dims_bgr = PyArray_SHAPE(bgr_arr);  // 获取bgr_arr的shape
        // 下面应该是从(3, h, w) -> (h, w, 3)
        int h = dims_bgr[1];
        int w = dims_bgr[2];

        npy_intp dims[3];
        dims[0] = h;
        dims[1] = w;
        dims[2] = 3;
        PyArrayObject *final_bgr_arr = PyArray_ZEROS(3, dims, NPY_UINT8, 0);

        int size = h * w * 3 * sizeof(uint8_t);
        memcpy(final_bgr_arr->data, bgr_arr->data + size, size); // 这里为什么加size？

        Py_XDECREF(bgr_arr);
        return final_bgr_arr;
    }
}



// 以下为打包

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
