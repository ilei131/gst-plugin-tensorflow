#include "filter.h"
#include <gst/gst.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <complex>
#ifndef MIN
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#include "pocketfft_hdronly.h"
#include "sin.h"
#include <tensorflow/lite/c/c_api.h>

#define PLUGIN_NAME "audiodenoise"

G_DEFINE_TYPE(GstAudioDenoise, gst_audio_denoise, GST_TYPE_ELEMENT);

//#define block_len		512
#define block_shift		256
#define fft_out_size    (block_len / 2 + 1)
using namespace pocketfft;
using namespace std;
typedef complex<double> cpx_type;

typedef unsigned char BYTE;
typedef std::complex<double> Complex;

#define S16_INPUT_RAW

// 将32位浮点格式的音频数据转换为16位整型格式，模拟16kHz采样率
void f32_16khz_to_s16_16khz(float* in, short* out, int count)
{
    for (int i = 0; i < 1; i++) // 每次处理BLOCK_SIZE个样本
    {
        for (int j = 0; j < count; j++)
            out[j] = in[j] * 32767.f; // 浮点数值转为整数值（乘以缩放因子）

        in += count; // 移动输入指针
        out += count; // 移动输出指针
    }
}

// 计算傅里叶变换结果的幅值和相位，存储在输入数组中
void calc_mag_phase(vector<cpx_type> fft_res, float* inp, int count)
{
    for (int i = 0; i < count; i++)
    {
        inp[i * 3] = fft_res[i].real(); // 实部
        inp[i * 3 + 1] = fft_res[i].imag(); // 虚部
        inp[i * 3 + 2] = 2 * log(sqrtf(fft_res[i].real() * fft_res[i].real() + fft_res[i].imag() * fft_res[i].imag()));// 幅值（对数形式）
    }
}

// 初始化TensorFlow Lite模型和解释器
void tflite_create(GstAudioDenoise *self)
{

    self->model_dpcrn = TfLiteModelCreateFromFile("/root/zhanggf/gst_plugin_ai/models/wxltest.tflite");// 加载tflite模型

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1); // 设置单线程运行

    self->interpreter_dpcrn = TfLiteInterpreterCreate(self->model_dpcrn, options);// 创建解释器
    if (self->interpreter_dpcrn == nullptr) {
        printf("Failed to create interpreter");
        return;
    }

    if (TfLiteInterpreterAllocateTensors(self->interpreter_dpcrn) != kTfLiteOk) {
        printf("Failed to allocate tensors!");
        return;
    }
    // 获取输入和输出张量
    self->input_details_1[0] = TfLiteInterpreterGetInputTensor(self->interpreter_dpcrn, 0);
    self->input_details_1[1] = TfLiteInterpreterGetInputTensor(self->interpreter_dpcrn, 1);
    self->output_details_1[0] = TfLiteInterpreterGetOutputTensor(self->interpreter_dpcrn, 0);
    self->output_details_1[1] = TfLiteInterpreterGetOutputTensor(self->interpreter_dpcrn, 1);
    self->output_details_1[2] = TfLiteInterpreterGetOutputTensor(self->interpreter_dpcrn, 2);
    self->output_details_1[3] = TfLiteInterpreterGetOutputTensor(self->interpreter_dpcrn, 3);
}

void tflite_destroy(GstAudioDenoise *self)
{
    TfLiteModelDelete(self->model_dpcrn);
}

// 执行推理
void tflite_infer(GstAudioDenoise *self)
{

    float inp[(block_len / 2 + 1) * 3] = { 0 }; // 存储幅值和相位数据
    float estimated_block[block_len];

    double fft_in[block_len];
    vector<cpx_type> fft_res(block_len);

    // 定义FFT变换的形状和步幅
    shape_t shape;
    shape.push_back(block_len);
    shape_t axes;
    axes.push_back(0);
    stride_t stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));

    // 应用窗函数
    for (int i = 0; i < block_len; i++)
    {
        fft_in[i] = self->in_buffer[i] * win_sin[i];
    }
    // 进行FFT变换
    r2c(shape, stridel, strideo, axes, FORWARD, fft_in, fft_res.data(), 1.0);
    // 计算幅值和相位
    calc_mag_phase(fft_res, inp, fft_out_size);
    // 设置模型输入
    memcpy(self->input_details_1[0]->data.f, inp, fft_out_size * 3 * sizeof(float));
    memcpy(self->input_details_1[1]->data.f, self->states_1, gru_size * sizeof(float));
    // 调用模型
    if (TfLiteInterpreterInvoke(self->interpreter_dpcrn) != kTfLiteOk) {
        printf("Error invoking detection model");
    }

    // 获取模型输出并调整FFT结果
    float* out_mask = self->output_details_1[0]->data.f;
    float* out_cos = self->output_details_1[1]->data.f;
    float* out_sin = self->output_details_1[2]->data.f;
    memcpy(self->states_1, self->output_details_1[3]->data.f, gru_size * sizeof(float));

    for (int i = 0; i < fft_out_size; i++) {
        fft_res[i] = complex<double>{ fft_res[i].real() * out_mask[i] * out_cos[i] - fft_res[i].imag() * out_mask[i] * out_sin[i], fft_res[i].real() * out_mask[i] * out_sin[i] + fft_res[i].imag() * out_mask[i] * out_cos[i] };
    }
    // 逆FFT
    c2r(shape, strideo, stridel, axes, BACKWARD, fft_res.data(), fft_in, 1.0);
    // 应用窗函数并更新输出缓冲区
    for (int i = 0; i < block_len; i++)
        estimated_block[i] = (fft_in[i] / block_len) * win_sin[i];
    memmove(self->out_buffer, self->out_buffer + block_shift, (block_len - block_shift) * sizeof(float));
    memset(self->out_buffer + (block_len - block_shift), 0, block_shift * sizeof(float));
    for (int i = 0; i < block_len; i++)
        self->out_buffer[i] += estimated_block[i];
}

void trg_denoise(float* samples, float* out, int sampleCount, GstAudioDenoise *self)
{
    int num_blocks = sampleCount / block_shift;

    for (int idx = 0; idx < num_blocks; idx++)
    {
        memmove(self->in_buffer, self->in_buffer + block_shift, (block_len - block_shift) * sizeof(float));
        memcpy(self->in_buffer + (block_len - block_shift), samples, block_shift * sizeof(float));
        tflite_infer(self);
        memcpy(out, self->out_buffer, block_shift * sizeof(float));
        samples += block_shift;
        out += block_shift;
    }
}

void s16_16khz_to_f32_16khz(short* in, float* out, int count)
{
    for (int j = 0; j < count; j++)
        out[j] = in[j] / 32767.f;
}

void floatTobytes(float* data, BYTE* bytes, int dataLength)
{
    int i;
    size_t length = sizeof(float) * dataLength;
    BYTE* pdata = (BYTE*)data;
    for (i = 0; i < length; i++)
    {
        bytes[i] = *pdata++;
    }
    return;
}

float bytesToFloat(BYTE* bytes)
{
    return *((float*)bytes);
}

void shortToByte(short* data, BYTE* bytes, int dataLength)
{
    for (int i = 0; i < dataLength; i++) {
        bytes[i * 2] = (BYTE)(0xff & data[i]);
        bytes[i * 2 + 1] = (BYTE)((0xff00 & data[i]) >> 8);
    }
    return;
}

short bytesToShort(BYTE* bytes)
{
    short addr = bytes[0] & 0xFF;
    addr |= ((bytes[1] << 8) & 0xFF00);
    return addr;
}

void ByteToChar(BYTE* bytes, char* chars, unsigned int count) {
    for (unsigned int i = 0; i < count; i++)
        chars[i] = (char)bytes[i];
}

// 音频降噪处理函数声明
static GstFlowReturn gst_audio_denoise_chain(GstPad *pad, GstObject *parent, GstBuffer *buf);

// 代码初始化
static void gst_audio_denoise_class_init(GstAudioDenoiseClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    gst_element_class_set_static_metadata(element_class,
        "AudioDenoise", "Filter/Audio",
        "Reduces background noise from audio streams",
        "Zhanggf <openwit@qq.com>");
}

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("audio/x-raw, format=(string)S16LE, rate=(int)16000, channels=(int)1"));

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("audio/x-raw, format=(string)S16LE, rate=(int)16000, channels=(int)1"));

// 初始化 TensorFlow Lite 相关资源
static gboolean gst_audio_denoise_init_model(GstAudioDenoise *self) {
    tflite_create(self);
    return TRUE;
}

// 初始化插件对象
static void gst_audio_denoise_init(GstAudioDenoise *self) {
    self->sinkpad = gst_pad_new_from_static_template(&sink_factory, "sink");
    gst_pad_set_chain_function(self->sinkpad, gst_audio_denoise_chain);
    gst_element_add_pad(GST_ELEMENT(self), self->sinkpad);

    self->srcpad = gst_pad_new_from_static_template(&src_factory, "src");
    gst_element_add_pad(GST_ELEMENT(self), self->srcpad);

    if (!gst_audio_denoise_init_model(self)) {
        g_printerr("Failed to initialize model.\n");
        gst_object_unref(self);
    }
}

// 输入音频帧处理和 TensorFlow Lite 推理
static GstFlowReturn gst_audio_denoise_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer) {
    GstAudioDenoise *self = GST_AUDIO_DENOISE(parent);

    // 获取音频数据
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
        g_printerr("Failed to map buffer.\n");
        return GST_FLOW_ERROR;
    }

    BYTE* inBuffer_byte_16k = (BYTE *)map.data;
    gsize inSampleCount = map.size;
    uint64_t BLOCK_SIZE = inSampleCount / 2;
    short inBuffer_s16_16k[BLOCK_SIZE];
    for (int i = 0, j = 0; i < inSampleCount; i = i + 2)
    {
        inBuffer_s16_16k[j] = bytesToShort(inBuffer_byte_16k);
        inBuffer_byte_16k = inBuffer_byte_16k + 2;
        j++;
    }

    float f32_sample[BLOCK_SIZE];
    float outBuffer_f32_16khz[BLOCK_SIZE];
    short out_s16_16khz[BLOCK_SIZE];
    BYTE out_bytes[BLOCK_SIZE * 2];

    s16_16khz_to_f32_16khz(inBuffer_s16_16k, f32_sample, BLOCK_SIZE);
    trg_denoise(f32_sample, outBuffer_f32_16khz, BLOCK_SIZE, self);
    f32_16khz_to_s16_16khz(outBuffer_f32_16khz, out_s16_16khz, BLOCK_SIZE);
    shortToByte(out_s16_16khz, out_bytes, BLOCK_SIZE);

    inBuffer_byte_16k = (BYTE *)map.data;
   // 将处理后的音频数据写回原始缓冲区
   for (size_t i = 0; i < inSampleCount; ++i) {
       inBuffer_byte_16k[i] = out_bytes[i];
   }

   gst_buffer_unmap(buffer, &map);

   // 将新缓冲区推送到下游元素
   return gst_pad_push(self->srcpad, buffer);
}

// 插件注册函数
gboolean ai_filter_plugin_init(GstPlugin *plugin) {
   return gst_element_register(plugin, PLUGIN_NAME, GST_RANK_NONE, GST_TYPE_AUDIO_DENOISE);
}