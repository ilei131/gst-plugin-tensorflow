#ifndef _FILTER_H_
#define _FILTER_H_

#include <gst/audio/audio.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/c_api.h>

G_BEGIN_DECLS
#define FRAME_SIZE 480  // 对应16kHz采样率下的30ms（暂时保留，可能需要根据实际情况调整）
//#define BLOCK_SIZE 256
#define gru_size 128 * 32 * 2
//#define fft_out_size (BLOCK_SIZE / 2 + 1)
#define block_len 512
//#define block_shift 256
#define GST_TYPE_AUDIO_DENOISE (gst_audio_denoise_get_type())
G_DECLARE_FINAL_TYPE(GstAudioDenoise, gst_audio_denoise, GST, AUDIO_DENOISE, GstElement)

typedef struct _GstAudioDenoise GstAudioDenoise;

/* 插件对象结构 */
struct _GstAudioDenoise {
  GstElement parent;
  GstPad *sinkpad;  // 输入 Pad
  GstPad *srcpad;   // 输出 Pad
      // TensorFlow Lite
     // TensorFlow Lite 推理器和模型
  float in_buffer[block_len] = { 0 };
  float out_buffer[block_len] = { 0 };
  float states_1[gru_size] = { 0 };

  TfLiteTensor* input_details_1[2];
  const TfLiteTensor* output_details_1[4];

  TfLiteInterpreter* interpreter_dpcrn;
  TfLiteModel* model_dpcrn;
};

gboolean ai_filter_plugin_init (GstPlugin * plugin);

G_END_DECLS
#endif /* _AI_FILTER_H_ */