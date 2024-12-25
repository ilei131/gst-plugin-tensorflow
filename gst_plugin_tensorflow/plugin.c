#include <gst/gst.h>
#include "filter.h"

#ifndef PACKAGE
#define PACKAGE "audiodenoise"
#endif

static gboolean init (GstPlugin * plugin)
{
  if (!ai_filter_plugin_init (plugin))
    return FALSE;

  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    audiodenoise,
    "Audio denoise plugin",
    init, "1.0", "LGPL", "GStreamer", "https://kurento.openvidu.io/")