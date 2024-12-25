# 编译环境：
ubuntu22.04
## 安装GStreamer插件和编译工具：
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev build-essential
## 安装libtensorflowlite依赖
拷贝tflite-dist/include下的内容到/usr/local/include
拷贝tflite-dist/libs/linux_x64的内容到/usr/local/lib
更新系统库：
sudo ldconfig
# 编译插件：
编译前请按需修改filter.cpp文件中模型的加载路径（/root/zhanggf/gst_plugin_ai/models/wxltest.tflite）,改为实际路径，然后执行：
make
# 拷贝插件：
cp libgstaudiodenoise.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstaudiodenoise.so
# 测试命令：
gst-launch-1.0 filesrc location=a.pcm ! audiodenoise ! filesink location=a1.pcm