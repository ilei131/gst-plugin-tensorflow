最近有个调研性需求，在媒体服务器测引入机器学习相关模型进行降噪处理。为此编写了一个Gstreamer插件用于降噪测试，采用开源模型，降噪效果不错，但是资源占用率较高，后续如果采用服务侧降噪需要进一步进行轻量化处理。
有编写Gstreamer插件需求的同学可以参考。完整代码：[https://github.com/ilei131/gst-plugin-tensorflow](https://github.com/ilei131/gst-plugin-tensorflow)

<!--more-->
## 编译环境：
Ubuntu22.04（Windows10 Hyper-V虚拟机）
## 安装GStreamer插件和编译工具：
```
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev build-essential
```
## 安装libtensorflowlite依赖
拷贝tflite-dist/include下的内容到/usr/local/include
拷贝tflite-dist/libs/linux_x64的内容到/usr/local/lib
更新系统库：
```
sudo ldconfig
```
## 编译插件：
编译前请按需修改filter.cpp文件中模型的加载路径，改为实际路径，然后执行：
```
make
```
## 拷贝插件：
```
cp libgstaudiodenoise.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstaudiodenoise.so
```
## 测试命令：
```
gst-launch-1.0 filesrc location=source.pcm ! audiodenoise ! filesink location=target.pcm
```


项目使用的开源模型为:
> [https://github.com/FragrantRookie/Realtime_Skip_Dpcrn_Tflite_Denoise](https://github.com/FragrantRookie/Realtime_Skip_Dpcrn_Tflite_Denoise)