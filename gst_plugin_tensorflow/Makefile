# 设置编译器
CC          = g++
CXX         = g++
PLUGIN_NAME = audiodenoise

# 获取所有.c和.cpp文件
C_SOURCES   = $(wildcard *.c)
CPP_SOURCES = $(wildcard *.cpp)
 
# 获取所有.h文件
HEADERS = $(wildcard *.h)

# 将.c和.cpp文件名转换为对应的.o文件名
C_OBJECTS   = $(C_SOURCES:.c=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
OBJ         = $(C_OBJECTS) $(CPP_OBJECTS)

# 包含路径和库路径
INCLUDE_DIRS = -I//usr/include/x86_64-linux-gnu/c++/11 -I/usr/include/c++/11 -I/usr/local/include -Iinclude
LIB_DIRS     = -L/usr/local/lib -Llibs

CFLAGS   = -fPIC `pkg-config --cflags gstreamer-1.0` $(INCLUDE_DIRS)
CXXFLAGS = -std=c++11 -fPIC `pkg-config --cflags gstreamer-1.0` $(INCLUDE_DIRS)
LIBS     = `pkg-config --libs gstreamer-1.0` $(LIB_DIRS) -ltensorflowlite_c
TARGET   = libgst$(PLUGIN_NAME).so

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -shared -o $@ $^ $(LIBS)

# 对.c文件使用$(CC)进行编译
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

# 对.cpp文件使用$(CXX)进行编译
%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)