# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

##
RUN apt-get update
RUN apt-get install libglib2.0-dev libsm6 libxrender1 libxext-dev -y 

## 在构建镜像时安装依赖包
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements/build.txt
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements/runtime.txt

##
WORKDIR code/train/env/
RUN pip install mmcv_full-2.3.0+torch1.3.0+cu100-cp36-cp36m-manylinux1_x86_64.whl
WORKDIR mmdetection-master
RUN pip install -r requirements/build.txt
WORKDIR ../cocoapi-master/pycocotools/
RUN python setup.py build
RUN python setup.py install --user
WORKDIR ../../mmdetection-master
RUN pip install --no-cache-dir -e . --user

WORKDIR /
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]


