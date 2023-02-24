# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 在构建镜像时安装依赖包
RUN sed -i -e 's/deb.debian.org/mirrors.huaweicloud.com/g' -e 's/security.debian.org/mirrors.huaweicloud.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y time && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
