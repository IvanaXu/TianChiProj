> Build, Ship and Run Any App, Anywhere.

根据个人习惯，以下采用叙事格式为：
``` shell
code
# output
```
修订如下：
> C001 调试用途，不需提供；
>
> C002 此处省略代码N行，仅输出Hello world，可得分30；
>
> C003 注意：/tcdata/num_list.csv无表头；
>
> C004 注意：TOP10 若包含重复值；

#### 一、先安装 Docker
以下示例环境：CentOS
``` shell
yum install -y docker
# ...
```
``` shell
service docker restart
# Redirecting to /bin/systemctl restart docker.service
``` 

``` shell
docker ps
# CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

#### 二、开通其容器镜像
<i>**<font color=orangered>仓库地域选择上海</font>**</i>

#### 三、构建镜像并推送
1、文件架构
``` shell
ls -l
# -rwxr-xr-x 1 root root 1381 Dec 27 05:29 rcmd.sh
# drwxr-xr-x 2 root root 4096 Dec 27 08:54 tcdata
# drwxr-xr-x 2 root root 4096 Dec 27 04:32 tianchi_submit_demo
``` 

``` shell
# C001 调试用途，不需提供
ls -l tcdata/
# -rw-r--r-- 1 root root 51 Dec 27 05:14 num_list.csv
``` 

``` shell
ls -l tianchi_submit_demo/
# -rw-r--r-- 1 root root 384 Dec 27 04:33 Dockerfile
# -rw-r--r-- 1 root root 438 Dec 27 05:27 hello_world.py
# -rw-r--r-- 1 root root  90 Dec 27 05:29 result.json
# -rw-r--r-- 1 root root  21 Dec 27 04:33 run.sh
```

2、tcdata
如题，任务文件在/tcdata/num_list.csv，以下会创建示例并ln -s软链接，用于代码调试。
注意：/tcdata/num_list.csv无表头
``` shell
cd /  
rm -rf tcdata  
ln -s $basep/tcdata tcdata  
ls -l tcdata
# lrwxrwxrwx 1 root root 35 Dec 27 10:15 tcdata -> /data/gproj/code/LearnDocker/tcdata

cd $basep  
cat /tcdata/num_list.csv
# 102
# 6
# 11
# 4310
# 2
# ...
```

3、tianchi_submit_demo
 - Dockerfile
``` shell
cat Dockerfile 
```
```
# Base Images  
## 从天池基础镜像构建  
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3  
  
## 把当前文件夹里的文件构建到镜像的根目录下  
ADD . /  
  
## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）  
WORKDIR /  
  
## 镜像启动后统一执行 sh run.sh  
CMD ["sh", "run.sh"]
```

 - hello_world. py
``` shell
cat hello_world.py
```
``` python
# hello_world.py
import json
# ...
# C002 此处省略代码N行，仅输出Hello world，可得分30
# ...
result = {
    "Q1": "Hello world",
    "Q2": 100,
    # C004 注意：TOP10 若包含重复值
    "Q3": []
}
with open("result.json", "w") as f:
    json.dump(result, f)
```

 - result. json
``` shell
cat result.json
# 
```

 - run. sh
``` shell
cat run.sh
# python hello_world.py
```

4、push
rcmd.sh作为本次主运行程序，在上述兼备的情况下，
并自填XXXXXXX部分。

```shell
# XXXXXXX 即仓库链接  
basep=/data/gproj/code/LearnDocker  
regurl=registry.cn-XXXXXXX
ver=1.0  
  
#  
echo ----------------------------------------------  
bak  
echo 00.base  
cd $basep  
  
#  
echo ----------------------------------------------  
echo 01.pull  
docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/python:3  
  
#  
echo ----------------------------------------------  
echo 02.check  
ls -l tianchi_submit_demo  
  
#  
echo ----------------------------------------------  
echo 03.test data  
cd /  
rm -rf tcdata  
ln -s $basep/tcdata tcdata  
ls -l tcdata  
  
cd $basep  
cat /tcdata/num_list.csv  
echo  
  
#  
echo ----------------------------------------------  
echo 04.test code  
cd $basep/tianchi_submit_demo  
ls -l  
sh run.sh  
cat result.json  
echo  
  
#  
echo ----------------------------------------------  
echo 05.build  
docker images  
docker rmi $regurl:$ver  
docker build -t $regurl:$ver .  
docker images  
  
#  
echo ----------------------------------------------  
echo 06.test images  
docker run --name testi $regurl:$ver sh run.sh  
docker ps -a  
docker rm testi  
docker ps -a  
  
# XXXXXXX 即账号  
echo ----------------------------------------------  
echo 07.push  
docker login --username=XXXXXXX registry.cn-shanghai.aliyuncs.com  
docker push $regurl:$ver  
  
#  
echo ----------------------------------------------  
echo $regurl:$ver  
echo 08.END  
  
#  
echo ----------------------------------------------
```

可得：
```
# 00.base  
# ----------------------------------------------  
# 01.pull  
# Trying to pull repository registry.cn-shanghai.aliyuncs.com/tcc-public/python ...  
# 3: Pulling from registry.cn-shanghai.aliyuncs.com/tcc-public/python  
# Digest: sha256:6268ecdce5f04d54bd411cba64e49c714589e53ae482a49c6c12eaf91a5d0425  
# Status: Image is up to date for registry.cn-shanghai.aliyuncs.com/tcc-public/python:3  
# ----------------------------------------------  
# 02.check  
# total 16  
# -rw-r--r-- 1 root root 384 Dec 27 04:33 Dockerfile  
# -rw-r--r-- 1 root root 438 Dec 27 05:27 hello_world.py  
# -rw-r--r-- 1 root root  90 Dec 27 12:20 result.json  
# -rw-r--r-- 1 root root  21 Dec 27 04:33 run.sh  
# ----------------------------------------------  
# 03.test data  
# lrwxrwxrwx 1 root root 35 Dec 27 12:44 tcdata -> /data/gproj/code/LearnDocker/tcdata  
# 102  
# 6  
# 11  
# 4310  
# 2  
# 6  
# 11  
# 43  
# 24  
# 45  
# 465  
# 656  
# 45  
# 4534  
# 5  
# 353  
# ----------------------------------------------  
# 04.test code  
# total 16  
# -rw-r--r-- 1 root root 384 Dec 27 04:33 Dockerfile  
# -rw-r--r-- 1 root root 438 Dec 27 05:27 hello_world.py  
# -rw-r--r-- 1 root root  90 Dec 27 12:20 result.json  
# -rw-r--r-- 1 root root  21 Dec 27 04:33 run.sh  
# 000 10618 [4534, 4310, 656, 465, 353, 102, 45, 45, 43, 24]  
# 001 Hello world  
# {"Q1": "Hello world", "Q2": 10618, "Q3": [4534, 4310, 656, 465, 353, 102, 45, 45, 43, 24]}  
# ----------------------------------------------  
# 05.build  
# REPOSITORY                                                                      TAG                 IMAGE ID            CREATED             SIZE  
# registry.cn-shanghai.aliyuncs.com/XXXXXXX                                       1.0                 5f20a9898cd9        23 minutes ago      929 MB  
# registry.cn-shanghai.aliyuncs.com/tcc-public/python                             3                   a4cc999cf2aa        7 months ago        929 MB  
# Untagged: registry.cn-shanghai.aliyuncs.com/XXXXXXX:1.0  
# Untagged: registry.cn-shanghai.aliyuncs.com/XXXXXXX@sha256:40cd6610a72f8a2da0d628ed84233380354df5f9f521627f74867ebb76d92bbb  
# Deleted: sha256:5f20a9898cd9e684106e9f6f3469e4536d1edd501b69fefa813556bea8d9534d  
# Deleted: sha256:7a4456df53e8c792edf64a412edd6c25596bfed56f9680df38a9ba6a8fabd631  
# Deleted: sha256:10f5cf32068de61f4ec19982795c0454d73c98f708a102c75c5fff033e92ec13  
# Deleted: sha256:77d51fafa8f15b5ef9789a81046584ce8838edf3c486ded60e17f99fb8193e29  
# Sending build context to Docker daemon  5.12 kB  
# Step 1/4 : FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3  
#  ---> a4cc999cf2aa  
# Step 2/4 : ADD . /  
#  ---> d0b96ac18123  
# Removing intermediate container 70ca1e187044  
# Step 3/4 : WORKDIR /  
#  ---> edfdb00575b1  
# Removing intermediate container 932e080ec58b  
# Step 4/4 : CMD sh run.sh  
#  ---> Running in 56063a448ead  
#  ---> 38843989b200  
# Removing intermediate container 56063a448ead  
# Successfully built 38843989b200  
# REPOSITORY                                                                      TAG                 IMAGE ID            CREATED                  SIZE  
# registry.cn-shanghai.aliyuncs.com/XXXXXXX                                       1.0                 38843989b200        Less than a second ago   929 MB  
# docker.io/redis                                                                 latest              dcf9ec9265e0        4 weeks ago              98.2 MB  
# registry.cn-shanghai.aliyuncs.com/tcc-public/python                             3                   a4cc999cf2aa        7 months ago             929 MB  
# ----------------------------------------------  
# 06.test images  
# Traceback (most recent call last):  
#   File "hello_world.py", line 5, in <module>  
#     with open("/tcdata/num_list.csv", "r") as f:  
# FileNotFoundError: [Errno 2] No such file or directory: '/tcdata/num_list.csv'  
# CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
# 0ad731344d71        registry.cn-shanghai.aliyuncs.com/XXXXXXX:1.0 "sh run.sh" 1 second ago        Exited (1) Less than a second ago                       testi  
# testi  
# CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
# ----------------------------------------------  
# 07.push  
# Password:  
# Login Succeeded  
# The push refers to a repository [registry.cn-shanghai.aliyuncs.com/XXXXXXX]  
# 56850b2c6636: Pushed  
# 2633623f6cf4: Layer already exists  
# 5194c23c2bc2: Layer already exists  
# 69bbfe9f27d4: Layer already exists  
# 2492a3be066b: Layer already exists  
# 910d7fd9e23e: Layer already exists  
# 4230ff7f2288: Layer already exists  
# 2c719774c1e1: Layer already exists  
# ec62f19bb3aa: Layer already exists  
# f94641f1fe1f: Layer already exists  
# 1.0: digest: sha256:e2206fd5bb5e74b3f926e165449d26ff6e60424a08c450d36a2efb1956b2ba89 size: 2425  
# ----------------------------------------------  
# registry.cn-shanghai.aliyuncs.com/XXXXXXX:1.0  
# 08.END
```
#### 四、提交
Nice！

> 相关链接 https://github.com/IvanaXu/TianChiProj/tree/master/LDocker
>
> 容器链接 docker pull docker.pkg.github.com/ivanaxu/tianchiproj/ldocker:1.0
>



