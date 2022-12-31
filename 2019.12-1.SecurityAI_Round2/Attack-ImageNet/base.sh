nvidia-docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/mmdetection:pytorch1.3-cuda10.1-py3
nvidia-docker run -ti -v /data:/data 76c152fbfd03 /bin/bash

# 修改apt-get源
cp /etc/apt/sources.list /etc/apt/sources.list.bak
cat > /etc/apt/sources.list << EOF
deb http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse
EOF

cat > /etc/apt/sources.list << EOF
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security multiverse   
EOF

apt-get update

###### mmdetection can
# ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
apt-get install libglib2.0-dev
# libglib2.0-dev : Depends: libpcre3-dev (>= 1:8.31) but it is not going to be installed
apt-get install libpcre3-dev
# libpcre3-dev : Depends: libpcre3 (= 1:8.31-2ubuntu2.3) but 2:8.38-3.1 is to be installed
apt-get install wget 
wget http://launchpadlibrarian.net/253825738/libpcre3-dev_8.31-2ubuntu2.3_amd64.deb

# ImportError: libSM.so.6: cannot open shared object file: No such file or directory
apt-get install libsm6

# ImportError: libXrender.so.1: cannot open shared object file: No such file or directory
apt-get install libxrender1

# ImportError: libXext.so.6: cannot open shared object file: No such file or directory
apt-get install libxext-dev
###### mmdetection can

pip install opencv-python pandas -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

cd /data/code/gproj/code/SecurityAI_Round2/code/Attack-ImageNet
python simple_attack_gpu.py

n=1;while(($n>0));do ll /root/gproj/code/SecurityAI_Round2/out/season2/t00001/images/*.png|wc -l;echo ------------------;sleep 60;done

# /opt/conda/conda-bld/pytorch_1570910687230/work/aten/src/THCUNN/ClassNLLCriterion.cu:57: void ClassNLLCriterion_updateOutput_no_reduce_kernel(int, THCDeviceTensor<Dtype, 2, int, DefaultPtrTraits>, THCDeviceTensor<long, 1, int, DefaultPtrTraits>, THCDeviceTensor<Dtype, 1, int, DefaultPtrTraits>, Dtype *, int, int) [with Dtype = float]: block: [0,0,0], thread: [3,0,0] Assertion `cur_target >= 0 && cur_target < n_classes` failed.
# Traceback (most recent call last):
#   File "simple_attack_gpu.py", line 108, in <module>
#     adv = attacker.attack(model, img.cuda(), label_true.cuda(), label_target.cuda())
#   File "/data/code/gproj/code/SecurityAI_Round2/code/Attack-ImageNet/attacker.py", line 78, in attack
#     best_loss[is_better] = loss[is_better]
# RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
## Note that we have modified the original dev.csv (the label has an offset of -1).


# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 10.1     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla P100-PCIE...  On   | 00000000:00:08.0 Off |                    0 |
# | N/A   52C    P0   140W / 250W |  10202MiB / 16280MiB |     99%      Default |
# +-------------------------------+----------------------+----------------------+
                                                                               
# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID   Type   Process name                             Usage      |
# |=============================================================================|
# |    0     15476      C   python                                      9939MiB |
# |    0     22411      C   /data/soft/py3/bin/python                    253MiB |
# +-----------------------------------------------------------------------------+



