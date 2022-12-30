#
basep=/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/CVmchar/codeS
regurl=cvmchar
ver=0.1

#
echo ----------------------------------------------
echo ----------------------------------------------
echo ----------------------------------------------
cd $basep
nvidia-docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/mmdetection:pytorch1.3-cuda10.1-py3

#
echo ----------------------------------------------
cd $basep/submit_step0
ls -l

#
echo ----------------------------------------------
echo build
docker images
docker rmi $regurl:$ver
docker build -t $regurl:$ver . 
docker images

#
echo ----------------------------------------------
echo test images

echo ----------------------------------------------
echo;echo;echo;echo;

# 
p1=/Users/ivan/Desktop/ALL/Data/CVmchar
p2=/tcdata

p3=/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/CVmchar/outs/myspace
p4=/myspace

docker run --name testi --shm-size 32G -v $p1:$p2 -v $p3:$p4 $regurl:$ver sh run.sh

echo;echo;echo;echo;
echo ----------------------------------------------

docker ps -a
docker rm testi
docker ps -a
echo ----------------------------------------------
echo ----------------------------------------------
echo ----------------------------------------------



