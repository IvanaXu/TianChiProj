# p01
clear

basep=~/Documents/project/
registry=registry.cn-beijing.aliyuncs.com
regurl=$registry/tianchi_ivanxu/aibiomedical1
ver=v1.0

#
echo ----------------------------------------------
docker pull registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10

#
echo ----------------------------------------------
cd $basep/code/submit
ls -l

#
echo ----------------------------------------------
echo build
docker images
docker rmi $regurl:$ver -f
docker build -t $regurl:$ver .
docker images

#
echo ----------------------------------------------
echo test images
docker run --name testi1 -v $basep/tcdata:/tcdata $regurl:$ver sh run.sh
docker ps -a
docker rm testi1 -f
docker ps -a

cat ~/Documents/P | docker login $registry --username=834400951@qq.com --password-stdin
docker push $regurl:$ver

echo $regurl:$ver
