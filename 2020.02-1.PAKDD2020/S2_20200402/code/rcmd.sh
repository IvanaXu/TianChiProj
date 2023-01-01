#
basep=/data/code/gproj/code/PAKDD2020/code_s2
regurl=registry.cn-shanghai.aliyuncs.com/tianchi_ivanxu/pakdd2020_s2_c
ver=1.0

#
echo ----------------------------------------------
cd $basep
docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

#
echo ----------------------------------------------
cd $basep/submit_v0
rm -rf predictions.csv
rm -rf result.zip
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
p1=/data/code/gproj/code/PAKDD2020/project/data
p2=/tcdata
docker run --name testi -v $p1:$p2 $regurl:$ver sh run.sh
docker ps -a
docker rm testi
docker ps -a

#
echo ----------------------------------------------
echo push
docker login registry.cn-shanghai.aliyuncs.com
docker push $regurl:$ver

#
echo ----------------------------------------------
echo $regurl:$ver
echo 08.END

#
echo ----------------------------------------------



