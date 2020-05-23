#
basep=/data/gproj/code/LearnDocker
regurl=registry.cn-shanghai.aliyuncs.com/tianchi_ivanxu/tianchi_ivanxu_20191227_c001
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

#
echo ----------------------------------------------
echo 07.push
docker login --username=XXX registry.cn-shanghai.aliyuncs.com
docker push $regurl:$ver

#
echo ----------------------------------------------
echo $regurl:$ver
echo 08.END

#
echo ----------------------------------------------



