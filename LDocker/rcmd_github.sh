#
basep=/home/ivan/data/code/test/LearnDocker
regurl=docker.pkg.github.com/ivanaxu/tianchiproj/ldocker
ver=1.0

#
echo ----------------------------------------------
echo 00.base
cd $basep

#
echo ----------------------------------------------
sudo docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

#
echo ----------------------------------------------
cd $basep/tianchi_submit_demo
ls -l

#
echo ----------------------------------------------
sudo docker images
sudo docker rmi $regurl:$ver
sudo docker build -t $regurl:$ver .
sudo docker images

#
echo ----------------------------------------------
sudo docker run --name testi -v /home/ivan/data/code/test/LearnDocker/tcdata:/tcdata $regurl:$ver sh run.sh
sudo docker ps -a
sudo docker rm testi
sudo docker ps -a

#
echo ----------------------------------------------
## https://github.com/settings/tokens/new 
export GH_TOKEN="YOUR_TOKEN_HERE"
sudo echo $GH_TOKEN|docker login docker.pkg.github.com -u IvanaXu --password-stdin 
sudo docker push $regurl:$ver
# docker.pkg.github.com/ivanaxu/tianchiproj/ldocker:1.0



