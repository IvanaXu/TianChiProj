cd tianchi-3rd-AAIG-CUP

docker build -t registry.cn-shanghai.aliyuncs.com/tianchi_ivanxu/tianchi_antispam:0.1 .

docker login registry.cn-shanghai.aliyuncs.com

docker push registry.cn-shanghai.aliyuncs.com/tianchi_ivanxu/tianchi_antispam:0.1
