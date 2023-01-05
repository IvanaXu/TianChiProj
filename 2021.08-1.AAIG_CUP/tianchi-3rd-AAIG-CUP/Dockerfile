#
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tianchi_antispam:v1
#
RUN pip uninstall -y tensorflow
#
RUN pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple tensorflow==2.5.0 tensorflow_addons==0.14.0
#
RUN apt-get update && apt-get install -y vim
#
ADD flink-conf.yaml /opt/flink-1.11.2/conf
#
ADD tf_main.py /root/tianchi_aiflow/workflows/tianchi_main
