# /bin/bash

# beready data
cd /
# trai
python /code/bdata0.py
# test
python /code/bdata1.py
# vals
python /code/bdata2.py


# model
mkdir -p /root/.cache/torch/checkpoints
cp /model/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth

cd /mmdetection
mkdir -p data/coco
ln -s /myspace data/coco/annotations
ln -s /myspace /mmdetection/work_dirs

cp /model/coco.py mmdet/datasets/coco.py
cp /model/class_names.py mmdet/core/evaluation/class_names.py

python tools/train.py /model/faster_rcnn_r50_fpn_1x.py --gpu 0 --work_dir work_dirs



