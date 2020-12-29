# Train
# data
cd /
python code/clean_data_code.py train

# model
mmdetn=1
mmdet=code/train/env/mmdetection-master

mkdir -p /root/.cache/torch/checkpoints
cp user_data/model_data/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth

mkdir -p $mmdet/data/coco
ln -s /user_data/tmp_data/work $mmdet/data/coco/annotations
ln -s /user_data/tmp_data/work $mmdet/work_dirs

cp code/train/coco$mmdetn.py $mmdet/mmdet/datasets/coco.py
cp code/train/class_names$mmdetn.py $mmdet/mmdet/core/evaluation/class_names.py

# 示例total_epochs = 1, 复现需修改
python $mmdet/tools/train.py code/train/faster_rcnn_r50_fpn_1x_coco.py --gpus 1 --work-dir $mmdet/work_dirs



