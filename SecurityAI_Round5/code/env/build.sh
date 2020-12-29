cd SecurityAI_Round4/
mkdir env
cd env/
unzip cocoapi-master.zip
unzip mmdetection-master.zip
pip install mmcv_full-latest+torch1.3.0+cu100-cp36-cp36m-manylinux1_x86_64.whl --user
cd mmdetection-master
pip install -r requirements/build.txt --user
cd ../
cd cocoapi-master
cd pycocotools/
python setup.py build
python setup.py install --user
cd ../
cd ../
cd mmdetection-master
pip install --no-cache-dir -e . --user
