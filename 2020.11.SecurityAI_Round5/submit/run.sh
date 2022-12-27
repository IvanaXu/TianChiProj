cd /

# Python
echo 1
python -c "import cv2"
python -c "import mmdet"

# Train
echo 2
# sh code/train.sh

# Tests
echo 3
sh code/run.sh
