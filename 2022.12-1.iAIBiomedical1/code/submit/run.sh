
time python3 train.py --train_path ./train_data

time python3 run.py --tcdata /tcdata

head -n 5 result.csv
tail -n 5 result.csv

ls -l result.csv
md5sum result.csv
md5sum -c result.md5
date
