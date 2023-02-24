
python3 submit/train.py --train_path ../user_data/train_data --model_path ../user_data/model_data

python3 submit/run.py --tcdata ../tcdata --model_path ../user_data/model_data

mv result.csv ../prediction_result/
