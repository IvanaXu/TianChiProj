
# 
ver=v1.3.2
# 

rm -rf ../outs/$ver
mkdir -p ../outs/$ver

python run_$ver.py
cp post_generate_process_$ver.py ../outs/$ver/post_generate_process.py

cd ../outs/$ver
zip test_predictions.zip test_predictions.json post_generate_process.py
