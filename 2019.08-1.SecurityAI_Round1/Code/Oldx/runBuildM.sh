# echo $0 $1 
cd /data/gproj/code/SecurityAI_Round1/Code
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
git push origin master
echo "01/...Run..."

/data/soft/py3/bin/python runBuildM.py

