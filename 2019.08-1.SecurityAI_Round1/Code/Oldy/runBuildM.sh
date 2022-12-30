# echo $0 $1 
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Code
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "01/...Run..."

/Users/ivan/Desktop/ALL/Soft/python3/bin/python runBuildM.py

