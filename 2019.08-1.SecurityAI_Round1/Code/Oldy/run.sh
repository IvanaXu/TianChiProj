# echo $0 $1 
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Code
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "01/...Run..."


n=0000000t
o=/Volumes/ESSD/SecurityAI_Round1/Data/out/

rm -rf $o$n
rm -rf /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result
mkdir $o$n $o$n/images
/Users/ivan/Desktop/ALL/Soft/python3/bin/python runC05.py $o$n
# /Users/ivan/Desktop/ALL/Soft/python3/bin/python runC05.py $o$n $1

cd $o$n
# zip -r images.zip images/
echo "END"
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Code

cat /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result

# /Users/ivan/Desktop/ALL/Soft/python3/bin/python test.py
# n=0;while ((n<100));do ((n++));echo $n;sh run.sh $n;done|grep -E "dets12|Result"

