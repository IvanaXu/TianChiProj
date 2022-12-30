# echo $0 $1 
cd /data/gproj/code/SecurityAI_Round1/Code

echo "00/...Git..."
echo "--------------------------------GIt add"
git add -A
echo "--------------------------------GIt commit"
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "--------------------------------GIt push"
git push origin master
sleep 1

echo "01/...Run..."

n=0000000t
o=/data/gproj/code/SecurityAI_Round1/Out/

rm -rf $o$n
rm -rf $o/result
mkdir $o$n $o$n/images $o$n/timages
/data/soft/py3/bin/python runC05.py $o$n -94.24 -94.07 -94.10

cd $o$n
zip -r images.zip images/
echo "END"
cd /data/gproj/code/SecurityAI_Round1/Code

cat $o/result

