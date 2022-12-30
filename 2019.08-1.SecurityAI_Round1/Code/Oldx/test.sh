codep=/data/gproj/code/SecurityAI_Round1/Code
outgp=/data/gproj/code/SecurityAI_Round1/Out/testR

cd $codep
echo "00/...Git..."
echo "--------------------------------GIt add"
git add --all .
echo "--------------------------------GIt commit"
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "--------------------------------GIt push"
git push origin master
sleep 1

echo "02/Train Start"
rm -rf $outgp
mkdir $outgp $outgp/images

cd $codep
/data/soft/py3/bin/python test.py

cd $outgp
# zip -r images.zip images/

cd $codep
