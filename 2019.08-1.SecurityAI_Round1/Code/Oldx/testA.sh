codep=/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round1/Code
outgp=/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round1/Out/testRA

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
echo score > $outgp/score

cd $codep
/Users/ivan/Desktop/ALL/Soft/python3/bin/python testA.py

cd $outgp
zip -r images$(date +%s%N).zip images/

cd $codep



