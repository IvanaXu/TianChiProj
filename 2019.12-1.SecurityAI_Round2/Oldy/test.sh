codep=/data/gproj/code/SecurityAI_Round2/code
outgp=/data/gproj/code/SecurityAI_Round2/out/testR

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

all=1280
b=80
for ((i=1; i<=$all/$b; i++)); do
{
    /data/soft/py3/bin/python test.py $i $b &
}
done
wait

cd $outgp
zip -r images.zip images/

cd $codep
