# echo $0 $1
codep=/data/gproj/code/SecurityAI_Round1/Code
outgp=/data/gproj/code/SecurityAI_Round1/Out/runBadtoModel

cd $codep
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
git push origin master
echo "01/...Run..."

rm -rf $outgp
mkdir $outgp $outgp/images $outgp/temp

cd $codep
all=800
b=80
for ((i=1; i<$all/$b+1; i++)); do
{
  /data/soft/py3/bin/python runBadtoModel.py $i $b &
}
done

wait

cd $outgp
zip -r images.zip images/

cd $codep

cat $outgp/temp/r* > $outgp/lresult
cat $outgp/lresult
wc -l $outgp/lresult
ls -l $outgp/images/*.jpg|wc -l

# nohup sh runBadtoModel.sh > runBadtoModel.log 2>&1 &

