# 
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round2/Code
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "01/...Run..."

/Users/ivan/Desktop/ALL/Soft/python3/bin/python trun.py /

n=00000073
o=/Volumes/ESSD/SecurityAI_Round2/Data/out/

rm -rf $o$n
mkdir $o$n $o$n/images

all=36
b=3
for ((i=1; i<=$all/$b; i++)); do
{
    /Users/ivan/Desktop/ALL/Soft/python3/bin/python Arun.py $o$n $i $b &
}
done
wait

cd $o$n
zip -r images.zip images/
echo "END"
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round2/Code



