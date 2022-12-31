# 
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round2/Code
echo "00/...Git..."
git add *
git commit -m "update code $(date "+%Y%m%d%H%M%S")"
echo "01/...Run..."

n=00000078
o=/Volumes/ESSD/SecurityAI_Round2/Data/out/

rm -rf $o$n
mkdir $o$n $o$n/images

/Users/ivan/Desktop/ALL/Soft/python3/bin/python Brun.py $o$n

cd $o$n
zip -r images.zip images/
echo "END"
cd /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round2/Code



