
n=0000000t
o=/Volumes/ESSD/SecurityAI_Round1/Data/out/

l=90
h=100

for ((i0=$l; i0<=$h; i0++)); do
{
    for ((i1=$l; i1<=$h; i1++)); do
    {
        for ((i2=$l; i2<=$h; i2++)); do
        {
            rm -rf $o$n
            rm -rf /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result
            mkdir $o$n $o$n/images
            rm -rf /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result123.csv
            /Users/ivan/Desktop/ALL/Soft/python3/bin/python runC05.py $o$n $i0 $i1 $i2
            i3=$(cat /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result|grep dets12)
            echo "Run:" $i0, $i1, $i2, $i3
            echo "Run:" $i0, $i1, $i2, $i3 >> /Users/ivan/Desktop/ALL/Code/PyProduct/SecurityAI_Round1/Out/result123.csv
        } done
    } done
} done



