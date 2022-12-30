
n=0000000t
o=/data/gproj/code/SecurityAI_Round1/Out/

l=90
h=100

for ((i0=$l; i0<=$h; i0++)); do
{
    for ((i1=$l; i1<=$h; i1++)); do
    {
        for ((i2=$l; i2<=$h; i2++)); do
        {
            rm -rf $o$n $o/result
            mkdir $o$n $o$n/images
            rm -rf $o/result123.csv
            /data/soft/py3/bin/python runC05.py $o$n $i0 $i1 $i2
            i3=$(cat $o/result|grep dets12)
            echo "Run:" $i0, $i1, $i2, $i3
            echo "Run:" $i0, $i1, $i2, $i3 >> $o/result123.csv
        } done
    } done
} done



