
#
pT1=/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round1/Out/T000001

#
echo 0
rm -rf $pT1/temp*
rm -rf $pT1/images/*

#
echo 1
# 1 712 2 6 0
# st, ed, chose, igroup, group
/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 0 &

/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 1 &

/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 2 &

/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 3 &

/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 4 &

/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9.py 1 12 2 6 5 &
wait

#
echo 2
/Users/ivan/Desktop/ALL/Soft/python3/bin/python run_v0.9_in.py

#
echo 3
cd $pT1
ls $pT1/images|wc -l
rm -rf images.zip
zip -r images.zip images/ -q

