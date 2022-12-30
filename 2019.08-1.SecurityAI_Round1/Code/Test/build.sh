echo =====================================
echo 00.START
rm -rf func* setup.py build/

echo =====================================
echo 01.Function
cat > func.pyx << EOF
import numpy as np
import pandas as pd

def whimax(tdata, b1, b2, tmin=-32, tmax=+32, tuse=1):
	base1 = np.random.randint(0, 1, size=tdata.shape)
	base1[:] = b1
	
	base2 = np.random.randint(0, 1, size=tdata.shape)	
	base2[:] = b2
	
	base12 = np.random.randint(0, 1, size=tdata.shape)
	base12[:] = [b1[0]+b2[0], b1[1]+b2[1], b1[2]+b2[2]]

	base3 = np.random.randint(0, 1, size=tdata.shape)
	base3[:] = [-200, -200, -200]

	rn = []
	for n in range(tmin, tmax+1):
		tdata1 = tdata.copy()
		tdata1 = tdata1 + n
		rn.append([n, np.array([tdata1,base1,base2,base12,base3]).var()])
	rn = pd.DataFrame(rn)
	rn.sort_values(by=1, inplace=True)
	return int(rn.head(1).iloc[0][0])

EOF
cat func.pyx
cp func.pyx func1.py

echo =====================================
echo 02.Make Setup
cat > setup.py << EOF
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("func.pyx")
)	
EOF
cat setup.py

echo =====================================
echo 03.Build In C
python setup.py build_ext --inplace

echo =====================================
echo 09.END
