
Overview
--------
This package is not intended for performance or industry-level stability. If you're looking for industry standard for quantum chemistry calculation, consider Gaussian. If you want top performance, consider Q-Chem for best CPU performance, and GPU4PySCF or TeraChem for best GPU performance. If you want free, open-source reference, consider PySCF and GPU4PySCF.

Instead, this package is intended for readability. We document all the equations we're using into the code. We reveal all the code generator scripts. We want every new quantum chemistry or computer science students to understand our code, and be able to easily grab the pieces they need.

Of course, we're making all efforts to make sure the results are correct. And although it's not our top priority, we adopt principals of high-performance computing, and design our code in a way that can be easily turned into a CPU multi thread/GPU accelerated program.

Compilation
--------
```sh
mkdir build;
cd build;
cmake ..;
make -j 32;
```

Set environment variables
--------
```sh
export PYTHONPATH=$PWD:$PYTHONPATH;
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH;
```
