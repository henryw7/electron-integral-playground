
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