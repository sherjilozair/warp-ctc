# Instructions

Install warp-ctc first using cmake

- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `make install`

You can also specify a prefix to tell cmake where to install, using 
- `cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..`

Now, in root directory, run `sudo python setup.py install`

To run a simple example, `cd python; python warpctc.py`
