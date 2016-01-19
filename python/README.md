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

Currently, only tested on Mac OS X with Torch7 not installed or not visible. 

Linux' `ctyles.util.find_library` behaviour is significantly different from OSX'. Trying to find a platform independent solution.
