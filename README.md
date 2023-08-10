## Overview
Binary soft decoder of LDPC codes with python cTypes wrapper. This is a refactored version of LDPC from [mMTC fading simulation](https://github.com/and-kirill/fading-joint) project. Supports the following decoders:
* sum-product
* min-sum with scale array and offset array
* Layered version of both decoders

For the sum-product case, a message passing algorithm is implemented with logarithm of input log-likelihood ratios using. This prevents us from hyperbolic tangents multiplication.
## Documentation
Compile [doc.tex](doc.tex) for detailed explanation of LDPC Tanner graph implementation.
## Usage
* Run [ldpc.py](ldpc.py) for a single decoding attempt
* Run [valgrind.sh](valgrind.sh) for memory check and profiling
* Parity check matrix is represented in [alist](http://www.inference.org.uk/mackay/codes/alist.html) format.
* To run simulations, please refer to [simulation](https://github.com/and-kirill/simulator_awgn_python) module
## Comparison with a reference implementation

## Implementation notes
There are two versions of ```logtanh``` function specified by ```FLOAT_LOGTANH``` macro.
Default version uses ```log(tanh(x))``` with clipping. Alternative version uses float instead of double and approximations at small and large input arguments. See [plot_logtanh.py](plot_logtanh.py) for a detailed explanation of approximation thresholds selected in [logtanh](ldpc.cpp) function.

