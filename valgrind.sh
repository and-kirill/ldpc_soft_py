#!/bin/bash
g++ -std=c++14 -g ldpc.cpp ldpc_ctypes.cpp valgrind.cpp && valgrind ./a.out
