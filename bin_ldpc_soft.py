"""
This module implements the soft decoder of binary LDPC codes.
C++ implementation is stored in impl directory.
cTypes are used to execute decoder from python
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/ldpc_soft_py/.
# Copyright (c) 2023 Kirill Andreev, Alexey Frolov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
 

import ctypes
import os
import numpy as np


class Alist:
    """
    Alist parser for binary matrices.
    See http://www.inference.org.uk/mackay/codes/alist.html for more details
    """
    @staticmethod
    def write(matr, filename):
        """
        Write matrix to file
        :param matr: 2D np.array to be processed
        :param filename: Filename to save (utf-8 encoded)
        """
        column_weights = np.sum(matr > 0, axis=0)
        row_weights = np.sum(matr > 0, axis=1)
        row_max = np.max(row_weights)
        col_max = np.max(column_weights)

        with open(filename, 'w', encoding='utf-8') as file:
            # Print shape
            print(f'{matr.shape[1]} {matr.shape[0]}', file=file)
            # Print maximum row weight and column weight
            print(f'{col_max} {row_max}', file=file)
            # Print column weights
            print(Alist.to_string(column_weights), file=file)
            # Print row weights
            print(Alist.to_string(row_weights), file=file)

            for i in range(matr.shape[1]):
                print(Alist.to_string(np.nonzero(matr[:, i])[0] + 1, col_max), file=file)
            for i in range(matr.shape[0]):
                print(Alist.to_string(np.nonzero(matr[i, :])[0] + 1, row_max), file=file)

    @staticmethod
    def read(filename):
        """
        Read aliist file and fill 2D numpy array
        :param filename: file with alist data (utf-8 encoded)
        :return: 2D binary np.array (np.uint8 type)
        """
        with open(filename, 'r', encoding='utf-8') as file:
            matrix_size = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            matr = np.zeros((matrix_size[1], matrix_size[0]), dtype=np.uint)
            max_counts = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            row_weights = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            col_weights = np.fromstring(file.readline(), sep=' ', dtype=np.uint)

            for i in range(matr.shape[1]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[0]
                # Remove zeros
                idx = idx[idx > 0] - 1
                assert len(idx) == row_weights[i]
                # Assign matrix elements
                matr[idx, i] = 1

            # The remaining of the file is redundant. Just sanity checks below
            for i in range(matr.shape[0]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[1]
                idx = idx[idx > 0] - 1
                assert len(idx) == col_weights[i]
                assert np.sum(matr[i, :]) == len(idx)
                assert np.sum(matr[i, idx]) == len(idx)

        return matr

    @staticmethod
    def read_shape(filename):
        """
        Read the parity check matrix shape only
        """
        with open(filename, 'r', encoding='utf-8') as file:
            matrix_size = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
        return matrix_size[1], matrix_size[0]

    @staticmethod
    def to_string(np_arr, length=None):
        """
        Convert numpy array to space-separated string
        :param np_arr:
        :param length: total length of the array (zero-padding if required)
        :return: string (space separated, as required by alist formart)
        """
        if length:
            np_arr = np.hstack([np_arr, np.array([0] * (length - len(np_arr)))])
        return ' '.join(map(str, np_arr.astype(np.uint).tolist()))


# C++ implementation: compilation, linking, and execution routines
if os.name == 'nt':
    LIB_PATH = 'bin_ldpc_soft_decoder.dll'
else:
    LIB_PATH = 'bin_ldpc_soft_decoder.so'


def lib_compile():
    """
    Compile the C++ SCL decoder implementation
    NOTE: In normalized offset min-sum, the offset option is implemented under OFFSETS_ENABLED ifdef
    NOTE: To enable a full-spport of offsets, add -DOFFSETS_ENABLED flag.
    NOTE: Mind performance issues if enabling offsets
    """
    wdir = os.path.dirname(__file__)
    # if os.path.isfile(LIB_PATH):
    #     return
    src = ['bin_alist_loader', 'bin_ldpc_soft_ctypes']
    src_abs = [os.path.join(wdir, s) for s in src]
    if os.name == 'nt':
        if not os.popen('where g++').read():
            raise RuntimeError('g++ not found.')
    else:
        if not os.popen('which g++').read():
            raise RuntimeError('g++ not found.')

    for src_file in src_abs:
        os.system(f'g++ -Wall -O3 -fPIC -c -o {src_file}.o {src_file}.cpp')
    os.system(
        'g++ -shared -o ' +
        os.path.join(wdir, LIB_PATH) + ' ' +
        ''.join([s + '.o ' for s in src_abs])
    )
    obj_files = os.path.join(wdir, '*.o')
    if os.name == 'nt':
        os.system(f'del {obj_files}')
    else:
        os.system(f'rm {obj_files}')


def load_lib():
    """
    Load shared library
    """
    wdir = os.path.abspath(os.path.dirname(__file__))
    lib = ctypes.CDLL(os.path.join(wdir, LIB_PATH))
    # Tanner graph initializer
    lib.init_ldpc.restype = ctypes.c_void_p
    lib.init_ldpc.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint,  # LLR types: 0: double, 1: float, 2: int
        ctypes.c_bool,  # If true, enable early termination by syndrome
        ctypes.c_uint,  # The number of decoding iterations
        np.ctypeslib.ndpointer(dtype=np.uint32),  # Row sequence
        np.ctypeslib.ndpointer(dtype=np.float64),  # Scale array
        np.ctypeslib.ndpointer(dtype=np.float64),  # Offset array
    ]

    # Decode functions
    lib.decode_siso_float32.restype = ctypes.c_uint
    lib.decode_siso_float32.argtypes = [
        ctypes.c_uint,                             # Command
        ctypes.c_void_p,                           # LDPC pointer
        np.ctypeslib.ndpointer(dtype=np.float32),  # Channel LLR
        ctypes.c_bool,                             # AZCW flag
        np.ctypeslib.ndpointer(dtype=np.float32),  # Output LLRs
    ]
    lib.decode_siso_float64.restype = ctypes.c_uint
    lib.decode_siso_float64.argtypes = [
        ctypes.c_uint,  # Command
        ctypes.c_void_p,  # LDPC pointer
        np.ctypeslib.ndpointer(dtype=np.float64),  # Channel LLR
        ctypes.c_bool,                             # AZCW flag
        np.ctypeslib.ndpointer(dtype=np.float64),  # Output LLRs
    ]
    # Destroy LDPC object
    lib.free_ldpc.restype = None
    lib.free_ldpc.argtypes = [
        ctypes.c_void_p
    ]
    return lib


class BinLdpcSoftDecoderBase:
    """
    This class creates basic LDPC codec instance given alist file
    and provides all LDPC decoding routines
    """

    def __init__(self, alist_filename, llr_type_str, n_iterations, llr_scale):
        self.ldpc_ptr = None
        self.shared_object = load_lib()
        self.n_checks, self.block_len = Alist.read_shape(alist_filename)
        if llr_type_str == 'float64':
            self.lib_decode_soft = self.lib_decode_soft_float64
            llr_type = 0  # LLR type. Check C++ implementation to match
        elif llr_type_str == 'float32':
            self.lib_decode_soft = self.lib_decode_soft_float32
            llr_type = 1  # LLR type. Check C++ implementation to match
        else:
            raise ValueError(f'Unknown / unspecified LLR type. Got {llr_type_str}.')

        self.ldpc_ptr = self.shared_object.init_ldpc(
            alist_filename.encode(),
            llr_type,
            True,  # Early stop by the syndrome
            n_iterations,
            np.arange(self.n_checks, dtype=np.uint32),  # Row sequence
            np.ones(self.block_len,) * llr_scale,  # LLR scales (per-column)
            np.zeros(self.block_len,)  # Offsets (per-column)
        )
        if not self.ldpc_ptr:
            raise ValueError('Failed to initialize decoder.')

    def lib_decode_soft_float64(self, llr_in, is_azcw, decoder_type):
        """
        Float-64 decoder. All input and output arrays have float64 types
        """
        llr_out = np.zeros_like(llr_in, dtype=np.float64)
        n_iter = self.shared_object.decode_siso_float64(
            decoder_type,
            self.ldpc_ptr,
            llr_in,
            is_azcw,
            llr_out
        )
        return llr_out, n_iter

    def lib_decode_soft_float32(self, llr_in, is_azcw, decoder_type):
        """
        Float-32 decoder. Nevertheless, all input and output arrays have float64 types
        """
        llr_out = np.zeros_like(llr_in, dtype=np.float32)
        n_iter = self.shared_object.decode_siso_float32(
            decoder_type,
            self.ldpc_ptr,
            llr_in.astype(np.float32),
            is_azcw,
            llr_out
        )
        return llr_out.astype(np.float64), n_iter

    def __del__(self):
        if self.ldpc_ptr is not None:
            self.shared_object.free_ldpc(self.ldpc_ptr)

    def _decode_soft(self, llr_in, is_azcw, decoder_type):
        """
        Run C++ implementation
        """
        assert self.block_len == len(llr_in)
        return self.lib_decode_soft(llr_in, is_azcw, decoder_type)


class BinLdpcSoftDecoder(BinLdpcSoftDecoderBase):
    """
    Binary soft LDPC decoder instance
    """
    def __init__(self, alist_filename, llr_type_str, n_iterations, llr_scale):
        super().__init__(alist_filename, llr_type_str, n_iterations, llr_scale)

    def sum_product(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=0)

    def min_sum(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=2)

    def h_layered_min_sum(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=3)

    def v_layered_min_sum(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=4)


class BinGldpcSoftDecoder(BinLdpcSoftDecoderBase):
    """
    Binary soft Generalized LDPC decoder instance.
    Provide a parity check matrix shat should be extended by Cordaro-Wagner code.
    The actual number of parity checks in the equivalent binary PCM will be twice larger
    than in the provided PCM.
    """
    def __init__(self, alist_filename, llr_type_str, n_iterations, llr_scale):
        super().__init__(alist_filename, llr_type_str, n_iterations, llr_scale)

    def h_layered_min_sum(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=5)

    def sum_product(self, llr_in, is_azcw):
        return self._decode_soft(llr_in, is_azcw, decoder_type=6)


if __name__ == '__main__':
    lib_compile()
