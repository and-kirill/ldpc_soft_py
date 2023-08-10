# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/ldpc_soft_py/.
# Copyright (c) 2023 Kirill Andreev.
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

import numpy as np
import matplotlib.pyplot as plt


def logtanh(x):
    return -np.log(np.tanh(x / 2))


def plot_large_x():
    import matplotlib.pyplot as plt
    x_series = np.arange(0, 35, 0.0001)
    exact_vals = logtanh(x_series)
    t = np.exp(-x_series)

    approx_vals = 2 * t
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = 2 * (t + t ** 3 / 3)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = 2 * (t + t ** 3 / 3 + t ** 5 / 5)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)
    plt.legend(['First-order', 'Third-order', 'Fifth-order'])
    plt.grid()
    plt.show()


def plot_small_x():
    logx = np.arange(-13, 0, 0.001)
    x_series = np.exp(logx)
    exact_vals = logtanh(x_series)

    approx_vals = -np.log(x_series / 2)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = -np.log(x_series / 2) + x_series ** 2 / 12
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)
    plt.xscale('log')
    plt.legend(['First-order', 'Second-order'])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_small_x()
    plot_large_x()

