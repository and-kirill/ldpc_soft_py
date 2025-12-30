/*
 * This file is part of the simulator_awgn_python distribution
 * https://github.com/and-kirill/ldpc_soft_py/.
 * Copyright (c) 2023 Kirill Andreev, Alexey Frolov
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LDPC_SETTINGS_H_
#define LDPC_SETTINGS_H_
#include <vector>
#include <cstdint>

template<typename T_OUT, typename T_IN>
std::vector<T_OUT>convert_vector(T_IN *vals, unsigned int length) {
  std::vector<T_OUT> converted = std::vector<T_OUT>(length);

  for (unsigned int i = 0; i < length; i++) {
    converted[i] = (T_OUT)vals[i];
  }
  return converted;
}

template<typename TL, typename TI>
struct DecParams {
  // Minimal config: includes the number of iterations, early termination
  // criterion. Used by sum-product settings
  DecParams(unsigned int n_iterations, bool terminate_syndrome) :
    n_iterations(n_iterations),
    terminate_syndrome(terminate_syndrome)
  {};

  // Min-sum config that includes offsets and scales
  DecParams(unsigned int n_iterations, bool terminate_syndrome,
            std::vector<TL>scales, std::vector<TL>offsets) :
    n_iterations(n_iterations),
    terminate_syndrome(terminate_syndrome),
    scales(scales),
    offsets(offsets),
    row_sequence(std::vector<TI>(offsets.size()))
  {
    for (TI i = 0; i < offsets.size(); i++) {
      row_sequence[i] = i;
    }
  };

  // Layered min-sum config with custom row sequence
  DecParams(unsigned int n_iterations, bool terminate_syndrome,
            std::vector<TL>scales, std::vector<TL>offsets,
            std::vector<TI>row_sequence) :
    n_iterations(n_iterations),
    terminate_syndrome(terminate_syndrome),
    scales(scales),
    offsets(offsets),
    row_sequence(row_sequence)
  {};

  unsigned int   n_iterations;
  bool           terminate_syndrome;
  std::vector<TL>scales;
  std::vector<TL>offsets;
  std::vector<TI>row_sequence;
};

template<typename TL>
void* params_factory(unsigned int  blocklen,
                     unsigned int  n_checks,
                     bool          syndrome_termintion,
                     unsigned int  n_iterations,
                     unsigned int *row_sequence,
                     double       *scales_array,
                     double       *offset_array) {
  if (blocklen > std::numeric_limits<uint16_t>::max()) {
    convert_vector<TL, double>(            scales_array, blocklen);
    convert_vector<TL, double>(            offset_array, blocklen);
    convert_vector<uint32_t, unsigned int>(row_sequence, n_checks);
    return new DecParams<TL, uint32_t>(
      n_iterations,
      syndrome_termintion,
      convert_vector<TL, double>(            scales_array, blocklen),
      convert_vector<TL, double>(            offset_array, blocklen),
      convert_vector<uint32_t, unsigned int>(row_sequence, n_checks)
      );
  } else {
    return new DecParams<TL, uint16_t>(
      n_iterations,
      syndrome_termintion,
      convert_vector<TL, double>(            scales_array, blocklen),
      convert_vector<TL, double>(            offset_array, blocklen),
      convert_vector<uint16_t, unsigned int>(row_sequence, n_checks)
      );
  }
}

#endif // ifndef LDPC_SETTINGS_H_
