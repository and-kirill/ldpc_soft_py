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

#include <string>
#include <random>

#include <iostream>
#include <iomanip>
#include <chrono>


#define ALIST_PATH "ldpc_k120_n600_pcm.alist"


extern "C"
void* init_ldpc(char *buf);

extern "C"
void  free_ldpc(void *ptr);


extern "C"
void decode_soft(
  unsigned int  command,
  void         *tng_ptr,
  double       *llr_in,
  unsigned int  n_iterations,
  double       *llr_out_buf,
  unsigned int *row_seq,
  double       *scale_array,
  double       *offset_array);


int main(void) {
  uint64_t ms_time       = 0;
  uint64_t lms_time      = 0;
  uint64_t sp_time       = 0;
  std::string alist_file = ALIST_PATH;
  int N                  = 640;
  int n_iter             = 50;
  int n_tests            = 10; // Increase if need profiling


  std::random_device rd{};
  std::mt19937 gen{ rd() };
  std::normal_distribution<> d { 0, 1 };

  void *ldpc = init_ldpc(ALIST_PATH);

  if (ldpc == 0) {
    std::cout << "Failed to initialize the Tanner graph" << std::endl;
    return 1;
  }

  double *llr_in        = (double *)malloc(N * sizeof(double));
  double *out_llr       = (double *)malloc(N * sizeof(double));
  unsigned int *row_seq = (unsigned int *)malloc(N * sizeof(unsigned int));
  double *scales        = (double *)malloc(N * sizeof(double));
  double *offsets       = (double *)malloc(N * sizeof(double));

  for (unsigned int i = 0; i < N; i++) {
    scales[i]  = 0.75;
    offsets[i] = 0;
    row_seq[i] = i;
  }

  for (int i = 0; i < n_tests; i++) {
    for (unsigned int i = 0; i < N; i++) {
      llr_in[i] = d(gen);
    }
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    start = std::chrono::steady_clock::now();
    decode_soft(0,            // Command, 0
                (void *)ldpc, // Pointer to the Tanner graph structure
                llr_in,       // Input log-likelihood ratios
                n_iter,       // The number of decoding iterations
                out_llr,      // Output LLRs
                row_seq,      // Parity checks processing schedule
                scales,       // Scaling coefficients (has effect for min-sum
                              // only)
                offsets);     // Offset coefficients (has effect for min-sum
                              // only)
    end      = std::chrono::steady_clock::now();
    sp_time +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();


    start = std::chrono::steady_clock::now();
    decode_soft(2,            // Command, 0
                (void *)ldpc, // Pointer to the Tanner graph structure
                llr_in,       // Input log-likelihood ratios
                n_iter,       // The number of decoding iterations
                out_llr,      // Output LLRs
                row_seq,      // Parity checks processing schedule
                scales,       // Scaling coefficients (has effect for min-sum
                              // only)
                offsets);     // Offset coefficients (has effect for min-sum
                              // only)
    end      = std::chrono::steady_clock::now();
    ms_time +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    decode_soft(3,            // Command, 0
                (void *)ldpc, // Pointer to the Tanner graph structure
                llr_in,       // Input log-likelihood ratios
                n_iter,       // The number of decoding iterations
                out_llr,      // Output LLRs
                row_seq,      // Parity checks processing schedule
                scales,       // Scaling coefficients (has effect for min-sum
                              // only)
                offsets);     // Offset coefficients (has effect for min-sum
                              // only)
    end       = std::chrono::steady_clock::now();
    lms_time +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  // Free allocated resources
  free(llr_in);
  free(out_llr);
  free(row_seq);
  free(scales);
  free(offsets);
  free_ldpc(ldpc);

  std::cout << "Time spent [nanosec]" << std::endl;
  std::cout << "Sum-product:     " << std::setw(9)  << sp_time / 1e9 <<
    std::endl;
  std::cout << "Min-sum:         " << std::setw(9)  << ms_time / 1e9 <<
    std::endl;
  std::cout << "Layered min-sum: " << std::setw(9) << lms_time / 1e9 <<
    std::endl;
  return 0;
}
