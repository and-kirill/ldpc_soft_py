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

#include <limits>
#include <cstdint>
#include <iostream>
#include "./bin_ldpc_soft_impl.h"
#include "./bin_ldpc_soft_settings.h"

// Specify LLR type constants
#define LLR_TYPE_FLOAT64 0
#define LLR_TYPE_FLOAT32 1

// Specify LDPC decoding commands:
#define DECODE_LDPC_SPA      0 // Sum-product
#define DECODE_LDPC_NOMS     2 // Normalized offset min-sum (NOMS)
#define DECODE_LDPC_NOMS_HL  3 // NOMS, horizontally layered
#define DECODE_LDPC_NOMS_VL  4 // NOMS, vertically layered

// Specify GLDPC decoding commands:
#define DECODE_GLDPC_NOMS_HL 5 // NOMS, horizontally layered
#define DECODE_GLDPC_SPA     6 // Sum-product


void* init_ldpc_settings(unsigned int blocklen,
                         unsigned int n_checks,
                         unsigned int llr_type,
                         bool
                         syndrome_termintion,
                         unsigned int  n_iterations,
                         unsigned int *row_sequence,
                         double       *scale_array,
                         double       *offset_array) {
  switch (llr_type) {
    case LLR_TYPE_FLOAT64:
      return params_factory<double>(blocklen,
                                    n_checks,
                                    syndrome_termintion,
                                    n_iterations,
                                    row_sequence,
                                    scale_array,
                                    offset_array);
    case LLR_TYPE_FLOAT32:
      return params_factory<float>(blocklen,
                                   n_checks,
                                   syndrome_termintion,
                                   n_iterations,
                                   row_sequence,
                                   scale_array,
                                   offset_array);
    default:
      return reinterpret_cast<void *>(0);
  }
}

/// Structure to be returned to python via ctypes
struct Decoder {
  Decoder(char         *alist_path,
          unsigned int  llr_type,
          bool          syndrome_termintion,
          unsigned int  n_iterations,
          unsigned int *row_sequence,
          double       *scales_array,
          double       *offset_array) :
    llr_type(llr_type) {
    tng = load_alist(alist_path, blocklength, n_checks);

    ldpc_settings = init_ldpc_settings(blocklength,
                                       n_checks,
                                       llr_type,
                                       syndrome_termintion,
                                       n_iterations,
                                       row_sequence,
                                       scales_array,
                                       offset_array);
  }

  ~Decoder() {
    if (blocklength < std::numeric_limits<uint16_t>::max()) {
      delete static_cast<TannerGraph<uint16_t> *>(tng);

      switch (llr_type) {
        case LLR_TYPE_FLOAT64:
          delete static_cast<DecParams<double, uint16_t> *>(ldpc_settings);
          break;
        case LLR_TYPE_FLOAT32:
          delete static_cast<DecParams<float, uint16_t> *>(ldpc_settings);
          break;
      }
    } else {
      delete static_cast<TannerGraph<uint32_t> *>(tng);

      switch (llr_type) {
        case LLR_TYPE_FLOAT64:
          delete static_cast<DecParams<double, uint32_t> *>(ldpc_settings);
          break;
        case LLR_TYPE_FLOAT32:
          delete static_cast<DecParams<float, uint32_t> *>(ldpc_settings);
          break;
      }
    }
  }

  /// LLR types (specified above)
  unsigned int llr_type;

  /// Block length. uint16_t indexing is used when block length allows this
  unsigned int blocklength;

  /// The number of parity checks
  unsigned int n_checks;

  /// Pointer to LDPC settings
  void *ldpc_settings;

  /// Pointer to Tanner graph representation
  void *tng;
};

void warn_gldpc(bool is_azcw) {
  // Nonzero codeword is not supported by GLDPC
  if (!is_azcw) {
    std::cerr << "WARNING: Non-zero codeword is not supported by GLDPC";
    std::cerr << std::endl;
  }
}

template<typename TL, typename TI>
unsigned int do_decode_soft(unsigned int command,
                            void        *ldpc_ptr,
                            TL          *llr_in,
                            bool         is_azcw,
                            TL          *llr_out) {
  Decoder *ldpc               = static_cast<Decoder *>(ldpc_ptr);
  TannerGraph<TI>   *tng      = static_cast<TannerGraph<TI> *>(ldpc->tng);
  DecParams<TL, TI> *settings =
    static_cast<DecParams<TL, TI> *>(ldpc->ldpc_settings);

  switch (command) {
    case DECODE_LDPC_SPA:
      return ldpc_sp<TL, TI>(*tng, *settings, llr_in, is_azcw, llr_out);
    case DECODE_LDPC_NOMS:
      return ldpc_noms<TL, TI>(*tng, *settings, llr_in, is_azcw, llr_out);
    case DECODE_LDPC_NOMS_HL:
      return ldpc_hl_noms<TL, TI>(*tng, *settings, llr_in, is_azcw, llr_out);
    case DECODE_LDPC_NOMS_VL:
      return ldpc_vl_noms<TL, TI>(*tng, *settings, llr_in, is_azcw, llr_out);
    case DECODE_GLDPC_NOMS_HL:
      warn_gldpc(is_azcw);
      return gldpc_hl_noms<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_GLDPC_SPA:
      warn_gldpc(is_azcw);
      return gldpc_sp<TL, TI>(*tng, *settings, llr_in, llr_out);
  }
  std::cerr << "WARNING: Unknown decoding command: " << command << std::endl;
  return 0;
}

template<typename TL>
unsigned int decode_soft(unsigned int command,
                         void        *ldpc_ptr,
                         TL          *llr_in,
                         bool         is_azcw,
                         TL          *llr_out) {
  Decoder *ldpc = static_cast<Decoder *>(ldpc_ptr);

  if (ldpc->blocklength > std::numeric_limits<uint16_t>::max()) {
    return do_decode_soft<TL, uint32_t>(command,
                                        ldpc_ptr,
                                        llr_in,
                                        is_azcw,
                                        llr_out);
  } else {
    return do_decode_soft<TL, uint16_t>(command,
                                        ldpc_ptr,
                                        llr_in,
                                        is_azcw,
                                        llr_out);
  }
}

extern "C"
void* init_ldpc(char         *alist_path,
                unsigned int  llr_type,
                bool          syndrome_termintion,
                unsigned int  n_iterations,
                unsigned int *row_sequence,
                double       *scales_array,
                double       *offset_array) {
  return new Decoder(alist_path,
                     llr_type,
                     syndrome_termintion,
                     n_iterations,
                     row_sequence,
                     scales_array,
                     offset_array);
}

extern "C"
void free_ldpc(void *ldpc_ptr) {
  delete (static_cast<Decoder *>(ldpc_ptr));
}

extern "C"
unsigned int decode_siso_float32(unsigned int command,
                                 void        *ldpc_ptr,
                                 float       *llr_in,
                                 bool         is_azcw,
                                 float       *llr_out) {
  return decode_soft<float>(command, ldpc_ptr, llr_in, is_azcw, llr_out);
}

extern "C"
unsigned int decode_siso_float64(unsigned int command,
                                 void        *ldpc_ptr,
                                 double      *llr_in,
                                 bool         is_azcw,
                                 double      *llr_out) {
  return decode_soft<double>(command, ldpc_ptr, llr_in, is_azcw, llr_out);
}
