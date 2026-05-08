/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// Common types, constants, and scalar utilities for Pauli/Majorana/Fermi
// operator algebras.

#pragma once

#include <cmath>
#include <complex>

#include "bits.h"
#include "bits_utils.h"
#include "error.h"

namespace fastfermion {

// Max supported qubits = 64 * SYS_NUM_ULONG
#define SYS_NUM_ULONG 2

#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))

// --- Type aliases ---

using ff_float = double;
using ff_complex = std::complex<double>;
using ff_ulong = BitSet<SYS_NUM_ULONG>;

// --- Display configuration ---

struct {
    char fermi_symbol = 'f';
    char majorana_symbol = 'm';
    char dagger_symbol = '^';
    char identity_symbol = 'I';
    char imaginary_symbol = 'j';
    int max_line_length = 120;
    int max_terms_to_show = 200;
} ff_config;

// --- Scalar arithmetic ---

// int * ff_complex promotion (avoids implicit int->double conversion)
ff_complex operator*(const int& a, const ff_complex& b) {
    return ff_complex(a * b.real(), a * b.imag());
}
ff_complex operator*(const ff_complex& b, const int& a) {
    return ff_complex(a * b.real(), a * b.imag());
}

namespace scalar_utils {

// (-1)^m
inline int m1pow(int m) { return ((m % 2 == 0) ? 1 : (-1)); }

// i^m, where i is the imaginary unit
inline ff_complex Ipow(int m) {
    if (m % 2 == 0)
        return ff_complex(m1pow(m / 2), 0);
    else
        return ff_complex(0, m1pow((m - 1) / 2));
}

}  // namespace scalar_utils

}  // namespace fastfermion
