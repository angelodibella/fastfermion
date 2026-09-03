/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// The GPU engine keeps the terms of a polynomial on the device as a sorted, duplicate-free array
// plus a tail of freshly created terms, deduplicated lazily. This header is the only interface
// between the header-only library and the CUDA translation unit engine.cu, which must not include
// any other fastfermion header: terms cross it as flat 64-bit words (the two bit-planes of a Pauli
// string, or the index mask of a Majorana monomial) and real coefficients.

namespace fastfermion {

namespace gpu {

// Number of CUDA devices (0 when the driver is absent)
int device_count();

class Engine {
public:
    // words: 64-bit words per bit-plane (1 for <= 64 qubits, 2 for <= 128), the key of a term being
    // 2*words words. key_format selects the device representation: 0 = the dense bit-planes, 1 or 2
    // = the support list of the string, with 7 (site, letter) slots per word (valid for weights up to
    // 7*key_format, as the caller ensures), -1 = the index mask of a Majorana monomial (its degree
    // being the popcount of the mask).
    // Buffers hold capacity_hint terms initially and grow geometrically, unless reserve_terms > 0
    // requests a one-shot allocation of that many slots: honored if it leaves 30% of the device memory
    // free, or unconditionally (failing loudly if impossible) when reserve_hard is set.
    // beta: the tail is deduplicated as soon as it exceeds beta times the sorted base (0 = after every
    // gate, < 0 = only when the buffers are full). This does not affect the results.
    Engine(int words, std::size_t capacity_hint, std::size_t reserve_terms = 0,
           bool reserve_hard = false, int key_format = 0, double beta = -1);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Replaces the terms with n terms (unsorted, duplicates allowed)
    void upload(const std::uint64_t* keys, const double* coeffs, std::size_t n);

    // Conjugates by the rotation e^{-i theta/2 P} with key p_key: the terms anticommuting with P
    // are scaled by cos(theta) in place and their partners, if of degree <= maxdegree, appended to
    // the tail with a real coefficient +-sin(theta) times their own
    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree);

    // Merges the tail into the base, summing the coefficients of equal keys, then discards the terms
    // of degree > maxdegree (INT_MAX: none) and those of magnitude <= mincoeff (0: none). Returns the
    // sum of the squared magnitudes discarded by the threshold, among the terms the degree rule keeps.
    double compact(double mincoeff, int maxdegree);

    // Number of terms held (base + tail)
    std::size_t size() const;

    // High-water mark of the engine's device allocations
    std::size_t peak_device_bytes() const;

    // Compacts (without threshold), then copies the terms to the host
    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Recomputes i^jpow of n products of Pauli strings on the device and compares with the expected
// values (from pauli_string_multiply)
bool phase_check(const std::uint64_t* a, const std::uint64_t* b, const int* expected_jpow_mod4,
                 std::size_t n, int words);

}

}
