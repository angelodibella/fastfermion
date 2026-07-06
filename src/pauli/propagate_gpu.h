/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// GPU Pauli propagation engine: sorted ping-pong term array with deferred
// deduplication (see GPU_PLAN.md).
//
// This header is the only seam between the header-only C++ library and the
// CUDA translation unit (propagate_gpu.cu). fastfermion is a single
// translation-unit, header-only library, and NVIDIA's compiler (nvcc) must
// never pull it in; keeping only flat machine words and real doubles in these
// declarations lets the .cu file include this one header alone. Every
// fastfermion type therefore stays on the host side — src/pauli/propagate.h
// owns the gate loop, key-width choice, Clifford segments, and the truncation
// schedule. A term's key crosses the boundary as 2*words uint64:
// [xory words..., yorz words...].

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fastfermion {
namespace pauli_gates {
namespace gpu {

// CUDA devices visible at runtime (0 when none, or the driver is absent).
int device_count();

// One propagation run: a device-resident term array kept as a sorted,
// deduplicated base plus an uncompacted tail of freshly emitted partners.
// Buffers are sized once from capacity_hint and grown geometrically (the
// trunc_w arena saturates, so growth is rare). words = 64-bit words per
// Pauli plane: 1 covers <=64 qubits, 2 covers <=128.
class Engine {
  public:
    Engine(int words, std::size_t capacity_hint);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Replace the device contents with n terms (unsorted, duplicates allowed).
    void upload(const std::uint64_t* keys, const double* coeffs, std::size_t n);

    // Conjugate by ROT(theta) with generator key p_key: anticommuting terms
    // are scaled by cos(theta) in place; their partners, weight-filtered at
    // emission (<= maxdegree), are appended to the tail with deduplication
    // deferred to the next compact(). Auto-compacts (threshold-free) when
    // the worst-case append would overflow.
    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree);

    // Sort the tail, merge into the base, sum coefficients of equal keys;
    // then discard |c| <= mincoeff when mincoeff > 0 (the host schedules
    // this to match the CPU truncation cadence).
    void compact(double mincoeff);

    std::size_t size() const;  // resident terms (base + uncompacted tail)

    // Compacts (threshold-free), then copies the term set back to the host.
    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Recompute i^jpow of n Pauli-string products on the device and compare with
// expectations from the host oracle (pauli_string_multiply). The one
// algebraic identity the engine duplicates, so it is tested independently.
bool phase_check(const std::uint64_t* a, const std::uint64_t* b,
                 const int* expected_jpow_mod4, std::size_t n, int words);

}  // namespace gpu
}  // namespace pauli_gates
}  // namespace fastfermion
