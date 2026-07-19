/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// GPU Pauli propagation engine: sorted ping-pong term array with deferred
// deduplication.
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
// words = 64-bit words per Pauli plane: 1 covers <=64 qubits, 2 covers <=128.
//
// Sizing: reserve_terms > 0 requests a one-shot allocation of that many term
// slots (the host passes 2x a term-count bound: the weight arena |P_{n,w}| on
// auto, or the user's expert estimate -- above 2x its base the engine
// auto-compacts, so growth never fires and the grow-time old+new copy spike
// cannot occur). With reserve_hard = false the request is best-effort: taken
// only when it fits comfortably in free device memory, else the engine falls
// back to capacity_hint and geometric growth (exact-size steps near the
// memory ceiling). With reserve_hard = true (user-specified sizing) the
// allocation is attempted as given and fails loudly if it does not fit.
class Engine {
  public:
    // sparse_words selects the key representation: 0 = dense bit-planes
    // (2*words machine words per term); 1 or 2 = the support-list key with
    // 7 (site, letter) slots per word -- one word covers weight <= 7 at
    // n <= 127, a quarter of the dense record at n = 100, and the sort and
    // merge are bandwidth-bound so the record size is the cost. Eligibility
    // (emission-enforced cutoff, initial weights within capacity) is the
    // caller's responsibility.
    // beta: compact when the unsorted tail exceeds beta * base (a plain dedup,
    // schedule-free by compaction invariance); 0 = per-gate, < 0 = only at
    // capacity pressure. The batch-length model predicts the optimum.
    Engine(int words, std::size_t capacity_hint, std::size_t reserve_terms = 0,
           bool reserve_hard = false, int sparse_words = 0, double beta = -1);
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
    // then discard weight > maxdegree and |c| <= mincoeff where those rules
    // are active (mincoeff > 0 / maxdegree below the 128-qubit no-op value;
    // the host schedules both to match the CPU truncation cadence). Returns
    // the discarded |c|^2 sum of the THRESHOLD rule alone — the squared HS
    // norm delta_e^2 of the event, feeding the run certificate Sum_e delta_e
    // — counting only terms the weight rule keeps, and 0 when it is inactive.
    double compact(double mincoeff, int maxdegree = 256);

    std::size_t size() const;  // resident terms (base + uncompacted tail)

    // High-water mark of the engine's own device allocations (term buffers +
    // CUB scratch + scalars), counted deterministically at allocation sites --
    // the run's peak memory metric, independent of what else shares the GPU.
    std::size_t peak_device_bytes() const;

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
