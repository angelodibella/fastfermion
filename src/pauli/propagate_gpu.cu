/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// CUDA engine behind src/pauli/propagate_gpu.h: a sorted ping-pong term array
// with deferred deduplication (design + measured rationale in GPU_PLAN.md).
//
// Per gate (emission): flag anticommuting terms whose partner survives the
// weight cutoff -> exclusive offsets by scan -> one pass that scales kept
// coefficients by cos(theta) in place and appends partners to the tail.
// Per compaction: sort the tail, DeviceMerge into the sorted base,
// DeviceReduceByKey to sum equal keys, optional threshold select.
//
// Deliberately self-contained: the rest of fastfermion is a single-TU
// header-only library, so no fastfermion header is included here — the two
// algebraic identities this file duplicates (symplectic commutation parity
// and the i^jpow product phase of pauli_string_multiply) are verified against
// the host oracle by phase_check() and the backend-agreement tests.
//
// Scope (v3.0): <= 128 qubits (words <= 2), < 2^31 resident terms.
// Determinism: scan-based placement and fixed-order reductions make results
// bit-reproducible run-to-run on one device.

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "pauli/propagate_gpu.h"

namespace fastfermion {
namespace pauli_gates {
namespace gpu {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("fastfermion gpu: ") + what + ": " +
                                 cudaGetErrorString(err));
}

constexpr int BLOCK = 256;
inline unsigned n_blocks(std::size_t n) { return static_cast<unsigned>((n + BLOCK - 1) / BLOCK); }

// A Pauli key: W words of xory bits then W words of yorz bits.
template <int W>
struct Key {
    unsigned long long v[2 * W];
    __host__ __device__ friend bool operator==(const Key& a, const Key& b) {
        for (int i = 0; i < 2 * W; i++)
            if (a.v[i] != b.v[i]) return false;
        return true;
    }
};

// Any fixed total order works (internal only); most-significant word first.
template <int W>
struct KeyLess {
    __device__ bool operator()(const Key<W>& a, const Key<W>& b) const {
        for (int i = 2 * W - 1; i >= 0; i--)
            if (a.v[i] != b.v[i]) return a.v[i] < b.v[i];
        return false;
    }
};

template <int W>
__device__ inline bool anticommutes(const Key<W>& p, const Key<W>& q) {
    // Symplectic inner-product parity (PauliString::commutes).
    int m = 0;
    for (int i = 0; i < W; i++)
        m += __popcll(p.v[i] & q.v[W + i]) + __popcll(p.v[W + i] & q.v[i]);
    return m & 1;
}

// Pauli weight of the product p*q: count the sites where the XORed planes
// still act (the partner's key is exactly this XOR).
template <int W>
__device__ inline int partner_weight(const Key<W>& p, const Key<W>& q) {
    int w = 0;
    for (int i = 0; i < W; i++) w += __popcll((p.v[i] ^ q.v[i]) | (p.v[W + i] ^ q.v[W + i]));
    return w;
}

// i^jpow of the product p*q (the pauli_string_multiply identity), mod 4.
template <int W>
__device__ inline int jpow_mod4(const Key<W>& p, const Key<W>& q) {
    int jp = 0;
    for (int i = 0; i < W; i++) {
        jp += __popcll(p.v[i] & p.v[W + i]) + __popcll(q.v[i] & q.v[W + i]) +
              2 * __popcll(p.v[W + i] & q.v[i]) -
              __popcll((p.v[i] ^ q.v[i]) & (p.v[W + i] ^ q.v[W + i]));
    }
    return ((jp % 4) + 4) % 4;
}

// --- kernels -------------------------------------------------------------

// Emission per gate runs as three steps with no atomics and no locks: k_flag
// marks each term 0/1; an inclusive prefix sum (running totals) turns the
// marks into output slots -- the k-th marked term reserves slot k -- and
// k_emit writes only into its own reserved slot, so no two threads ever share
// a write target. Slots are reserved up front (instead of the simpler shared
// atomic counter) for reproducibility: it fixes the append order, and
// floating-point sums depend on their order, so every run is bit-identical.

// Mark term i when it anticommutes with the gate p (so ROT rotates it) and the
// partner it would spawn survives the degree cutoff (weight <= maxw).
template <int W>
__global__ void k_flag(const Key<W>* keys, std::size_t n, Key<W> p, int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const Key<W> q = keys[i];
    flag[i] = (anticommutes(p, q) && partner_weight(p, q) <= maxw) ? 1u : 0u;
}

// pos now holds the prefix sums from k_flag. Scale every anticommuting
// coefficient by cos(theta) in place (done for all anticommuting terms,
// independent of the weight filter, matching the CPU conjugation); a term that
// also passed the filter writes its partner into its reserved tail slot.
template <int W>
__global__ void k_emit(Key<W>* keys, double* coeffs, std::size_t n, Key<W> p, double cos_t,
                       double sin_t, const unsigned* pos) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const Key<W> q = keys[i];
    if (!anticommutes(p, q)) return;
    const double a = coeffs[i];
    // pos[i-1] counts the flagged terms before i, which is this term's
    // reserved slot; the flag itself is recovered as pos[i] > pos[i-1].
    const unsigned prev = i ? pos[i - 1] : 0u;
    if (pos[i] > prev) {
        Key<W> r;
        for (int k = 0; k < 2 * W; k++) r.v[k] = p.v[k] ^ q.v[k];
        // Partner coefficient a * (i sin) * i^jpow = a * sin * i^(jpow+1);
        // jpow is odd for anticommuting pairs, so the factor is +-sin.
        const int jp = jpow_mod4(p, q);
        keys[n + prev] = r;
        coeffs[n + prev] = (jp & 2) ? a * sin_t : -a * sin_t;
    }
    coeffs[i] = a * cos_t;
}

// Pauli weight of a key: sites where either plane acts.
template <int W>
__device__ inline int key_weight(const Key<W>& q) {
    int w = 0;
    for (int i = 0; i < W; i++) w += __popcll(q.v[i] | q.v[W + i]);
    return w;
}

// Mark the terms that survive a truncation event -- the coefficient threshold
// (thr <= 0 keeps all: the CPU rule discards |c| <= thr only when active) and
// the deferred weight cutoff (maxw at/above the qubit count keeps all);
// k_gather then compacts exactly the marked ones (same flag -> prefix-sum ->
// slot pipeline as emission).
template <int W>
__global__ void k_flag_keep(const Key<W>* keys, const double* coeffs, std::size_t n, double thr,
                            int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = ((thr <= 0 || fabs(coeffs[i]) > thr) && key_weight(keys[i]) <= maxw) ? 1u : 0u;
}

// Per-term contribution to the event's discarded-norm certificate: |c|^2 of
// terms the threshold drops AMONG those the weight rule keeps (delta_e is
// measured against the weight-only reference, so weight discards do not
// count). Summed by cub::DeviceReduce through a transform iterator -- exact
// accumulation of the discards themselves, immune to the cancellation a
// norm-before-minus-norm-after difference would suffer when the discarded
// mass is tiny next to the total.
template <int W>
struct Disc2Op {
    const Key<W>* keys;
    const double* coeffs;
    double thr;
    int maxw;
    __device__ double operator()(std::size_t i) const {
        const double m = fabs(coeffs[i]);
        return (key_weight(keys[i]) <= maxw && m <= thr) ? m * m : 0.0;
    }
};

// Compact the keep-flagged terms into keys_out/coeffs_out using the same
// prefix-sum slots as emission (here pos was scanned from the threshold flags).
template <int W>
__global__ void k_gather(const Key<W>* keys_in, const double* coeffs_in, std::size_t n,
                         const unsigned* pos, Key<W>* keys_out, double* coeffs_out) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const unsigned prev = i ? pos[i - 1] : 0u;
    if (pos[i] > prev) {
        keys_out[prev] = keys_in[i];
        coeffs_out[prev] = coeffs_in[i];
    }
}

// Recompute the product phase of each string pair on the device and raise
// `mismatch` if any disagrees with the host-computed expectation.
template <int W>
__global__ void k_phase_check(const Key<W>* a, const Key<W>* b, const int* expected,
                              std::size_t n, int* mismatch) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    if (jpow_mod4(a[i], b[i]) != expected[i]) atomicOr(mismatch, 1);
}

// --- engine --------------------------------------------------------------

// Key width W (64-bit words per Pauli plane) is a compile-time template so
// the per-word loops in the kernels unroll to straight-line code; the compiler
// emits one full kernel set for W=1 (<=64 qubits) and one for W=2 (<=128).
// That makes EngineT<1> and EngineT<2> two unrelated C++ types, yet the width
// is only known at runtime (from the circuit), so no single member variable
// could hold "whichever was chosen". This small virtual interface solves
// that: both instantiations derive from ImplBase, and one ImplBase pointer
// can hold either. The price is one virtual call per host-side operation --
// negligible next to the kernels it launches; nothing on the GPU is virtual.
struct ImplBase {
    virtual ~ImplBase() = default;
    virtual void upload(const std::uint64_t*, const double*, std::size_t) = 0;
    virtual void apply_rot(const std::uint64_t*, double, int) = 0;
    virtual double compact(double, int) = 0;
    virtual std::size_t size() const = 0;
    virtual void download(std::vector<std::uint64_t>&, std::vector<double>&) = 0;
};

template <int W>
struct EngineT final : ImplBase {
    // The term array is one allocation split into two regions: a sorted,
    // duplicate-free prefix of length `base`, then an unsorted `tail` of
    // freshly emitted partners. Emission grows the tail; compact() folds the
    // tail into a new, larger base. `cap` is the allocated term capacity.
    std::size_t cap = 0, base = 0, tail = 0;
    // Two interchangeable buffer sets: ka/ca is current, kb/cb is scratch.
    // Parallel merge and duplicate-summing cannot run in place, so they read
    // the current set and write the other; a swap then makes the result
    // current. The roles alternate ("ping-pong") across compactions.
    Key<W>*ka = nullptr, *kb = nullptr;
    double *ca = nullptr, *cb = nullptr;
    unsigned* pos = nullptr;  // per-term 0/1 flags, prefix-summed in place to slots
    int* d_nruns = nullptr;   // device int: deduplication writes its unique-key count here
    void* tmp = nullptr;      // shared CUB scratch buffer (see cub_tmp)
    std::size_t tmp_cap = 0;
    double* d_disc2 = nullptr;  // device double: the event's discarded |c|^2 sum
    // Page-locked ("pinned") host scalars that the GPU can write directly,
    // for reading single counts back cheaply.
    unsigned* h_count = nullptr;
    int* h_nruns = nullptr;
    double* h_disc2 = nullptr;

    explicit EngineT(std::size_t capacity_hint) {
        cuda_check(cudaFree(nullptr), "context init");  // fail early, clearly
        cuda_check(cudaMallocHost(&h_count, sizeof(unsigned)), "pinned alloc");
        cuda_check(cudaMallocHost(&h_nruns, sizeof(int)), "pinned alloc");
        cuda_check(cudaMallocHost(&h_disc2, sizeof(double)), "pinned alloc");
        cuda_check(cudaMalloc(&d_nruns, sizeof(int)), "device alloc");
        cuda_check(cudaMalloc(&d_disc2, sizeof(double)), "device alloc");
        grow(std::max<std::size_t>(4 * capacity_hint, std::size_t(1) << 16));
    }
    ~EngineT() override {
        cudaFree(ka), cudaFree(kb), cudaFree(ca), cudaFree(cb);
        cudaFree(pos), cudaFree(d_nruns), cudaFree(d_disc2), cudaFree(tmp);
        cudaFreeHost(h_count), cudaFreeHost(h_nruns), cudaFreeHost(h_disc2);
    }

    // Grow the buffers to hold at least `need` terms. Capacity doubles so the
    // total copying over all grows stays proportional to the final size (each
    // term moves at most twice on average) and grows stay rare. The old and
    // new allocations coexist during the copy, so a grow transiently needs
    // both at once -- cheap normally, costly near the memory ceiling.
    void grow(std::size_t need) {
        if (need <= cap) return;
        std::size_t ncap = std::max(need, 2 * cap);
        if (ncap > std::size_t(1) << 31)
            throw std::runtime_error("fastfermion gpu: term count exceeds the 2^31 backend limit");
        Key<W>*nka, *nkb;
        double *nca, *ncb;
        unsigned* npos;
        cuda_check(cudaMalloc(&nka, ncap * sizeof(Key<W>)), "device alloc (keys)");
        cuda_check(cudaMalloc(&nkb, ncap * sizeof(Key<W>)), "device alloc (keys)");
        cuda_check(cudaMalloc(&nca, ncap * sizeof(double)), "device alloc (coeffs)");
        cuda_check(cudaMalloc(&ncb, ncap * sizeof(double)), "device alloc (coeffs)");
        cuda_check(cudaMalloc(&npos, ncap * sizeof(unsigned)), "device alloc (offsets)");
        std::size_t n = base + tail;
        if (n) {  // only the current buffer holds live data
            cuda_check(cudaMemcpy(nka, ka, n * sizeof(Key<W>), cudaMemcpyDeviceToDevice), "grow");
            cuda_check(cudaMemcpy(nca, ca, n * sizeof(double), cudaMemcpyDeviceToDevice), "grow");
        }
        cudaFree(ka), cudaFree(kb), cudaFree(ca), cudaFree(cb), cudaFree(pos);
        ka = nka, kb = nkb, ca = nca, cb = ncb, pos = npos, cap = ncap;
    }

    // CUB routines need device scratch memory, and their API sizes it like
    // this: called with a NULL workspace pointer, a routine does no work and
    // only writes the byte count it needs into `bytes`; called again with a
    // real buffer, it executes. That is why every CUB operation in this file
    // appears as two consecutive identical-looking calls -- a size query and
    // then the actual run -- not a mistake. This helper provides the buffer,
    // regrown only when a request exceeds the largest seen, so the steady
    // state never allocates.
    void* cub_tmp(std::size_t bytes) {
        if (bytes > tmp_cap) {
            cudaFree(tmp);
            cuda_check(cudaMalloc(&tmp, bytes), "device alloc (cub workspace)");
            tmp_cap = bytes;
        }
        return tmp;
    }

    // Inclusive prefix sum (running totals) of pos[0:n) in place; the last
    // entry, read back, is the total number of set flags.
    unsigned scan_pos(std::size_t n) {
        std::size_t bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, bytes, pos, pos, n);         // NULL ptr: only size the scratch
        cub::DeviceScan::InclusiveSum(cub_tmp(bytes), bytes, pos, pos, n);  // real buffer: run the scan
        cuda_check(cudaMemcpy(h_count, pos + n - 1, sizeof(unsigned), cudaMemcpyDeviceToHost),
                   "scan readback");
        return *h_count;
    }

    void upload(const std::uint64_t* keys, const double* coeffs, std::size_t n) override {
        grow(2 * n);
        base = 0, tail = n;
        if (n) {
            cuda_check(cudaMemcpy(ka, keys, n * sizeof(Key<W>), cudaMemcpyHostToDevice), "upload");
            cuda_check(cudaMemcpy(ca, coeffs, n * sizeof(double), cudaMemcpyHostToDevice),
                       "upload");
            compact(0.0, 64 * W);  // sort + dedup once; the base stays sorted thereafter
        }
    }

    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree) override {
        std::size_t n = base + tail;
        if (n == 0) return;
        if (2 * n > cap) {  // worst case every term forks (n -> 2n): tidy first, grow only if still short
            compact(0.0, 64 * W);
            n = base;
            grow(2 * n);
        }
        Key<W> p;
        for (int k = 0; k < 2 * W; k++) p.v[k] = p_key[k];
        // Deterministic emission (see "--- kernels"): flag terms, reserve one
        // tail slot per surviving partner via the prefix sum, then emit.
        k_flag<W><<<n_blocks(n), BLOCK>>>(ka, n, p, maxdegree, pos);
        const unsigned emitted = scan_pos(n);
        k_emit<W><<<n_blocks(n), BLOCK>>>(ka, ca, n, p, std::cos(theta), std::sin(theta), pos);
        cuda_check(cudaGetLastError(), "emission");
        tail += emitted;
    }

    // Fold the emitted tail into the base so the whole array is sorted and
    // duplicate-free again: sort the tail, merge it into the sorted base, then
    // sum the coefficients of equal keys. With mincoeff > 0 a final pass drops
    // terms at or below the threshold. (Every CUB operation below appears as
    // two identical-looking calls: the first only sizes its scratch, the
    // second runs -- see cub_tmp.)
    double compact(double mincoeff, int maxw) override {
        std::size_t n = base + tail;
        if (n == 0) return 0.0;
        if (tail) {
            // Sort just the freshly emitted tail by key.
            std::size_t bytes = 0;
            cub::DeviceMergeSort::SortPairs(nullptr, bytes, ka + base, ca + base, tail,
                                            KeyLess<W>{});
            cub::DeviceMergeSort::SortPairs(cub_tmp(bytes), bytes, ka + base, ca + base, tail,
                                            KeyLess<W>{});
            if (base) {
                // Merge the sorted tail with the sorted base into the scratch
                // set (a merge cannot overwrite its own inputs in place).
                bytes = 0;
                cub::DeviceMerge::MergePairs(nullptr, bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<W>{});
                cub::DeviceMerge::MergePairs(cub_tmp(bytes), bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<W>{});
                std::swap(ka, kb), std::swap(ca, cb);  // merged result is now current
            }
            // Collapse equal adjacent keys, summing coefficients; the run count
            // (number of unique keys) lands in d_nruns. Reads current, writes scratch.
            bytes = 0;
            cub::DeviceReduce::ReduceByKey(nullptr, bytes, ka, kb, ca, cb, d_nruns, cub::Sum{}, n);
            cub::DeviceReduce::ReduceByKey(cub_tmp(bytes), bytes, ka, kb, ca, cb, d_nruns,
                                           cub::Sum{}, n);
            cuda_check(cudaMemcpy(h_nruns, d_nruns, sizeof(int), cudaMemcpyDeviceToHost),
                       "reduce readback");
            std::swap(ka, kb), std::swap(ca, cb);  // deduplicated result is now current
            base = static_cast<std::size_t>(*h_nruns);
            tail = 0;
        }
        double disc2 = 0.0;
        const bool w_active = maxw < 64 * W;  // weight never exceeds the qubit count
        if ((mincoeff > 0 || w_active) && base) {
            if (mincoeff > 0) {
                // The certificate's delta_e^2: reduce the dropped |c|^2 directly
                // (weight-surviving terms only) before the survivors are gathered.
                Disc2Op<W> op{ka, ca, mincoeff, w_active ? maxw : 64 * W};
                cub::CountingInputIterator<std::size_t> cnt(0);
                cub::TransformInputIterator<double, Disc2Op<W>, decltype(cnt)> it(cnt, op);
                std::size_t bytes = 0;
                cub::DeviceReduce::Sum(nullptr, bytes, it, d_disc2, base);
                cub::DeviceReduce::Sum(cub_tmp(bytes), bytes, it, d_disc2, base);
                cuda_check(cudaMemcpy(h_disc2, d_disc2, sizeof(double), cudaMemcpyDeviceToHost),
                           "disc2 readback");
                disc2 = *h_disc2;
            }
            // Truncation event via the same flag -> prefix-sum -> gather slot
            // pipeline as emission: flag terms to keep, scan to slots, gather.
            k_flag_keep<W><<<n_blocks(base), BLOCK>>>(ka, ca, base, mincoeff,
                                                      w_active ? maxw : 64 * W, pos);
            const unsigned kept = scan_pos(base);
            k_gather<W><<<n_blocks(base), BLOCK>>>(ka, ca, base, pos, kb, cb);
            cuda_check(cudaGetLastError(), "truncation event");
            std::swap(ka, kb), std::swap(ca, cb);
            base = kept;
        }
        return disc2;
    }

    std::size_t size() const override { return base + tail; }

    // Copy the term set back to the host, compacting first so what leaves the
    // device is sorted and duplicate-free.
    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs) override {
        compact(0.0, 64 * W);
        keys.resize(2 * W * base);
        coeffs.resize(base);
        if (base) {
            cuda_check(cudaMemcpy(keys.data(), ka, base * sizeof(Key<W>), cudaMemcpyDeviceToHost),
                       "download");
            cuda_check(cudaMemcpy(coeffs.data(), ca, base * sizeof(double), cudaMemcpyDeviceToHost),
                       "download");
        }
    }
};

}  // namespace

int device_count() {
    int n = 0;
    return (cudaGetDeviceCount(&n) == cudaSuccess) ? n : 0;
}

struct Engine::Impl {
    std::unique_ptr<ImplBase> e;
};

// Pick the compiled kernel set matching this circuit's key width (see the
// ImplBase note above).
Engine::Engine(int words, std::size_t capacity_hint) : impl(new Impl) {
    if (words == 1)
        impl->e = std::make_unique<EngineT<1>>(capacity_hint);
    else if (words == 2)
        impl->e = std::make_unique<EngineT<2>>(capacity_hint);
    else
        throw std::runtime_error("fastfermion gpu: only 1- or 2-word keys (<=128 qubits)");
}
Engine::~Engine() = default;

void Engine::upload(const std::uint64_t* k, const double* c, std::size_t n) {
    impl->e->upload(k, c, n);
}
void Engine::apply_rot(const std::uint64_t* p, double theta, int maxdegree) {
    impl->e->apply_rot(p, theta, maxdegree);
}
double Engine::compact(double mincoeff, int maxdegree) {
    return impl->e->compact(mincoeff, maxdegree);
}
std::size_t Engine::size() const { return impl->e->size(); }
void Engine::download(std::vector<std::uint64_t>& k, std::vector<double>& c) {
    impl->e->download(k, c);
}

// Ship n string pairs and their host-computed phases to the device, recompute
// there, and report whether every pair agreed.
template <int W>
static bool phase_check_impl(const std::uint64_t* a, const std::uint64_t* b, const int* expected,
                             std::size_t n) {
    Key<W>*da, *db;
    int *de, *dm;
    cuda_check(cudaMalloc(&da, n * sizeof(Key<W>)), "phase check");
    cuda_check(cudaMalloc(&db, n * sizeof(Key<W>)), "phase check");
    cuda_check(cudaMalloc(&de, n * sizeof(int)), "phase check");
    cuda_check(cudaMalloc(&dm, sizeof(int)), "phase check");
    cuda_check(cudaMemcpy(da, a, n * sizeof(Key<W>), cudaMemcpyHostToDevice), "phase check");
    cuda_check(cudaMemcpy(db, b, n * sizeof(Key<W>), cudaMemcpyHostToDevice), "phase check");
    cuda_check(cudaMemcpy(de, expected, n * sizeof(int), cudaMemcpyHostToDevice), "phase check");
    cuda_check(cudaMemset(dm, 0, sizeof(int)), "phase check");
    k_phase_check<W><<<n_blocks(n), BLOCK>>>(da, db, de, n, dm);
    int mismatch = 0;
    cuda_check(cudaMemcpy(&mismatch, dm, sizeof(int), cudaMemcpyDeviceToHost), "phase check");
    cudaFree(da), cudaFree(db), cudaFree(de), cudaFree(dm);
    return mismatch == 0;
}

bool phase_check(const std::uint64_t* a, const std::uint64_t* b, const int* expected_jpow_mod4,
                 std::size_t n, int words) {
    if (words == 1) return phase_check_impl<1>(a, b, expected_jpow_mod4, n);
    if (words == 2) return phase_check_impl<2>(a, b, expected_jpow_mod4, n);
    throw std::runtime_error("fastfermion gpu: only 1- or 2-word keys (<=128 qubits)");
}

}  // namespace gpu
}  // namespace pauli_gates
}  // namespace fastfermion
