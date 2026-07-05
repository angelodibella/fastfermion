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

template <int W>
__global__ void k_flag(const Key<W>* keys, std::size_t n, Key<W> p, int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const Key<W> q = keys[i];
    flag[i] = (anticommutes(p, q) && partner_weight(p, q) <= maxw) ? 1u : 0u;
}

// pos = inclusive prefix sum of the flags. Scales every anticommuting
// coefficient by cos(theta) in place (independent of the partner filter,
// matching the CPU conjugation) and appends surviving partners at the tail.
template <int W>
__global__ void k_emit(Key<W>* keys, double* coeffs, std::size_t n, Key<W> p, double cos_t,
                       double sin_t, const unsigned* pos) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const Key<W> q = keys[i];
    if (!anticommutes(p, q)) return;
    const double a = coeffs[i];
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

__global__ void k_flag_thresh(const double* coeffs, std::size_t n, double thr, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = (fabs(coeffs[i]) > thr) ? 1u : 0u;  // CPU rule discards |c| <= thr
}

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

template <int W>
__global__ void k_phase_check(const Key<W>* a, const Key<W>* b, const int* expected,
                              std::size_t n, int* mismatch) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    if (jpow_mod4(a[i], b[i]) != expected[i]) atomicOr(mismatch, 1);
}

// --- engine --------------------------------------------------------------

// Width-erased interface; EngineT<W> holds the device state.
struct ImplBase {
    virtual ~ImplBase() = default;
    virtual void upload(const std::uint64_t*, const double*, std::size_t) = 0;
    virtual void apply_rot(const std::uint64_t*, double, int) = 0;
    virtual void compact(double) = 0;
    virtual std::size_t size() const = 0;
    virtual void download(std::vector<std::uint64_t>&, std::vector<double>&) = 0;
};

template <int W>
struct EngineT final : ImplBase {
    std::size_t cap = 0, base = 0, tail = 0;
    Key<W>*ka = nullptr, *kb = nullptr;  // ping-pong key buffers (ka = current)
    double *ca = nullptr, *cb = nullptr;
    unsigned* pos = nullptr;  // per-term flags, scanned in place to offsets
    int* d_nruns = nullptr;
    void* tmp = nullptr;  // reusable cub workspace
    std::size_t tmp_cap = 0;
    unsigned* h_count = nullptr;  // pinned readback scalars
    int* h_nruns = nullptr;

    explicit EngineT(std::size_t capacity_hint) {
        cuda_check(cudaFree(nullptr), "context init");  // fail early, clearly
        cuda_check(cudaMallocHost(&h_count, sizeof(unsigned)), "pinned alloc");
        cuda_check(cudaMallocHost(&h_nruns, sizeof(int)), "pinned alloc");
        cuda_check(cudaMalloc(&d_nruns, sizeof(int)), "device alloc");
        grow(std::max<std::size_t>(4 * capacity_hint, std::size_t(1) << 16));
    }
    ~EngineT() override {
        cudaFree(ka), cudaFree(kb), cudaFree(ca), cudaFree(cb);
        cudaFree(pos), cudaFree(d_nruns), cudaFree(tmp);
        cudaFreeHost(h_count), cudaFreeHost(h_nruns);
    }

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

    void* cub_tmp(std::size_t bytes) {
        if (bytes > tmp_cap) {
            cudaFree(tmp);
            cuda_check(cudaMalloc(&tmp, bytes), "device alloc (cub workspace)");
            tmp_cap = bytes;
        }
        return tmp;
    }

    // Inclusive prefix sum of pos[0:n) in place; returns the total.
    unsigned scan_pos(std::size_t n) {
        std::size_t bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, bytes, pos, pos, n);
        cub::DeviceScan::InclusiveSum(cub_tmp(bytes), bytes, pos, pos, n);
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
            compact(0.0);  // sort + dedup once; the base stays sorted thereafter
        }
    }

    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree) override {
        std::size_t n = base + tail;
        if (n == 0) return;
        if (2 * n > cap) {  // deferred-dedup budget reached: compact, then grow if still short
            compact(0.0);
            n = base;
            grow(2 * n);
        }
        Key<W> p;
        for (int k = 0; k < 2 * W; k++) p.v[k] = p_key[k];
        k_flag<W><<<n_blocks(n), BLOCK>>>(ka, n, p, maxdegree, pos);
        const unsigned emitted = scan_pos(n);
        k_emit<W><<<n_blocks(n), BLOCK>>>(ka, ca, n, p, std::cos(theta), std::sin(theta), pos);
        cuda_check(cudaGetLastError(), "emission");
        tail += emitted;
    }

    void compact(double mincoeff) override {
        std::size_t n = base + tail;
        if (n == 0) return;
        if (tail) {
            std::size_t bytes = 0;
            cub::DeviceMergeSort::SortPairs(nullptr, bytes, ka + base, ca + base, tail,
                                            KeyLess<W>{});
            cub::DeviceMergeSort::SortPairs(cub_tmp(bytes), bytes, ka + base, ca + base, tail,
                                            KeyLess<W>{});
            if (base) {
                bytes = 0;
                cub::DeviceMerge::MergePairs(nullptr, bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<W>{});
                cub::DeviceMerge::MergePairs(cub_tmp(bytes), bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<W>{});
                std::swap(ka, kb), std::swap(ca, cb);  // merged now in current
            }
            bytes = 0;
            cub::DeviceReduce::ReduceByKey(nullptr, bytes, ka, kb, ca, cb, d_nruns, cub::Sum{}, n);
            cub::DeviceReduce::ReduceByKey(cub_tmp(bytes), bytes, ka, kb, ca, cb, d_nruns,
                                           cub::Sum{}, n);
            cuda_check(cudaMemcpy(h_nruns, d_nruns, sizeof(int), cudaMemcpyDeviceToHost),
                       "reduce readback");
            std::swap(ka, kb), std::swap(ca, cb);
            base = static_cast<std::size_t>(*h_nruns);
            tail = 0;
        }
        if (mincoeff > 0 && base) {
            k_flag_thresh<<<n_blocks(base), BLOCK>>>(ca, base, mincoeff, pos);
            const unsigned kept = scan_pos(base);
            k_gather<W><<<n_blocks(base), BLOCK>>>(ka, ca, base, pos, kb, cb);
            cuda_check(cudaGetLastError(), "threshold");
            std::swap(ka, kb), std::swap(ca, cb);
            base = kept;
        }
    }

    std::size_t size() const override { return base + tail; }

    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs) override {
        compact(0.0);
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
void Engine::compact(double mincoeff) { impl->e->compact(mincoeff); }
std::size_t Engine::size() const { return impl->e->size(); }
void Engine::download(std::vector<std::uint64_t>& k, std::vector<double>& c) {
    impl->e->download(k, c);
}

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
