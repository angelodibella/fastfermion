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
#include <cstddef>
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
// Generic over the key type: both the dense Key<W> and the sparse SKey<S>
// store their words in a member array v, and for the sparse key the sorted-
// slot canonical form makes bitwise order a valid total order too.
template <class K>
struct KeyLess {
    __device__ bool operator()(const K& a, const K& b) const {
        constexpr int NW = sizeof(K) / 8;
        for (int i = NW - 1; i >= 0; i--)
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

// --- support-list ("sparse") keys ------------------------------------------
//
// Under an emission-enforced weight cutoff a retained string has at most w
// non-identity sites, so the dense 2-bits-per-qubit key is far from tight:
// w slots of (site, letter) suffice. Slot = 9 bits, (site << 2) | letter with
// letters X=1, Y=2, Z=3 (the xory/yorz planes' per-site bit pairs; 0 marks an
// EMPTY slot only). 7 slots fit one 64-bit word: n <= 127, w <= 7 in a single
// word against the dense format's four at n = 100 — and the sort and merge
// are bandwidth-bound, so bytes moved is the cost that matters. Slots are
// kept sorted by site with empty slots (0x1FF, sorting last) at the end, so
// equal strings are bitwise-equal keys and any word comparison is a valid
// total order — the compaction pipeline is unchanged. The gate acts on <= 2
// sites, so the per-term algebra is an O(slots) register walk.
constexpr unsigned SLOT_BITS = 9, SLOT_MASK = 0x1FF, SLOT_EMPTY = 0x1FF;
constexpr int SLOTS_PER_WORD = 7;  // floor(64 / 9)

template <int S>
struct SKey {
    unsigned long long v[S];
    __host__ __device__ friend bool operator==(const SKey& a, const SKey& b) {
        for (int i = 0; i < S; i++)
            if (a.v[i] != b.v[i]) return false;
        return true;
    }
};

template <int S>
__device__ __host__ inline unsigned skey_get(const SKey<S>& k, int i) {
    return (k.v[i / SLOTS_PER_WORD] >> (SLOT_BITS * (i % SLOTS_PER_WORD))) & SLOT_MASK;
}
template <int S>
__device__ __host__ inline void skey_set(SKey<S>& k, int i, unsigned slot) {
    const int w = i / SLOTS_PER_WORD, sh = SLOT_BITS * (i % SLOTS_PER_WORD);
    k.v[w] = (k.v[w] & ~(static_cast<unsigned long long>(SLOT_MASK) << sh)) |
             (static_cast<unsigned long long>(slot) << sh);
}
template <int S>
__device__ __host__ inline SKey<S> skey_blank() {
    SKey<S> k;
    for (int i = 0; i < S; i++) k.v[i] = 0;
    for (int i = 0; i < S * SLOTS_PER_WORD; i++) skey_set(k, i, SLOT_EMPTY);
    return k;
}
template <int S>
__device__ inline int skey_weight(const SKey<S>& k) {
    int w = 0;
    while (w < S * SLOTS_PER_WORD && skey_get(k, w) != SLOT_EMPTY) w++;
    return w;
}

// A gate for the sparse path: <= 2 (site, letter) pairs, sites ascending.
struct SGate {
    int site[2];
    unsigned letter[2];
    int nsites;
};

// Per-site letter algebra via the same (xory, yorz) bit pair the dense path
// uses: letter -> (x = letter & 1 flipped...) — encode: X=1=(x1,z0), Y=2=(x1,z1),
// Z=3=(x0,z1). x-bit = (letter != 3), z-bit = (letter != 1).
__device__ inline unsigned letter_x(unsigned l) { return (l == 1 || l == 2) ? 1u : 0u; }
__device__ inline unsigned letter_z(unsigned l) { return (l == 2 || l == 3) ? 1u : 0u; }
__device__ inline unsigned letter_mul(unsigned a, unsigned b) {  // product letter (0 = identity)
    const unsigned x = letter_x(a) ^ letter_x(b), z = letter_z(a) ^ letter_z(b);
    return x ? (z ? 2u : 1u) : (z ? 3u : 0u);
}
// Single-site contribution to the dense jpow identity, letters a (gate), b (term).
__device__ inline int letter_jpow(unsigned a, unsigned b) {
    const unsigned r = letter_mul(a, b);
    const int ax = letter_x(a), az = letter_z(a), bx = letter_x(b), bz = letter_z(b);
    const int rx = r ? letter_x(r) : 0, rz = r ? letter_z(r) : 0;
    return (ax & az) + (bx & bz) + 2 * (az & bx) - (rx & rz);
}

template <int S>
__device__ inline bool s_anticommutes(const SGate& g, const SKey<S>& q) {
    int m = 0;
    for (int i = 0, gi = 0; i < S * SLOTS_PER_WORD && gi < g.nsites; i++) {
        const unsigned slot = skey_get(q, i);
        if (slot == SLOT_EMPTY) break;
        const int site = slot >> 2;
        while (gi < g.nsites && g.site[gi] < site) gi++;
        if (gi < g.nsites && g.site[gi] == site) {
            const unsigned a = g.letter[gi], b = slot & 3u;
            m += (letter_x(a) & letter_z(b)) ^ (letter_z(a) & letter_x(b));
        }
    }
    return m & 1;
}

// Partner key r = p*q as a sorted-slot merge of the gate's <= 2 sites into the
// term's slot list; returns the slot count (the partner's weight), with the
// key valid only when the count fits the capacity (the caller's weight filter
// guarantees this in sparse mode, where the cutoff is emission-enforced).
template <int S>
__device__ inline int s_partner(const SGate& g, const SKey<S>& q, SKey<S>& r, int& jp) {
    r = skey_blank<S>();
    int out = 0, gi = 0;
    jp = 0;
    const int cap = S * SLOTS_PER_WORD;
    for (int i = 0; i < cap; i++) {
        const unsigned slot = skey_get(q, i);
        if (slot == SLOT_EMPTY) break;
        const int site = slot >> 2;
        // gate sites strictly before this term site: inserted as new slots
        while (gi < g.nsites && g.site[gi] < site) {
            if (out < cap) skey_set(r, out, (unsigned(g.site[gi]) << 2) | g.letter[gi]);
            out++, jp += letter_jpow(g.letter[gi], 0u), gi++;
        }
        if (gi < g.nsites && g.site[gi] == site) {  // shared site: letters multiply
            const unsigned b = slot & 3u, a = g.letter[gi];
            const unsigned m = letter_mul(a, b);
            jp += letter_jpow(a, b);
            if (m) {
                if (out < cap) skey_set(r, out, (unsigned(site) << 2) | m);
                out++;
            }
            gi++;
        } else {  // gate absent here: term slot passes through
            if (out < cap) skey_set(r, out, slot);
            out++;
        }
    }
    while (gi < g.nsites) {  // gate sites past the last term slot
        if (out < cap) skey_set(r, out, (unsigned(g.site[gi]) << 2) | g.letter[gi]);
        out++, jp += letter_jpow(g.letter[gi], 0u), gi++;
    }
    jp = ((jp % 4) + 4) % 4;
    return out;
}

// --- kernels -------------------------------------------------------------

// Emission per gate runs as three steps with no atomics and no locks: k_flag
// marks each term 0/1; an inclusive prefix sum (running totals) turns the
// marks into output slots -- the k-th marked term reserves slot k -- and
// k_emit writes only into its own reserved slot, so no two threads ever share
// a write target. Slots are reserved up front (instead of the simpler shared
// atomic counter) for reproducibility: it fixes the append order, and
// floating-point sums depend on their order, so every run is bit-identical.

// The kernels below are templated on an Ops policy -- the one seam between
// the compaction pipeline (format-agnostic: sort, merge, reduce, gather) and
// the key format's algebra. DenseOps wraps the bit-plane functions above;
// SparseOps wraps the slot-walk functions. Each Ops provides the Key and
// Gate types plus anticommutes / partner_weight / partner / weight.

template <int W>
struct DenseOps {
    using KeyT = Key<W>;
    using GateT = Key<W>;
    static __device__ bool ac(const GateT& p, const KeyT& q) { return anticommutes(p, q); }
    static __device__ int pweight(const GateT& p, const KeyT& q) { return partner_weight(p, q); }
    // Partner key + the sign of its coefficient: a * (i sin) * i^jpow with
    // jpow odd for anticommuting pairs, so the factor is +-sin (see k_emit).
    static __device__ double partner(const GateT& p, const KeyT& q, KeyT& r) {
        for (int k = 0; k < 2 * W; k++) r.v[k] = p.v[k] ^ q.v[k];
        return (jpow_mod4(p, q) & 2) ? 1.0 : -1.0;
    }
    static __device__ int weight(const KeyT& q) { return key_weight(q); }
    // Host-side seam helpers: the flat wire format IS the dense key layout.
    static constexpr bool kFlatNative = true;
    static GateT make_gate(const std::uint64_t* p_key, int) {
        GateT p;
        for (int k = 0; k < 2 * W; k++) p.v[k] = p_key[k];
        return p;
    }
};

template <int S>
struct SparseOps {
    using KeyT = SKey<S>;
    using GateT = SGate;
    static __device__ bool ac(const GateT& g, const KeyT& q) { return s_anticommutes(g, q); }
    static __device__ int pweight(const GateT& g, const KeyT& q) {
        KeyT r;
        int jp;
        return s_partner(g, q, r, jp);
    }
    static __device__ double partner(const GateT& g, const KeyT& q, KeyT& r) {
        int jp;
        s_partner(g, q, r, jp);
        return (jp & 2) ? 1.0 : -1.0;
    }
    static __device__ int weight(const KeyT& q) { return skey_weight(q); }
    // Host-side seam helpers: terms cross as flat dense words and convert on
    // device (k_encode / k_decode); the gate (<= 2 sites) parses on the host.
    static constexpr bool kFlatNative = false;
    static GateT make_gate(const std::uint64_t* p_key, int words) {
        GateT g;
        g.nsites = 0;
        for (int w = 0; w < words; w++) {
            std::uint64_t x = p_key[w], z = p_key[words + w], any = x | z;
            while (any && g.nsites < 2) {
                const int b = __builtin_ctzll(any);
                const unsigned xb = (x >> b) & 1u, zb = (z >> b) & 1u;
                g.site[g.nsites] = 64 * w + b;
                g.letter[g.nsites] = xb ? (zb ? 2u : 1u) : 3u;
                g.nsites++;
                any &= any - 1;
            }
        }
        return g;
    }
};

// Mark term i when it anticommutes with the gate p (so ROT rotates it) and the
// partner it would spawn survives the degree cutoff (weight <= maxw).
template <class Ops>
__global__ void k_flag(const typename Ops::KeyT* keys, std::size_t n, typename Ops::GateT p,
                       int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = (Ops::ac(p, keys[i]) && Ops::pweight(p, keys[i]) <= maxw) ? 1u : 0u;
}

// pos now holds the prefix sums from k_flag. Scale every anticommuting
// coefficient by cos(theta) in place (done for all anticommuting terms,
// independent of the weight filter, matching the CPU conjugation); a term that
// also passed the filter writes its partner into its reserved tail slot.
template <class Ops>
__global__ void k_emit(typename Ops::KeyT* keys, double* coeffs, std::size_t n,
                       typename Ops::GateT p, double cos_t, double sin_t, const unsigned* pos) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const typename Ops::KeyT q = keys[i];
    if (!Ops::ac(p, q)) return;
    const double a = coeffs[i];
    // pos[i-1] counts the flagged terms before i, which is this term's
    // reserved slot; the flag itself is recovered as pos[i] > pos[i-1].
    const unsigned prev = i ? pos[i - 1] : 0u;
    if (pos[i] > prev) {
        typename Ops::KeyT r;
        // Partner coefficient a * (i sin) * i^jpow = a * sin * i^(jpow+1);
        // jpow is odd for anticommuting pairs, so the factor is +-sin --
        // Ops::partner returns that sign with the partner key.
        const double sgn = Ops::partner(p, q, r);
        keys[n + prev] = r;
        coeffs[n + prev] = sgn * a * sin_t;
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
template <class Ops>
__global__ void k_flag_keep(const typename Ops::KeyT* keys, const double* coeffs, std::size_t n,
                            double thr, int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = ((thr <= 0 || fabs(coeffs[i]) > thr) && Ops::weight(keys[i]) <= maxw) ? 1u : 0u;
}

// Per-term contribution to the event's discarded-norm certificate: |c|^2 of
// terms the threshold drops AMONG those the weight rule keeps (delta_e is
// measured against the weight-only reference, so weight discards do not
// count). Summed by cub::DeviceReduce through a transform iterator -- exact
// accumulation of the discards themselves, immune to the cancellation a
// norm-before-minus-norm-after difference would suffer when the discarded
// mass is tiny next to the total.
template <class Ops>
struct Disc2Op {
    const typename Ops::KeyT* keys;
    const double* coeffs;
    double thr;
    int maxw;
    __device__ double operator()(std::size_t i) const {
        const double m = fabs(coeffs[i]);
        return (Ops::weight(keys[i]) <= maxw && m <= thr) ? m * m : 0.0;
    }
};

// Compact the keep-flagged terms into keys_out/coeffs_out using the same
// prefix-sum slots as emission (here pos was scanned from the threshold flags).
template <class K>
__global__ void k_gather(const K* keys_in, const double* coeffs_in, std::size_t n,
                         const unsigned* pos, K* keys_out, double* coeffs_out) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const unsigned prev = i ? pos[i - 1] : 0u;
    if (pos[i] > prev) {
        keys_out[prev] = keys_in[i];
        coeffs_out[prev] = coeffs_in[i];
    }
}

// Boundary conversion for the sparse engine: the host seam always speaks the
// flat dense format (2*W words per term, [xory..., yorz...]), so the sparse
// engine converts once on upload and once on download. O(n_qubits) per term,
// twice per run -- irrelevant next to the per-gate pipeline it accelerates.
template <int S>
__global__ void k_encode(const unsigned long long* flat, int words, std::size_t n,
                         SKey<S>* out) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    SKey<S> k = skey_blank<S>();
    int slot = 0;
    for (int w = 0; w < words; w++) {
        unsigned long long x = flat[2 * std::size_t(words) * i + w];
        unsigned long long z = flat[2 * std::size_t(words) * i + words + w];
        unsigned long long any = x | z;
        while (any) {
            const int b = __ffsll(any) - 1;
            const unsigned xb = (x >> b) & 1u, zb = (z >> b) & 1u;
            const unsigned letter = xb ? (zb ? 2u : 1u) : 3u;
            if (slot < S * SLOTS_PER_WORD)
                skey_set(k, slot, (unsigned(64 * w + b) << 2) | letter);
            slot++;
            any &= any - 1;
        }
    }
    out[i] = k;
}

template <int S>
__global__ void k_decode(const SKey<S>* in, std::size_t n, int words,
                         unsigned long long* flat) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    for (int w = 0; w < 2 * words; w++) flat[2 * std::size_t(words) * i + w] = 0;
    for (int sl = 0; sl < S * SLOTS_PER_WORD; sl++) {
        const unsigned slot = skey_get(in[i], sl);
        if (slot == SLOT_EMPTY) break;
        const int site = slot >> 2, w = site / 64, b = site % 64;
        const unsigned letter = slot & 3u;
        if (letter == 1 || letter == 2)
            flat[2 * std::size_t(words) * i + w] |= 1ull << b;
        if (letter == 2 || letter == 3)
            flat[2 * std::size_t(words) * i + words + w] |= 1ull << b;
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
    virtual std::size_t peak_device_bytes() const = 0;
    virtual void download(std::vector<std::uint64_t>&, std::vector<double>&) = 0;
};

template <class Ops>
struct EngineT final : ImplBase {
    using K = typename Ops::KeyT;
    int words_;      // flat words per Pauli plane at the host seam
    double beta_;    // tail budget: compact when tail > beta * base (< 0 = capacity-driven)
    // The term array is one allocation split into two regions: a sorted,
    // duplicate-free prefix of length `base`, then an unsorted `tail` of
    // freshly emitted partners. Emission grows the tail; compact() folds the
    // tail into a new, larger base. `cap` is the allocated term capacity.
    std::size_t cap = 0, base = 0, tail = 0;
    // Two interchangeable buffer sets: ka/ca is current, kb/cb is scratch.
    // Parallel merge and duplicate-summing cannot run in place, so they read
    // the current set and write the other; a swap then makes the result
    // current. The roles alternate ("ping-pong") across compactions.
    K *ka = nullptr, *kb = nullptr;
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
    // Deterministic allocator accounting: every cudaMalloc/cudaFree the engine
    // makes updates cur_bytes; peak_bytes is the run's memory metric (a
    // counter, not a cudaMemGetInfo sample, so repeated runs report
    // identical numbers regardless of what else shares the device).
    std::size_t cur_bytes = 0, peak_bytes = 0;
    void account(std::ptrdiff_t delta) {
        cur_bytes = std::size_t(std::ptrdiff_t(cur_bytes) + delta);
        peak_bytes = std::max(peak_bytes, cur_bytes);
    }
    // Device bytes for a capacity of c term slots: two key buffers, two
    // coefficient buffers, one flag/slot buffer.
    static std::size_t term_bytes(std::size_t c) {
        return c * (2 * sizeof(K) + 2 * sizeof(double) + sizeof(unsigned));
    }

    EngineT(int words, std::size_t capacity_hint, std::size_t reserve_terms,
            bool reserve_hard, double beta)
        : words_(words), beta_(beta) {
        cuda_check(cudaFree(nullptr), "context init");  // fail early, clearly
        cuda_check(cudaMallocHost(&h_count, sizeof(unsigned)), "pinned alloc");
        cuda_check(cudaMallocHost(&h_nruns, sizeof(int)), "pinned alloc");
        cuda_check(cudaMallocHost(&h_disc2, sizeof(double)), "pinned alloc");
        cuda_check(cudaMalloc(&d_nruns, sizeof(int)), "device alloc");
        cuda_check(cudaMalloc(&d_disc2, sizeof(double)), "device alloc");
        account(sizeof(int) + sizeof(double));
        // Best-effort (auto) reserve: honored only when it leaves real
        // headroom -- the term buffers may take at most 70% of free device
        // memory, the rest covering CUB scratch (O(resident terms)) and other
        // users of the device; an unhonored request falls back to the growth
        // path, so correctness never depends on the reservation. A HARD
        // (user-sized) reserve skips the check: the expert asked for exactly
        // this, and a loud allocation failure beats a silently ignored size.
        std::size_t start = std::max<std::size_t>(4 * capacity_hint, std::size_t(1) << 16);
        if (reserve_terms > start && reserve_terms <= (std::size_t(1) << 31)) {
            if (reserve_hard) {
                start = reserve_terms;
            } else {
                std::size_t free_b = 0, total_b = 0;
                cuda_check(cudaMemGetInfo(&free_b, &total_b), "meminfo");
                if (term_bytes(reserve_terms) <= free_b / 10 * 7) start = reserve_terms;
            }
        }
        grow(start);
    }
    ~EngineT() override {
        cudaFree(ka), cudaFree(kb), cudaFree(ca), cudaFree(cb);
        cudaFree(pos), cudaFree(d_nruns), cudaFree(d_disc2), cudaFree(tmp);
        cudaFreeHost(h_count), cudaFreeHost(h_nruns), cudaFreeHost(h_disc2);
    }

    // Grow the buffers to hold at least `need` terms. Capacity doubles so the
    // total copying over all grows stays proportional to the final size (each
    // term moves at most twice on average) and grows stay rare -- but when a
    // doubling would not fit in free device memory, the step falls back to the
    // exact `need`, trading future regrows for reachability at the ceiling.
    // Only the live key/coefficient pair is copied; the scratch set and the
    // flag buffer hold no live data across a grow, so they are FREED FIRST and
    // reallocated after. The transient old+new coexistence is thus one
    // buffer pair, not all five -- the allocation spike that made the largest
    // runs die in grow() rather than at their true resident size.
    void grow(std::size_t need) {
        if (need <= cap) return;
        std::size_t ncap = std::max(need, 2 * cap);
        if (need <= (std::size_t(1) << 31) && ncap > need) {
            std::size_t free_b = 0, total_b = 0;
            cuda_check(cudaMemGetInfo(&free_b, &total_b), "meminfo");
            // Freed-first scratch releases term_bytes(cap) minus the live
            // pair; require the new full set plus the still-held live pair to
            // fit with ~10% slack, else take the exact step.
            const std::size_t live = cap * (sizeof(K) + sizeof(double));
            if (term_bytes(ncap) + live > free_b / 10 * 9 + term_bytes(cap)) ncap = need;
        }
        if (ncap > std::size_t(1) << 31)
            throw std::runtime_error("fastfermion gpu: term count exceeds the 2^31 backend limit");
        // Free the dead buffers before allocating their larger replacements.
        cudaFree(kb), cudaFree(cb), cudaFree(pos);
        account(-std::ptrdiff_t(cap * (sizeof(K) + sizeof(double) + sizeof(unsigned))));
        K* nka;
        double* nca;
        cuda_check(cudaMalloc(&nka, ncap * sizeof(K)), "device alloc (keys)");
        cuda_check(cudaMalloc(&nca, ncap * sizeof(double)), "device alloc (coeffs)");
        account(ncap * (sizeof(K) + sizeof(double)));
        std::size_t n = base + tail;
        if (n) {  // only the current buffer holds live data
            cuda_check(cudaMemcpy(nka, ka, n * sizeof(K), cudaMemcpyDeviceToDevice), "grow");
            cuda_check(cudaMemcpy(nca, ca, n * sizeof(double), cudaMemcpyDeviceToDevice), "grow");
        }
        cudaFree(ka), cudaFree(ca);
        account(-std::ptrdiff_t(cap * (sizeof(K) + sizeof(double))));
        ka = nka, ca = nca;
        cuda_check(cudaMalloc(&kb, ncap * sizeof(K)), "device alloc (keys)");
        cuda_check(cudaMalloc(&cb, ncap * sizeof(double)), "device alloc (coeffs)");
        cuda_check(cudaMalloc(&pos, ncap * sizeof(unsigned)), "device alloc (offsets)");
        account(std::ptrdiff_t(ncap * (sizeof(K) + sizeof(double) + sizeof(unsigned))));
        cap = ncap;
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
            account(-std::ptrdiff_t(tmp_cap));
            cuda_check(cudaMalloc(&tmp, bytes), "device alloc (cub workspace)");
            account(std::ptrdiff_t(bytes));
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
            if constexpr (Ops::kFlatNative) {
                cuda_check(cudaMemcpy(ka, keys, n * sizeof(K), cudaMemcpyHostToDevice), "upload");
            } else {
                // Terms arrive in the flat dense wire format; convert on device.
                const std::size_t fb = 2 * std::size_t(words_) * n * sizeof(std::uint64_t);
                unsigned long long* flat;
                cuda_check(cudaMalloc(&flat, fb), "device alloc (upload staging)");
                account(fb);
                cuda_check(cudaMemcpy(flat, keys, fb, cudaMemcpyHostToDevice), "upload");
                k_encode<sizeof(K) / 8><<<n_blocks(n), BLOCK>>>(flat, words_, n, ka);
                cuda_check(cudaGetLastError(), "encode");
                cudaFree(flat);
                account(-std::ptrdiff_t(fb));
            }
            cuda_check(cudaMemcpy(ca, coeffs, n * sizeof(double), cudaMemcpyHostToDevice),
                       "upload");
            compact(0.0, 64 * words_);  // sort + dedup once; the base stays sorted thereafter
        }
    }

    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree) override {
        std::size_t n = base + tail;
        if (n == 0) return;
        // Tail budget: a plain dedup (no threshold, no weight event -- free to
        // schedule by the compaction-invariance proposition) once the unsorted
        // tail outgrows beta * base. beta = 0 compacts every gate (the
        // cuPauliProp-shaped cadence); beta < 0 defers to the capacity check
        // below. prop:batch-length predicts the optimum beta*.
        if (beta_ >= 0 && base && double(tail) > beta_ * double(base)) {
            compact(0.0, 64 * words_);
            n = base;
        }
        if (2 * n > cap) {  // worst case every term forks (n -> 2n): tidy first, grow only if still short
            compact(0.0, 64 * words_);
            n = base;
            grow(2 * n);
        }
        const typename Ops::GateT p = Ops::make_gate(p_key, words_);
        // Deterministic emission (see "--- kernels"): flag terms, reserve one
        // tail slot per surviving partner via the prefix sum, then emit.
        k_flag<Ops><<<n_blocks(n), BLOCK>>>(ka, n, p, maxdegree, pos);
        const unsigned emitted = scan_pos(n);
        k_emit<Ops><<<n_blocks(n), BLOCK>>>(ka, ca, n, p, std::cos(theta), std::sin(theta), pos);
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
                                            KeyLess<K>{});
            cub::DeviceMergeSort::SortPairs(cub_tmp(bytes), bytes, ka + base, ca + base, tail,
                                            KeyLess<K>{});
            if (base) {
                // Merge the sorted tail with the sorted base into the scratch
                // set (a merge cannot overwrite its own inputs in place).
                bytes = 0;
                cub::DeviceMerge::MergePairs(nullptr, bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<K>{});
                cub::DeviceMerge::MergePairs(cub_tmp(bytes), bytes, ka, ca, static_cast<int>(base),
                                             ka + base, ca + base, static_cast<int>(tail), kb, cb,
                                             KeyLess<K>{});
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
        const bool w_active = maxw < 64 * words_;  // weight never exceeds the qubit count
        if ((mincoeff > 0 || w_active) && base) {
            if (mincoeff > 0) {
                // The certificate's delta_e^2: reduce the dropped |c|^2 directly
                // (weight-surviving terms only) before the survivors are gathered.
                Disc2Op<Ops> op{ka, ca, mincoeff, w_active ? maxw : 64 * words_};
                cub::CountingInputIterator<std::size_t> cnt(0);
                cub::TransformInputIterator<double, Disc2Op<Ops>, decltype(cnt)> it(cnt, op);
                std::size_t bytes = 0;
                cub::DeviceReduce::Sum(nullptr, bytes, it, d_disc2, base);
                cub::DeviceReduce::Sum(cub_tmp(bytes), bytes, it, d_disc2, base);
                cuda_check(cudaMemcpy(h_disc2, d_disc2, sizeof(double), cudaMemcpyDeviceToHost),
                           "disc2 readback");
                disc2 = *h_disc2;
            }
            // Truncation event via the same flag -> prefix-sum -> gather slot
            // pipeline as emission: flag terms to keep, scan to slots, gather.
            k_flag_keep<Ops><<<n_blocks(base), BLOCK>>>(ka, ca, base, mincoeff,
                                                        w_active ? maxw : 64 * words_, pos);
            const unsigned kept = scan_pos(base);
            k_gather<K><<<n_blocks(base), BLOCK>>>(ka, ca, base, pos, kb, cb);
            cuda_check(cudaGetLastError(), "truncation event");
            std::swap(ka, kb), std::swap(ca, cb);
            base = kept;
        }
        return disc2;
    }

    std::size_t size() const override { return base + tail; }

    std::size_t peak_device_bytes() const override { return peak_bytes; }

    // Copy the term set back to the host, compacting first so what leaves the
    // device is sorted and duplicate-free.
    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs) override {
        compact(0.0, 64 * words_);
        keys.resize(2 * std::size_t(words_) * base);
        coeffs.resize(base);
        if (base) {
            if constexpr (Ops::kFlatNative) {
                cuda_check(
                    cudaMemcpy(keys.data(), ka, base * sizeof(K), cudaMemcpyDeviceToHost),
                    "download");
            } else {
                const std::size_t fb = 2 * std::size_t(words_) * base * sizeof(std::uint64_t);
                unsigned long long* flat;
                cuda_check(cudaMalloc(&flat, fb), "device alloc (download staging)");
                account(fb);
                k_decode<sizeof(K) / 8><<<n_blocks(base), BLOCK>>>(ka, base, words_, flat);
                cuda_check(cudaGetLastError(), "decode");
                cuda_check(cudaMemcpy(keys.data(), flat, fb, cudaMemcpyDeviceToHost), "download");
                cudaFree(flat);
                account(-std::ptrdiff_t(fb));
            }
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
Engine::Engine(int words, std::size_t capacity_hint, std::size_t reserve_terms,
               bool reserve_hard, int sparse_words, double beta)
    : impl(new Impl) {
    // sparse_words = 0 selects the dense bit-plane key; 1 or 2 selects the
    // support-list key with that many words (7 slots per word). The host
    // driver owns the eligibility rules; here the request is taken as given.
    if (sparse_words == 1)
        impl->e = std::make_unique<EngineT<SparseOps<1>>>(words, capacity_hint, reserve_terms,
                                                          reserve_hard, beta);
    else if (sparse_words == 2)
        impl->e = std::make_unique<EngineT<SparseOps<2>>>(words, capacity_hint, reserve_terms,
                                                          reserve_hard, beta);
    else if (words == 1)
        impl->e = std::make_unique<EngineT<DenseOps<1>>>(words, capacity_hint, reserve_terms,
                                                         reserve_hard, beta);
    else if (words == 2)
        impl->e = std::make_unique<EngineT<DenseOps<2>>>(words, capacity_hint, reserve_terms,
                                                         reserve_hard, beta);
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
std::size_t Engine::peak_device_bytes() const { return impl->e->peak_device_bytes(); }
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
