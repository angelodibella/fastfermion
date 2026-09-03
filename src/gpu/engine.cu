/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// CUDA engine behind engine.h: a sorted term array with a tail of new terms deduplicated lazily.
//
// Per gate: flag the terms that anticommute with the generator and whose partner survives the
// degree cutoff, prefix-sum the flags into output slots, then one pass scales the flagged
// coefficients by cos(theta) in place and writes the partners into their slots at the end of the
// array. Per compaction: sort the tail (cub::DeviceMergeSort), merge it into the sorted base
// (cub::DeviceMerge), sum the coefficients of equal keys (cub::DeviceReduceByKey), then drop the
// terms a truncation rule rejects with the same flag / prefix-sum / gather pipeline.
//
// No fastfermion header is included: the identities duplicated here (commutation parity and product
// phase of Pauli strings, anticommutation and sign of Majorana monomials) are checked against the
// host by phase_check() and by the tests comparing the GPU and CPU propagations. Limits: <= 128
// qubits or <= 256 Majorana operators, i.e., words <= 2, and < 2^31 terms. Slots are reserved by
// prefix sums and reductions run in a fixed order, so the results are reproducible bit for bit.

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "engine.h"

namespace fastfermion {
namespace gpu {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("fastfermion gpu: ") + what + ": " +
                                 cudaGetErrorString(err));
}

constexpr int BLOCK = 256;
inline unsigned n_blocks(std::size_t n) { return static_cast<unsigned>((n + BLOCK - 1) / BLOCK); }

// --- dense keys ------------------------------------------------------------

// A Pauli string as W words of xory bits followed by W words of yorz bits
template <int W>
struct Key {
    unsigned long long v[2 * W];
    __host__ __device__ friend bool operator==(const Key& a, const Key& b) {
        for (int i = 0; i < 2 * W; i++)
            if (a.v[i] != b.v[i]) return false;
        return true;
    }
};

// Any fixed total order on keys does: most significant word first. Also valid for the sparse
// key, whose sorted slots make equal strings equal words.
template <class K>
struct KeyLess {
    __device__ bool operator()(const K& a, const K& b) const {
        constexpr int NW = sizeof(K) / 8;
        for (int i = NW - 1; i >= 0; i--)
            if (a.v[i] != b.v[i]) return a.v[i] < b.v[i];
        return false;
    }
};

// Symplectic inner-product parity (PauliString::commutes)
template <int W>
__device__ inline bool anticommutes(const Key<W>& p, const Key<W>& q) {
    int m = 0;
    for (int i = 0; i < W; i++)
        m += __popcll(p.v[i] & q.v[W + i]) + __popcll(p.v[W + i] & q.v[i]);
    return m & 1;
}

// Pauli weight of the product p*q, whose key is the XOR of the planes
template <int W>
__device__ inline int partner_weight(const Key<W>& p, const Key<W>& q) {
    int w = 0;
    for (int i = 0; i < W; i++) w += __popcll((p.v[i] ^ q.v[i]) | (p.v[W + i] ^ q.v[W + i]));
    return w;
}

// Exponent of i in the product p*q, mod 4 (pauli_string_multiply)
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

template <int W>
__device__ inline int key_weight(const Key<W>& q) {
    int w = 0;
    for (int i = 0; i < W; i++) w += __popcll(q.v[i] | q.v[W + i]);
    return w;
}

// --- support-list keys -----------------------------------------------------
//
// Under a degree cutoff w enforced at emission a string has at most w non-identity sites, so
// w slots of (site, letter) are a much smaller key than 2 bits per qubit: a slot is 9 bits,
// (site << 2) | letter with X=1, Y=2, Z=3, and 7 slots fit in a word (n <= 127, w <= 7 per word).
// The slots are sorted by site, empty slots (0x1FF) last, so equal strings are equal words and
// the compaction pipeline is unchanged; a gate acts on <= 2 sites.

constexpr unsigned SLOT_BITS = 9, SLOT_MASK = 0x1FF, SLOT_EMPTY = 0x1FF;
constexpr int SLOTS_PER_WORD = 7;

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

// A gate on the support-list path: <= 2 (site, letter) pairs, sites ascending
struct SGate {
    int site[2];
    unsigned letter[2];
    int nsites;
};

// Single-site algebra on letters through the (xory, yorz) bits of the dense encoding
__device__ inline unsigned letter_x(unsigned l) { return (l == 1 || l == 2) ? 1u : 0u; }
__device__ inline unsigned letter_z(unsigned l) { return (l == 2 || l == 3) ? 1u : 0u; }
__device__ inline unsigned letter_mul(unsigned a, unsigned b) {  // product letter (0 = identity)
    const unsigned x = letter_x(a) ^ letter_x(b), z = letter_z(a) ^ letter_z(b);
    return x ? (z ? 2u : 1u) : (z ? 3u : 0u);
}
// Contribution of one site to jpow_mod4, letters a (gate) and b (term)
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

// Partner key r = p*q by merging the gate's sites into the term's sorted slots; returns the
// number of slots of r (its weight), r being valid only if that fits the capacity
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
        while (gi < g.nsites && g.site[gi] < site) {  // gate sites before this term site
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

// --- Majorana keys ---------------------------------------------------------
//
// A Majorana monomial is a subset S of the Majorana operators; its key is the index mask, stored
// in the same 2*W-word record as a dense Pauli key, and its degree the popcount, so the whole
// pipeline is reused with the degree as the weight. Coefficients are those of the Hermitian
// monomials Gamma_S = i^{m(k)} gamma_S (m(k) = k(k-1)/2 mod 2, k = |S|), folded by the host, so
// that every gate factor is real as for Pauli strings.

template <int W>
__device__ inline int maj_weight(const Key<W>& q) {
    int w = 0;
    for (int i = 0; i < 2 * W; i++) w += __popcll(q.v[i]);
    return w;
}

// Gamma_P and Gamma_Q anticommute iff |P||Q| - |P n Q| is odd
template <int W>
__device__ inline bool maj_anticommutes(const Key<W>& p, const Key<W>& q) {
    int wp = 0, wq = 0, wi = 0;
    for (int i = 0; i < 2 * W; i++) {
        wp += __popcll(p.v[i]);
        wq += __popcll(q.v[i]);
        wi += __popcll(p.v[i] & q.v[i]);
    }
    return ((wp * wq - wi) & 1) != 0;
}

// The sign s in i Gamma_P Gamma_Q = s Gamma_{P xor Q} for anticommuting P, Q:
// s = i^{1 + m(p) + m(q) - m(r) + 2 x(P,Q)}, with x(P,Q) the number of pairs (mu in P, nu in Q)
// with mu > nu (gamma_P gamma_Q = (-1)^x gamma_{P xor Q}). x is counted by walking the few set
// bits of the gate mask P and counting the bits of Q below each.
template <int W>
__device__ inline double maj_sign(const Key<W>& p, const Key<W>& q, int wp, int wq, int wr) {
    int x = 0;
    int base = 0;  // bits of q in the words below word i
    int qbelow[2 * W];
    for (int i = 0; i < 2 * W; i++) {
        qbelow[i] = base;
        base += __popcll(q.v[i]);
    }
    for (int i = 0; i < 2 * W; i++) {
        unsigned long long pw = p.v[i];
        while (pw) {
            int b = __ffsll((long long)pw) - 1;
            unsigned long long below = (b == 0) ? 0ull : (q.v[i] & ((1ull << b) - 1ull));
            x += qbelow[i] + __popcll(below);
            pw &= pw - 1;
        }
    }
    auto m = [](int k) { return (k * (k - 1) / 2) & 1; };
    int jp = (1 + m(wp) + m(wq) + 3 * m(wr) + 2 * x) & 3;  // -m == 3m (mod 4)
    return (jp & 2) ? -1.0 : 1.0;
}

// --- key-format policies ---------------------------------------------------
//
// The kernels are templated on an Ops policy providing the key and gate types and the algebra
// (anticommutation, partner weight, partner key and sign, weight); the compaction pipeline is
// format-agnostic. kFlatNative says whether the flat host format is the device key itself.

template <int W>
struct DenseOps {
    using KeyT = Key<W>;
    using GateT = Key<W>;
    static __device__ bool ac(const GateT& p, const KeyT& q) { return anticommutes(p, q); }
    static __device__ int pweight(const GateT& p, const KeyT& q) { return partner_weight(p, q); }
    // Partner key and the sign of its coefficient: i sin(theta) i^jpow with jpow odd for
    // anticommuting strings, i.e., +-sin(theta)
    static __device__ double partner(const GateT& p, const KeyT& q, KeyT& r) {
        for (int k = 0; k < 2 * W; k++) r.v[k] = p.v[k] ^ q.v[k];
        return (jpow_mod4(p, q) & 2) ? 1.0 : -1.0;
    }
    static __device__ int weight(const KeyT& q) { return key_weight(q); }
    static constexpr bool kFlatNative = true;
    static GateT make_gate(const std::uint64_t* p_key, int) {
        GateT p;
        for (int k = 0; k < 2 * W; k++) p.v[k] = p_key[k];
        return p;
    }
};

template <int W>
struct MajoranaOps {
    using KeyT = Key<W>;
    using GateT = Key<W>;
    static __device__ bool ac(const GateT& p, const KeyT& q) { return maj_anticommutes(p, q); }
    static __device__ int pweight(const GateT& p, const KeyT& q) {
        int w = 0;
        for (int i = 0; i < 2 * W; i++) w += __popcll(p.v[i] ^ q.v[i]);
        return w;
    }
    static __device__ double partner(const GateT& p, const KeyT& q, KeyT& r) {
        int wp = 0, wq = 0;
        for (int i = 0; i < 2 * W; i++) {
            wp += __popcll(p.v[i]);
            wq += __popcll(q.v[i]);
            r.v[i] = p.v[i] ^ q.v[i];
        }
        int wr = 0;
        for (int i = 0; i < 2 * W; i++) wr += __popcll(r.v[i]);
        return maj_sign(p, q, wp, wq, wr);
    }
    static __device__ int weight(const KeyT& q) { return maj_weight(q); }
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
    // Terms arrive in the flat dense format and are converted on the device (k_encode / k_decode);
    // the gate, on <= 2 sites, is parsed on the host
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

// --- kernels ---------------------------------------------------------------
//
// Emission has no atomics: k_flag marks the terms, an inclusive prefix sum turns the marks into
// output slots (the k-th marked term gets slot k), and k_emit writes each partner to its own slot.
// The order of the appended terms, hence of the floating-point sums, is thus fixed.

// Marks term i if it anticommutes with the gate p and its partner has weight <= maxw
template <class Ops>
__global__ void k_flag(const typename Ops::KeyT* keys, std::size_t n, typename Ops::GateT p,
                       int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = (Ops::ac(p, keys[i]) && Ops::pweight(p, keys[i]) <= maxw) ? 1u : 0u;
}

// pos holds the prefix sums of the flags. Scales every anticommuting coefficient by cos(theta)
// in place (whether or not its partner passed the weight filter, as on the CPU) and writes the
// partners of the flagged terms into their slots after the n existing terms.
template <class Ops>
__global__ void k_emit(typename Ops::KeyT* keys, double* coeffs, std::size_t n,
                       typename Ops::GateT p, double cos_t, double sin_t, const unsigned* pos) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const typename Ops::KeyT q = keys[i];
    if (!Ops::ac(p, q)) return;
    const double a = coeffs[i];
    const unsigned prev = i ? pos[i - 1] : 0u;  // slot of this term; flagged iff pos[i] > prev
    if (pos[i] > prev) {
        typename Ops::KeyT r;
        const double sgn = Ops::partner(p, q, r);
        keys[n + prev] = r;
        coeffs[n + prev] = sgn * a * sin_t;
    }
    coeffs[i] = a * cos_t;
}

// Marks the terms a truncation event keeps: |c| > thr (thr <= 0 keeps all) and weight <= maxw
template <class Ops>
__global__ void k_flag_keep(const typename Ops::KeyT* keys, const double* coeffs, std::size_t n,
                            double thr, int maxw, unsigned* flag) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    flag[i] = ((thr <= 0 || fabs(coeffs[i]) > thr) && Ops::weight(keys[i]) <= maxw) ? 1u : 0u;
}

// |c|^2 of the terms the threshold discards among those the weight rule keeps, summed by
// cub::DeviceReduce through a transform iterator (the discards themselves, not a difference of
// norms, which would cancel)
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

// Compacts the flagged terms into keys_out/coeffs_out, pos holding the prefix sums of the flags
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

// Conversions between the flat dense format (2*words words per term) and the support-list key,
// once on upload and once on download
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

template <int W>
__global__ void k_phase_check(const Key<W>* a, const Key<W>* b, const int* expected,
                              std::size_t n, int* mismatch) {
    std::size_t i = std::size_t(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    if (jpow_mod4(a[i], b[i]) != expected[i]) atomicOr(mismatch, 1);
}

// --- engine ----------------------------------------------------------------

// The key width is a template parameter, so that the per-word loops unroll, but only known at
// run time: the instantiations share this virtual interface (one virtual call per host-side
// operation, nothing virtual on the device).
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
    int words_;    // flat words per bit-plane at the host interface
    double beta_;  // compact when tail > beta * base (< 0: only when the buffers are full)
    // The term array is a sorted, duplicate-free prefix of length base followed by an unsorted
    // tail of new terms; cap is the allocated capacity
    std::size_t cap = 0, base = 0, tail = 0;
    // Two buffer sets: ka/ca is current, kb/cb is scratch. Merging and duplicate-summing cannot run
    // in place, so they read the current set, write the other, and the two are swapped.
    K *ka = nullptr, *kb = nullptr;
    double *ca = nullptr, *cb = nullptr;
    unsigned* pos = nullptr;  // per-term flags, prefix-summed in place into slots
    int* d_nruns = nullptr;   // number of unique keys, written by the deduplication
    void* tmp = nullptr;      // CUB scratch buffer (see cub_tmp)
    std::size_t tmp_cap = 0;
    double* d_disc2 = nullptr;  // the discarded |c|^2 sum of a truncation event
    // Pinned host scalars for reading single counts back
    unsigned* h_count = nullptr;
    int* h_nruns = nullptr;
    double* h_disc2 = nullptr;
    // Every allocation of the engine is accounted here, so that repeated runs report the same
    // peak whatever else shares the device
    std::size_t cur_bytes = 0, peak_bytes = 0;
    void account(std::ptrdiff_t delta) {
        cur_bytes = std::size_t(std::ptrdiff_t(cur_bytes) + delta);
        peak_bytes = std::max(peak_bytes, cur_bytes);
    }
    // Device bytes of a capacity of c terms: two key buffers, two coefficient buffers, the flags
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
        // A reservation is honored if the term buffers take at most 70% of the free device memory
        // (the rest for the CUB scratch and other users), or unconditionally when hard; otherwise
        // the buffers grow on demand
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

    // Grows the buffers to hold at least need terms, doubling the capacity unless the doubling
    // would not fit in the free device memory, in which case the step is exact. Only the current
    // key/coefficient pair holds live data: the scratch set and the flags are freed first and
    // reallocated after, so old and new buffers coexist for one pair only.
    void grow(std::size_t need) {
        if (need <= cap) return;
        std::size_t ncap = std::max(need, 2 * cap);
        if (need <= (std::size_t(1) << 31) && ncap > need) {
            std::size_t free_b = 0, total_b = 0;
            cuda_check(cudaMemGetInfo(&free_b, &total_b), "meminfo");
            // The new full set plus the live pair must fit, with 10% slack, in the free memory
            // plus what freeing the scratch releases
            const std::size_t live = cap * (sizeof(K) + sizeof(double));
            if (term_bytes(ncap) + live > free_b / 10 * 9 + term_bytes(cap)) ncap = need;
        }
        if (ncap > std::size_t(1) << 31)
            throw std::runtime_error("fastfermion gpu: term count exceeds the 2^31 backend limit");
        cudaFree(kb), cudaFree(cb), cudaFree(pos);
        account(-std::ptrdiff_t(cap * (sizeof(K) + sizeof(double) + sizeof(unsigned))));
        K* nka;
        double* nca;
        cuda_check(cudaMalloc(&nka, ncap * sizeof(K)), "device alloc (keys)");
        cuda_check(cudaMalloc(&nca, ncap * sizeof(double)), "device alloc (coeffs)");
        account(ncap * (sizeof(K) + sizeof(double)));
        std::size_t n = base + tail;
        if (n) {
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

    // CUB routines are called twice: with a null workspace they only report the scratch bytes
    // they need, then they run with this buffer, regrown only when a request exceeds it
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

    // Inclusive prefix sum of pos[0:n) in place; returns the total, i.e., the number of set flags
    unsigned scan_pos(std::size_t n) {
        std::size_t bytes = 0;
        cuda_check(cub::DeviceScan::InclusiveSum(nullptr, bytes, pos, pos, n), "scan");
        cuda_check(cub::DeviceScan::InclusiveSum(cub_tmp(bytes), bytes, pos, pos, n), "scan");
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
            compact(0.0, INT_MAX);  // sort and deduplicate once; the base stays sorted
        }
    }

    void apply_rot(const std::uint64_t* p_key, double theta, int maxdegree) override {
        std::size_t n = base + tail;
        if (n == 0) return;
        if (beta_ >= 0 && base && double(tail) > beta_ * double(base)) {
            compact(0.0, INT_MAX);
            n = base;
        }
        const typename Ops::GateT p = Ops::make_gate(p_key, words_);
        // Flag, prefix-sum into slots, emit. The capacity is judged on the exact number of new
        // terms the prefix sum gives; if they do not fit, compact (or grow when there is no tail)
        // and flag again, as the slots are bound to the array. At most two retries.
        for (;;) {
            k_flag<Ops><<<n_blocks(n), BLOCK>>>(ka, n, p, maxdegree, pos);
            const unsigned emitted = scan_pos(n);
            if (n + emitted <= cap) {
                k_emit<Ops><<<n_blocks(n), BLOCK>>>(ka, ca, n, p, std::cos(theta),
                                                    std::sin(theta), pos);
                cuda_check(cudaGetLastError(), "emission");
                tail += emitted;
                return;
            }
            if (tail) {
                compact(0.0, INT_MAX);
                n = base;
            } else {
                grow(n + emitted);
            }
        }
    }

    double compact(double mincoeff, int maxw) override {
        std::size_t n = base + tail;
        if (n == 0) return 0.0;
        if (tail) {
            std::size_t bytes = 0;
            cuda_check(cub::DeviceMergeSort::SortPairs(nullptr, bytes, ka + base, ca + base, tail,
                                                       KeyLess<K>{}),
                       "sort");
            cuda_check(cub::DeviceMergeSort::SortPairs(cub_tmp(bytes), bytes, ka + base, ca + base,
                                                       tail, KeyLess<K>{}),
                       "sort");
            if (base) {
                bytes = 0;
                cuda_check(cub::DeviceMerge::MergePairs(nullptr, bytes, ka, ca,
                                                        static_cast<int>(base), ka + base, ca + base,
                                                        static_cast<int>(tail), kb, cb, KeyLess<K>{}),
                           "merge");
                cuda_check(cub::DeviceMerge::MergePairs(cub_tmp(bytes), bytes, ka, ca,
                                                        static_cast<int>(base), ka + base, ca + base,
                                                        static_cast<int>(tail), kb, cb, KeyLess<K>{}),
                           "merge");
                std::swap(ka, kb), std::swap(ca, cb);
            }
            bytes = 0;
            cuda_check(cub::DeviceReduce::ReduceByKey(nullptr, bytes, ka, kb, ca, cb, d_nruns,
                                                      cub::Sum{}, n),
                       "reduce by key");
            cuda_check(cub::DeviceReduce::ReduceByKey(cub_tmp(bytes), bytes, ka, kb, ca, cb, d_nruns,
                                                      cub::Sum{}, n),
                       "reduce by key");
            cuda_check(cudaMemcpy(h_nruns, d_nruns, sizeof(int), cudaMemcpyDeviceToHost),
                       "reduce readback");
            std::swap(ka, kb), std::swap(ca, cb);
            base = static_cast<std::size_t>(*h_nruns);
            tail = 0;
        }
        double disc2 = 0.0;
        const bool w_active = maxw < INT_MAX;  // INT_MAX: the degree rule is off
        if ((mincoeff > 0 || w_active) && base) {
            if (mincoeff > 0) {
                Disc2Op<Ops> op{ka, ca, mincoeff, maxw};
                cub::CountingInputIterator<std::size_t> cnt(0);
                cub::TransformInputIterator<double, Disc2Op<Ops>, decltype(cnt)> it(cnt, op);
                std::size_t bytes = 0;
                cuda_check(cub::DeviceReduce::Sum(nullptr, bytes, it, d_disc2, base), "reduce");
                cuda_check(cub::DeviceReduce::Sum(cub_tmp(bytes), bytes, it, d_disc2, base), "reduce");
                cuda_check(cudaMemcpy(h_disc2, d_disc2, sizeof(double), cudaMemcpyDeviceToHost),
                           "disc2 readback");
                disc2 = *h_disc2;
            }
            k_flag_keep<Ops><<<n_blocks(base), BLOCK>>>(ka, ca, base, mincoeff, maxw, pos);
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

    void download(std::vector<std::uint64_t>& keys, std::vector<double>& coeffs) override {
        compact(0.0, INT_MAX);
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

Engine::Engine(int words, std::size_t capacity_hint, std::size_t reserve_terms,
               bool reserve_hard, int key_format, double beta)
    : impl(new Impl) {
    if (key_format == -1 && words == 1)
        impl->e = std::make_unique<EngineT<MajoranaOps<1>>>(words, capacity_hint, reserve_terms,
                                                            reserve_hard, beta);
    else if (key_format == -1 && words == 2)
        impl->e = std::make_unique<EngineT<MajoranaOps<2>>>(words, capacity_hint, reserve_terms,
                                                            reserve_hard, beta);
    else if (key_format == 1)
        impl->e = std::make_unique<EngineT<SparseOps<1>>>(words, capacity_hint, reserve_terms,
                                                          reserve_hard, beta);
    else if (key_format == 2)
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
}  // namespace fastfermion
