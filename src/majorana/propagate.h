// Sparse Majorana dynamics: Heisenberg propagation of a MajoranaPolynomial
// through a circuit of Majorana rotations, with the same truncation rules,
// schedules, per-run certificate, and parallel backends as the Pauli driver
// (pauli/propagate.h). The transfer is exact: monomials are a keyed
// orthonormal basis, gates act entrywise and isometrically, and the
// structural cutoff reads only the key, so every rule/schedule/certificate
// argument goes through unchanged. Backends: serial (hash map), serial-merge
// (parallel emission, single-threaded rebuild -- the diagnostic baseline),
// sharded (key-ownership shards, all-parallel merge).
// The GPU engine port follows separately through the key-policy seam.

#pragma once
#include <climits>
#include <string>

#include "common.h"
#include "majorana/gates.h"
#include "majorana/truncate.h"

#ifdef FF_GPU
#include "pauli/propagate_gpu.h"
#endif

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {

namespace majorana_gates {

// For now, a MajoranaCircuit is simply a sequence of Majorana Rotations
using MajoranaCircuit = std::vector<MROT>;
using MajMap = decltype(MajoranaPolynomial{}.terms);

// Batching: a maximal run of consecutive gates whose generators pairwise
// commute forms one window; rules fire once per window on the cadence of
// truncate_all. Under the structural cutoff the batched and per-gate
// retained sets are identical (compaction invariance); under coefficient
// rules the window is part of the rule's schedule, as on the Pauli side.
inline bool _commutes_with_batch(const std::vector<const MROT*>& batch, const MROT& g) {
    for (const MROT* b : batch)
        if (!b->ms.commutes(g.ms)) return false;
    return true;
}

// Per-gate emission constants: e^{-i theta/2 M} with M = i^r ms sends an
// anticommuting monomial x to cos(theta) x + [i i^r sin(theta)] ms*x
// (MROT::apply_inplace); the two backends below repeat that arithmetic on
// their own storage layouts.
struct GateConsts {
    ff_float cos_t;
    ff_complex iirsin_t;
    explicit GateConsts(const MROT& g)
        : cos_t(std::cos(g.theta)),
          iirsin_t(g._r ? ff_complex(-std::sin(g.theta), 0) : ff_complex(0, std::sin(g.theta))) {}
};

#ifdef FF_OPENMP

// =========================================================================
// Per-gate conjugation: parallel emission, serial hash-map rebuild.
// Kept as the measured baseline showing why the merge must be partitioned:
// its rebuild is single-threaded, a floor no thread count clears. The
// sharded backend below is the production parallel path.
// =========================================================================
inline void conjugate_omp(MajoranaPolynomial& obs, const MROT& gate, int maxdegree, int n_threads,
                          std::vector<std::pair<MajoranaString, ff_complex>>& snap) {
    snap.clear();
    snap.insert(snap.end(), obs.terms.begin(), obs.terms.end());
    const GateConsts gc(gate);
    int chunk = (snap.size() + n_threads - 1) / n_threads;
    std::vector<std::vector<std::pair<MajoranaString, ff_complex>>> all_kept(n_threads);
    std::vector<std::vector<std::pair<MajoranaString, ff_complex>>> all_partners(n_threads);

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        all_kept[tid].reserve(chunk);
        all_partners[tid].reserve(chunk);
#pragma omp for schedule(static)
        for (int j = 0; j < (int)snap.size(); j++) {
            const auto& [ms_q, c] = snap[j];
            if (ms_q.commutes(gate.ms))
                all_kept[tid].emplace_back(ms_q, c);
            else {
                all_kept[tid].emplace_back(ms_q, c * gc.cos_t);
                MajoranaMonomial partner = gate.ms * ms_q;
                // Emission-time structural cutoff: the only enforcement at
                // period 1 (the deferred schedule passes maxdegree = INT_MAX).
                if (partner.s.degree() <= maxdegree)
                    all_partners[tid].emplace_back(partner.majorana_string(),
                                                   c * gc.iirsin_t * partner.coefficient());
            }
        }
    }
    obs.terms.clear();
    for (int t = 0; t < n_threads; t++) {
        for (const auto& [x, c] : all_kept[t]) obs.terms[x] += c;
        for (const auto& [x, c] : all_partners[t]) obs.terms[x] += c;
    }
}

// =========================================================================
// Per-gate conjugation: sharded hash map (all-parallel merge). The
// polynomial is partitioned into n_threads shards by hashing the monomial
// key; each thread owns one shard, partners route to their owner through
// persistent buffers (allocated once per propagation; rebuilding the t*t
// grid per gate is a measured t^2 scaling ceiling), and no thread writes
// another's shard.
// =========================================================================
using ShardedMajPoly = std::vector<MajMap>;
using MajSendBuf = std::vector<std::pair<MajoranaString, ff_complex>>;

inline int shard_of(const MajoranaString& ms, int n_shards) {
    return static_cast<int>(ms.hash() % static_cast<std::uint64_t>(n_shards));
}

inline ShardedMajPoly to_sharded(const MajoranaPolynomial& poly, int n_shards) {
    ShardedMajPoly shards(n_shards);
    for (const auto& [ms, c] : poly.terms) shards[shard_of(ms, n_shards)][ms] += c;
    return shards;
}

inline MajoranaPolynomial from_sharded(const ShardedMajPoly& shards) {
    MajoranaPolynomial out;
    for (const auto& shard : shards)
        for (const auto& [ms, c] : shard) out.terms[ms] += c;
    return out;
}

struct MajShardBuffers {
    std::vector<std::vector<MajSendBuf>> outgoing;
    std::vector<MajSendBuf> local;
    explicit MajShardBuffers(int n_threads)
        : outgoing(n_threads, std::vector<MajSendBuf>(n_threads)), local(n_threads) {}
};

inline void conjugate_sharded(ShardedMajPoly& shards, const MROT& gate, int maxdegree,
                              int n_threads, MajShardBuffers& buf) {
    const GateConsts gc(gate);
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        MajSendBuf& local_new = buf.local[tid];
        local_new.clear();
        for (auto& out : buf.outgoing[tid]) out.clear();

        for (auto& [x, coeff] : shard) {
            if (!x.commutes(gate.ms)) {
                MajoranaMonomial partner = gate.ms * x;
                // Emission-time structural cutoff, as in conjugate_omp.
                if (partner.s.degree() <= maxdegree) {
                    auto pk = partner.majorana_string();
                    auto pc = coeff * gc.iirsin_t * partner.coefficient();
                    int dest = shard_of(pk, n_threads);
                    if (dest == tid)
                        local_new.emplace_back(pk, pc);
                    else
                        buf.outgoing[tid][dest].emplace_back(pk, pc);
                }
                coeff *= gc.cos_t;
            }
        }
        for (const auto& [k, c] : local_new) shard[k] += c;
    }
    // Implicit barrier; each thread merges its incoming partners.
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        for (int src = 0; src < n_threads; src++)
            for (const auto& [k, c] : buf.outgoing[src][tid]) shard[k] += c;
    }
}

// Scheduled truncation on shards. Threshold and degree are separable per
// shard; firing decisions are made once and the discarded norm is reduced
// across shards, so one event contributes one delta_e to the certificate
// and the stats never see a data race. Top-K couples shards globally --
// merge, truncate, re-shard, as on the Pauli side.
inline void truncate_sharded(ShardedMajPoly& shards, int n_threads, ff_float mincoeff, int topk,
                             int rot_last, int maxdegree = INT_MAX, int maxdegree_period = 1,
                             int mincoeff_period = 1, int rot_first = -1) {
    if (rot_first < 0) rot_first = rot_last;
    if (topk > 0) {
        MajoranaPolynomial merged = from_sharded(shards);
        truncate_all(merged, mincoeff, topk, rot_last, maxdegree, maxdegree_period,
                     mincoeff_period, rot_first);
        shards = to_sharded(merged, n_threads);
        return;
    }
    const bool fire_w =
        maxdegree_period > 1 && period_crossed(rot_first, rot_last, maxdegree_period);
    const bool fire_tau = mincoeff > 0 && period_crossed(rot_first, rot_last, mincoeff_period);
    double disc2 = 0;
    std::size_t n_terms = 0;
#pragma omp parallel num_threads(n_threads) reduction(+ : disc2, n_terms)
    {
        int tid = omp_get_thread_num();
        MajoranaPolynomial tmp;
        tmp.terms = std::move(shards[tid]);
        n_terms += tmp.terms.size();
        if (fire_w) truncate_degree(tmp, maxdegree);  // before the threshold: delta_e
        if (fire_tau) disc2 += truncate_threshold(tmp, mincoeff);  // counts survivors only
        shards[tid] = std::move(tmp.terms);
    }
    auto& stats = trunc_stats();
    stats.peak_terms = std::max(stats.peak_terms, n_terms);
    if (fire_w) stats.n_w_events++;
    if (fire_tau) {
        stats.cert_tau += std::sqrt(disc2);
        stats.n_tau_events++;
    }
}

#endif  // FF_OPENMP

#ifdef FF_GPU
// =========================================================================
// GPU path. The Majorana monomial's index mask rides the engine's flat
// 2*words record unchanged (sparse_words = -1 selects the Majorana algebra
// on device; degree = mask popcount, so the engine's weight machinery is the
// degree cutoff). Coefficients cross as the real coefficients of the
// Hermitian monomials Gamma_S = i^{m(k)} gamma_S: fold i^{-m} at upload,
// unfold at download. Scheduling and the certificate follow the CPU cadence
// exactly, as on the Pauli side.
// =========================================================================

inline int _gpu_maj_words(const MajoranaCircuit& circuit, const MajoranaPolynomial& a) {
    int maxidx = 1;
    for (const auto& [x, c] : a.terms) maxidx = std::max(maxidx, x.extent());
    for (const auto& g : circuit) maxidx = std::max(maxidx, g.ms.extent());
    return (maxidx + 2 * WORD_LENGTH - 1) / (2 * WORD_LENGTH);  // record = 2*words u64
}

inline void _gpu_maj_flatten(const MajoranaString& ms, int words, std::uint64_t* out) {
    for (int k = 0; k < 2 * words; k++)
        out[k] = (k < (int)ms.alpha.words.size()) ? ms.alpha.words[k] : 0;
}

inline int _maj_mphase(int k) { return (k * (k - 1) / 2) & 1; }  // m(k)

// Saturating degree arena Sum_{k<=d} C(2M, k); the a-priori retained-set
// bound of the emission-enforced degree cutoff.
inline double _degree_arena(int n_idx, int d) {
    double total = 0, c = 1;
    for (int k = 0; k <= std::min(d, n_idx); k++) {
        total += c;
        if (total > 4e9) return 4e9;
        c = c * (n_idx - k) / (k + 1);
    }
    return total;
}

inline MajoranaPolynomial propagate_gpu_path(const MajoranaCircuit& circuit,
                                             const MajoranaPolynomial& a, int maxdegree,
                                             ff_float mincoeff, bool batched, int maxdegree_period,
                                             int mincoeff_period, long long reserve_terms,
                                             const std::string& gpu_key, double gpu_beta) {
    if (pauli_gates::gpu::device_count() == 0)
        throw_error("gpu backend: no CUDA device available");
    if (gpu_key != "auto" && gpu_key != "dense")
        throw_error("majorana gpu: gpu_key must be \"auto\" or \"dense\" (the support-list "
                    "key is not implemented for the Majorana algebra)");
    const int words = _gpu_maj_words(circuit, a);
    if (words > 2) throw_error("majorana gpu backend supports at most 128 modes");

    // One-shot arena reservation exactly as on the Pauli side: 2x the a-priori
    // bound never grows (auto, best-effort), expert sizing is honored hard,
    // and a deferred degree schedule has no bound so auto reserves nothing.
    std::size_t reserve = 0;
    bool reserve_hard = false;
    if (reserve_terms > 0) {
        reserve = 2 * std::size_t(reserve_terms);
        reserve_hard = true;
    } else if (reserve_terms < 0 && maxdegree_period == 1) {
        int n_idx = 0;
        for (const auto& g : circuit) n_idx = std::max(n_idx, g.ms.extent());
        for (const auto& [x, c] : a.terms) n_idx = std::max(n_idx, x.extent());
        const double arena = _degree_arena(n_idx, maxdegree);
        if (arena < 2e9) reserve = 2 * std::size_t(arena) + 64;
    }
    pauli_gates::gpu::Engine eng(words, a.terms.size(), reserve, reserve_hard,
                                 /*sparse_words=*/-1, gpu_beta);

    // Upload: real Hermitian-monomial coefficients (fold i^{-m(k)}).
    {
        std::vector<std::uint64_t> keys(2 * std::size_t(words) * a.terms.size());
        std::vector<double> coeffs(a.terms.size());
        std::size_t i = 0;
        for (const auto& [x, c] : a.terms) {
            _gpu_maj_flatten(x, words, &keys[2 * std::size_t(words) * i]);
            const ff_complex h = _maj_mphase(x.degree()) ? c * ff_complex(0, -1) : c;
            if (std::abs(h.imag()) > 1e-12 * (1 + std::abs(h.real())))
                throw_error("majorana gpu: observable is not self-adjoint");
            coeffs[i++] = h.real();
        }
        eng.upload(keys.data(), coeffs.data(), a.terms.size());
    }

    const int emit_deg = (maxdegree_period <= 1) ? maxdegree : 4 * WORD_LENGTH;
    int rot_count = 0;
    auto& stats = trunc_stats();
    auto event = [&](int rot_first, int rot_last) {
        const bool fire_w =
            maxdegree_period > 1 && period_crossed(rot_first, rot_last, maxdegree_period);
        const bool fire_tau = mincoeff > 0 && period_crossed(rot_first, rot_last, mincoeff_period);
        stats.peak_terms = std::max(stats.peak_terms, eng.size());
        if (!fire_w && !fire_tau) return;
        const double disc2 =
            eng.compact(fire_tau ? mincoeff : 0, fire_w ? maxdegree : 4 * WORD_LENGTH);
        if (fire_w) stats.n_w_events++;
        if (fire_tau) {
            stats.cert_tau += std::sqrt(disc2);
            stats.n_tau_events++;
        }
    };

    std::vector<std::uint64_t> pkey(2 * words);
    auto rot = [&](const MROT& g) {
        _gpu_maj_flatten(g.ms, words, pkey.data());
        eng.apply_rot(pkey.data(), g.theta, emit_deg);
    };
    int i = (int)circuit.size() - 1;
    while (i >= 0) {
        const int first = rot_count;
        int j = i;
        if (batched) {
            std::vector<const MROT*> batch{&circuit[i]};
            int k = i - 1;
            while (k >= 0 && _commutes_with_batch(batch, circuit[k])) {
                batch.push_back(&circuit[k]);
                k--;
            }
            j = k + 1;
        }
        for (int g = i; g >= j; g--) {
            rot(circuit[g]);
            rot_count++;
        }
        i = j - 1;
        event(first, rot_count - 1);
    }
    stats.peak_device_bytes = std::max(stats.peak_device_bytes, eng.peak_device_bytes());

    // Download: unfold the i^{m(k)} phase back onto gamma_S coefficients.
    MajoranaPolynomial out;
    {
        std::vector<std::uint64_t> keys;
        std::vector<double> coeffs;
        eng.download(keys, coeffs);
        const std::size_t n = coeffs.size();
        for (std::size_t t = 0; t < n; t++) {
            MajoranaString ms;
            for (int k = 0; k < 2 * words && k < (int)ms.alpha.words.size(); k++)
                ms.alpha.words[k] = keys[2 * std::size_t(words) * t + k];
            const ff_complex c = _maj_mphase(ms.degree())
                                     ? ff_complex(0, 1) * coeffs[t]
                                     : ff_complex(coeffs[t], 0);
            if (coeffs[t] != 0) out.terms[ms] += c;
        }
    }
    return out;
}
#endif  // FF_GPU

// =========================================================================
// Driver. Gate windows (one gate unbatched, one commuting batch batched)
// advance in Heisenberg order; the emission filter enforces the structural
// cutoff at period 1 and is lifted under a deferred schedule; truncate_*
// fires per window on each rule's cadence. Strategy strings are validated
// at entry so a typo can never silently return the un-evolved observable.
// =========================================================================
enum class Backend { serial, omp, sharded };

inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs,
                                    int maxdegree = INT_MAX, ff_float mincoeff = 0, int topk = 0,
                                    int maxdegree_period = 1, int mincoeff_period = 1,
                                    bool batched = true, int n_threads = 1,
                                    const std::string& parallel = "auto",
                                    long long reserve_terms = -1,
                                    const std::string& gpu_key = "auto", double gpu_beta = -1.0) {
    Backend be;
    if (parallel == "auto")
        be = (n_threads > 1) ? Backend::sharded : Backend::serial;
    else if (parallel == "serial")
        be = Backend::serial;
    else if (parallel == "serial-merge")
        be = Backend::omp;
    else if (parallel == "sharded")
        be = Backend::sharded;
    else if (parallel == "gpu") {
#ifdef FF_GPU
        trunc_stats().reset();
        return propagate_gpu_path(circuit, obs, maxdegree, mincoeff, batched, maxdegree_period,
                                  mincoeff_period, reserve_terms, gpu_key, gpu_beta);
#else
        throw_error("gpu backend requested but fastfermion was built without -Dgpu=enabled");
#endif
    } else
        throw_error("unknown parallel strategy '"
                    << parallel << "' (valid: auto, serial, serial-merge, sharded, gpu)");
#ifndef FF_OPENMP
    be = Backend::serial;
#endif
    if (n_threads <= 1) be = Backend::serial;

    trunc_stats().reset();
    const int L = (int)circuit.size();
    const int emit_deg = (maxdegree_period <= 1) ? maxdegree : INT_MAX;
    auto emit_ok = [emit_deg](const MajoranaString& s) { return s.degree() <= emit_deg; };

    MajoranaPolynomial ret(obs);
#ifdef FF_OPENMP
    ShardedMajPoly shards;
    MajShardBuffers buf(be == Backend::sharded ? n_threads : 1);
    std::vector<std::pair<MajoranaString, ff_complex>> snap;
    if (be == Backend::sharded) shards = to_sharded(ret, n_threads);
#endif

    int applied = 0;
    int i = L - 1;
    while (i >= 0) {
        int first = applied;
        // Determine the window [i .. j+1] (batched: maximal commuting run).
        int j = i;
        if (batched) {
            std::vector<const MROT*> batch{&circuit[i]};
            int k = i - 1;
            while (k >= 0 && _commutes_with_batch(batch, circuit[k])) {
                batch.push_back(&circuit[k]);
                k--;
            }
            j = k + 1;
        }
        for (int g = i; g >= j; g--) {
            switch (be) {
                case Backend::serial:
                    circuit[g].apply_inplace(ret, emit_ok);
                    break;
#ifdef FF_OPENMP
                case Backend::omp:
                    conjugate_omp(ret, circuit[g], emit_deg, n_threads, snap);
                    break;
                case Backend::sharded:
                    conjugate_sharded(shards, circuit[g], emit_deg, n_threads, buf);
                    break;
#else
                default:
                    break;
#endif
            }
            applied++;
        }
        i = j - 1;
#ifdef FF_OPENMP
        if (be == Backend::sharded) {
            truncate_sharded(shards, n_threads, mincoeff, topk, applied - 1, maxdegree,
                             maxdegree_period, mincoeff_period, first);
            continue;
        }
#endif
        truncate_all(ret, mincoeff, topk, applied - 1, maxdegree, maxdegree_period,
                     mincoeff_period, first);
    }
#ifdef FF_OPENMP
    if (be == Backend::sharded) return from_sharded(shards);
#endif
    return ret;
}




// Back-compatible overloads (the pre-port public surface).
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs,
                                    const ff_float& mincoeff) {
    return propagate(circuit, obs, INT_MAX, mincoeff);
}
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs,
                                    const ff_float& mincoeff = 0) {
    return propagate(circuit, MajoranaPolynomial(obs), INT_MAX, mincoeff);
}
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs,
                                    const int& maxdegree, const ff_float& mincoeff = 0) {
    return propagate(circuit, MajoranaPolynomial(obs), maxdegree, mincoeff);
}

}  // namespace majorana_gates

}  // namespace fastfermion
