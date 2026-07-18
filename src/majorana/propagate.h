// Sparse Majorana dynamics: Heisenberg propagation of a MajoranaPolynomial
// through a circuit of Majorana rotations, with the same truncation rules,
// schedules, per-run certificate, and parallel backends as the Pauli driver
// (pauli/propagate.h) -- the transfer is licensed by workbook
// rem:keyed-systems: monomials are a keyed orthonormal basis, gates act
// entrywise and isometrically, and the structural cutoff reads only the key.
// Backends: serial (hash map), omp (parallel emission, serial merge -- the
// diagnostic baseline), sharded (key-ownership shards, all-parallel merge).
// The GPU engine port follows separately through the key-policy seam.

#pragma once
#include <climits>
#include <string>

#include "common.h"
#include "majorana/gates.h"
#include "majorana/truncate.h"

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
// Kept as the measured baseline showing why the merge must be partitioned
// (workbook sec:merge-profile); the sharded backend below is the production
// parallel path, exactly as on the Pauli side.
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
// persistent buffers (allocated once per propagation -- the transient-grid
// lesson of workbook sec:h2d-plateau), and no thread writes another's shard.
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
                                    const std::string& parallel = "auto") {
    Backend be;
    if (parallel == "auto")
        be = (n_threads > 1) ? Backend::sharded : Backend::serial;
    else if (parallel == "serial")
        be = Backend::serial;
    else if (parallel == "omp")
        be = Backend::omp;
    else if (parallel == "sharded")
        be = Backend::sharded;
    else
        throw_error("unknown parallel strategy '" << parallel
                                                  << "' (valid: auto, serial, omp, sharded)");
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
