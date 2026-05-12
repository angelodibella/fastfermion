// Pauli propagation: truncated Heisenberg-picture evolution.
//
// Single entry point: propagate(). The `parallel` parameter selects the
// parallelisation strategy:
//
//   "serial"   — baseline, one thread
//   "omp"      — parallel emission, serial hash-map rebuild
//   "sharded"  — sharded hash-map, all-parallel merge
//   "auto"     — serial when n_threads=1, sharded when n_threads>1
//
// Gate batching is on by default.

#pragma once
#include <string>
#include <vector>

#include "common.h"
#include "pauli/gates.h"
#include "pauli/truncate.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {
namespace pauli_gates {

// =========================================================================
// Per-gate conjugation: serial
// =========================================================================

inline void conjugate(PauliPolynomial& obs, const ROT& gate, int maxdegree) {
    std::vector<std::pair<PauliString, ff_complex>> new_terms;
    new_terms.reserve(obs.terms.size());

    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    for (auto& [x, coeff] : obs.terms) {
        if (!x.commutes(ps)) {
            PauliMonomial partner = ps * x;
            if (partner.degree_total() <= maxdegree)
                new_terms.emplace_back(partner.pauli_string(),
                                       coeff * isin_t * partner.coefficient());
            coeff *= cos_t;
        }
    }
    for (const auto& [x, coeff] : new_terms) obs.terms[x] += coeff;
}

// =========================================================================
// Per-gate conjugation: OMP (parallel emission, serial rebuild)
// =========================================================================

#ifdef FF_OPENMP

inline void conjugate_omp(PauliPolynomial& obs, const ROT& gate, int maxdegree, int n_threads,
                          std::vector<std::pair<PauliString, ff_complex>>& snap) {
    snap.clear();
    snap.insert(snap.end(), obs.terms.begin(), obs.terms.end());

    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    int chunk = (snap.size() + n_threads - 1) / n_threads;
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_kept(n_threads);
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_partners(n_threads);

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        all_kept[tid].reserve(chunk);
        all_partners[tid].reserve(chunk);

#pragma omp for schedule(static)
        for (int j = 0; j < (int)snap.size(); j++) {
            const auto& [ps_q, c] = snap[j];
            if (ps_q.commutes(ps))
                all_kept[tid].emplace_back(ps_q, c);
            else {
                all_kept[tid].emplace_back(ps_q, c * cos_t);
                PauliMonomial partner = ps * ps_q;
                if (partner.degree_total() <= maxdegree)
                    all_partners[tid].emplace_back(partner.pauli_string(),
                                                   c * isin_t * partner.coefficient());
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
// Per-gate conjugation: sharded hash-map (all-parallel merge)
//
// The polynomial is partitioned into n_threads shards by hashing the
// PauliString key. Each thread owns one shard. After emission, partners
// destined for other shards are exchanged via thread-local buffers.
// No serial phase: merging is embarrassingly parallel.
// =========================================================================

using ShardedPoly = std::vector<PauliPolynomialMap>;
using SendBuf = std::vector<std::pair<PauliString, ff_complex>>;

inline int shard_of(const PauliString& ps, int n_shards) {
    return static_cast<int>(ps.hash() % static_cast<std::uint64_t>(n_shards));
}

inline ShardedPoly to_sharded(const PauliPolynomial& poly, int n_shards) {
    ShardedPoly shards(n_shards);
    for (const auto& [ps, c] : poly.terms) shards[shard_of(ps, n_shards)][ps] += c;
    return shards;
}

inline PauliPolynomial from_sharded(const ShardedPoly& shards) {
    PauliPolynomial out;
    for (const auto& shard : shards)
        for (const auto& [ps, c] : shard) out.terms[ps] += c;
    return out;
}

inline void conjugate_sharded(ShardedPoly& shards, const ROT& gate, int maxdegree, int n_threads) {
    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    // outgoing[src][dst]: partners from thread src destined for thread dst
    std::vector<std::vector<SendBuf>> outgoing(n_threads, std::vector<SendBuf>(n_threads));

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        SendBuf local_new;

        for (auto& [x, coeff] : shard) {
            if (!x.commutes(ps)) {
                PauliMonomial partner = ps * x;
                if (partner.degree_total() <= maxdegree) {
                    auto pk = partner.pauli_string();
                    auto pc = coeff * isin_t * partner.coefficient();
                    int dest = shard_of(pk, n_threads);
                    if (dest == tid)
                        local_new.emplace_back(pk, pc);
                    else
                        outgoing[tid][dest].emplace_back(pk, pc);
                }
                coeff *= cos_t;
            }
        }

        // Insert partners that stayed in this shard
        for (const auto& [k, c] : local_new) shard[k] += c;
    }
    // Implicit barrier

    // Each thread merges incoming partners from all other threads
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        for (int src = 0; src < n_threads; src++) {
            for (const auto& [k, c] : outgoing[src][tid]) shard[k] += c;
        }
    }
}

inline void truncate_sharded(ShardedPoly& shards, int n_threads, ff_float mincoeff, int topk,
                             int max_xweight, int xtrunc_period, int rot_count) {
    if (topk > 0) {
        // Top-K requires a global view — merge, truncate, re-shard
        PauliPolynomial merged = from_sharded(shards);
        truncate_all(merged, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
        shards = to_sharded(merged, n_threads);
        return;
    }
    // Per-shard truncation (threshold and x-weight are separable)
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        PauliPolynomial tmp;
        tmp.terms = std::move(shards[tid]);
        truncate_all(tmp, mincoeff, 0, max_xweight, xtrunc_period, rot_count);
        shards[tid] = std::move(tmp.terms);
    }
}

#endif  // FF_OPENMP

// =========================================================================
// Gate batching
// =========================================================================

inline std::vector<std::vector<ROT>> batch_commuting_gates(const std::vector<ROT>& gates) {
    std::vector<std::vector<ROT>> batches;
    std::vector<ROT> current;
    for (const auto& gate : gates) {
        bool ok = true;
        for (const auto& g : current)
            if (!gate.ps.commutes(g.ps)) {
                ok = false;
                break;
            }
        if (ok)
            current.push_back(gate);
        else {
            if (!current.empty()) batches.push_back(std::move(current));
            current = {gate};
        }
    }
    if (!current.empty()) batches.push_back(std::move(current));
    return batches;
}

// =========================================================================
// Public API
// =========================================================================

inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                                 int n_threads = 1, int maxdegree = 128, ff_float mincoeff = 0,
                                 int topk = 0, int max_xweight = -1, int xtrunc_period = 1,
                                 bool batched = true, const std::string& parallel = "auto") {
#ifndef FF_OPENMP
    n_threads = 1;
#endif

    // Resolve "auto": serial when single-threaded, sharded otherwise
    std::string strategy = parallel;
    if (strategy == "auto") strategy = (n_threads > 1) ? "sharded" : "serial";
    if (n_threads <= 1) strategy = "serial";

    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    // -----------------------------------------------------------------
    // Serial path
    // -----------------------------------------------------------------
    if (strategy == "serial") {
        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    conjugate(obs, std::get<ROT>(circuit[i]), maxdegree);
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch) conjugate(obs, gate, maxdegree);
                    rot_count += batch.size();
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }
        if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        return obs;
    }

#ifdef FF_OPENMP
    // -----------------------------------------------------------------
    // OMP path (parallel emission, serial rebuild)
    // -----------------------------------------------------------------
    if (strategy == "omp") {
        std::vector<std::pair<PauliString, ff_complex>> snap;
        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    conjugate_omp(obs, std::get<ROT>(circuit[i]), maxdegree, n_threads, snap);
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch)
                        conjugate_omp(obs, gate, maxdegree, n_threads, snap);
                    rot_count += batch.size();
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }
        if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        return obs;
    }

    // -----------------------------------------------------------------
    // Sharded path (all-parallel merge)
    // -----------------------------------------------------------------
    if (strategy == "sharded") {
        ShardedPoly shards = to_sharded(obs, n_threads);

        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        obs = from_sharded(shards);
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        shards = to_sharded(obs, n_threads);
                        pending_clifford = false;
                    }
                    conjugate_sharded(shards, std::get<ROT>(circuit[i]), maxdegree, n_threads);
                    truncate_sharded(shards, n_threads, mincoeff, topk, max_xweight, xtrunc_period,
                                     rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch)
                        conjugate_sharded(shards, gate, maxdegree, n_threads);
                    rot_count += batch.size();
                    truncate_sharded(shards, n_threads, mincoeff, topk, max_xweight, xtrunc_period,
                                     rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        obs = from_sharded(shards);
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        shards = to_sharded(obs, n_threads);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }

        if (pending_clifford) {
            obs = from_sharded(shards);
            _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
            return obs;
        }
        return from_sharded(shards);
    }
#endif  // FF_OPENMP

    return obs;
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a,
                                 int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), 1, maxdegree);
}

}  // namespace pauli_gates
}  // namespace fastfermion
