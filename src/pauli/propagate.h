// Pauli propagation: truncated Heisenberg-picture evolution.
//
// Single entry point: propagate(). Dispatches to serial or OpenMP
// internally based on n_threads and whether FF_OPENMP is defined.
// Gate batching is on by default.

#pragma once
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
// Per-gate conjugation
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
            if (!gate.ps.commutes(g.ps)) { ok = false; break; }
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
                                 int n_threads = 1, int maxdegree = 128,
                                 ff_float mincoeff = 0, int topk = 0,
                                 int max_xweight = -1, int xtrunc_period = 1,
                                 bool batched = true) {
    // Clamp to serial when OpenMP is unavailable
#ifndef FF_OPENMP
    n_threads = 1;
#endif
    const bool use_omp [[maybe_unused]] = n_threads > 1;

    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    if (!batched) {
        // Per-gate truncation (baseline mode)
        for (int i = circuit.size() - 1; i >= 0; i--) {
            if (circuit[i].index() == 0) {
                if (!pending_clifford) { clifford_begin = i; pending_clifford = true; }
            } else if (circuit[i].index() == 1) {
                if (pending_clifford) {
                    _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                    pending_clifford = false;
                }
#ifdef FF_OPENMP
                if (use_omp) {
                    // Lazily initialised — only allocated when OMP is actually used
                    static thread_local std::vector<std::pair<PauliString, ff_complex>> snap;
                    conjugate_omp(obs, std::get<ROT>(circuit[i]), maxdegree, n_threads, snap);
                } else
#endif
                    conjugate(obs, std::get<ROT>(circuit[i]), maxdegree);
                truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
                rot_count++;
            }
        }
    } else {
        // Batched truncation (default)
        std::vector<ROT> rot_buffer;
#ifdef FF_OPENMP
        std::vector<std::pair<PauliString, ff_complex>> snap;
#endif

        auto flush_rots = [&]() {
            if (rot_buffer.empty()) return;
            auto batches = batch_commuting_gates(rot_buffer);
            for (const auto& batch : batches) {
                for (const auto& gate : batch) {
#ifdef FF_OPENMP
                    if (use_omp)
                        conjugate_omp(obs, gate, maxdegree, n_threads, snap);
                    else
#endif
                        conjugate(obs, gate, maxdegree);
                }
                rot_count += batch.size();
                truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
            }
            rot_buffer.clear();
        };

        for (int i = circuit.size() - 1; i >= 0; i--) {
            if (circuit[i].index() == 0) {
                flush_rots();
                if (!pending_clifford) { clifford_begin = i; pending_clifford = true; }
            } else if (circuit[i].index() == 1) {
                if (pending_clifford) {
                    _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                    pending_clifford = false;
                }
                rot_buffer.push_back(std::get<ROT>(circuit[i]));
            }
        }
        flush_rots();
    }

    if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    return obs;
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a,
                                 int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), 1, maxdegree);
}

}  // namespace pauli_gates
}  // namespace fastfermion
