// OpenMP-parallel Pauli propagation (hash-map backend).
// Falls back to serial when FF_OPENMP is not defined or n_threads == 1.

#pragma once

#include "common.h"
#include "pauli_propagation.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {
namespace pauli_gates {

PauliPolynomial propagate_omp(const Circuit& circuit, const PauliPolynomial& a, int n_threads = 1,
                              const int& maxdegree = 128, const ff_float& mincoeff = 0,
                              const int topk = 0, const int max_xweight = -1,
                              const int xtrunc_period = 1) {
    if (n_threads <= 1) return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);

#ifdef FF_OPENMP
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    // Parallel data structure must be indexable; also, multiple
    // threads editing the same hash map may lead to a race condition
    auto obs_snap =
        std::vector<std::pair<PauliString, ff_complex>>(obs.terms.begin(), obs.terms.end());

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

            // Clear but keep allocated memory
            obs_snap.clear();
            obs_snap.insert(obs_snap.end(), obs.terms.begin(), obs.terms.end());

            const ROT& gate = std::get<ROT>(circuit[i]);
            const PauliString& ps = gate.ps;
            const ff_float cos_t = cos(gate.theta);
            const ff_complex isin_t = ff_complex(0, sin(gate.theta));

            // One pair of vectors per thread
            int chunk = (obs_snap.size() + n_threads - 1) / n_threads;
            std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_kept(n_threads);
            std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_partners(n_threads);

#pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                auto& my_kept = all_kept[tid];
                auto& my_partners = all_partners[tid];
                my_kept.reserve(chunk);
                my_partners.reserve(chunk);

                // Static schedule since conversion from hash map essentially
                // randomly scatters commuting/anticommuting terms, so no load
                // imbalance
#pragma omp for schedule(static)
                for (int j = 0; j < (int)obs_snap.size(); j++) {
                    const auto& [ps_q, c] = obs_snap[j];

                    if (ps_q.commutes(ps))
                        my_kept.emplace_back(ps_q, c);
                    else {
                        my_kept.emplace_back(ps_q, c * cos_t);
                        PauliMonomial partner = ps * ps_q;
                        if (partner.degree_total() <= maxdegree)
                            my_partners.emplace_back(partner.pauli_string(),
                                                     c * isin_t * partner.coefficient());
                    }
                }
            }

            // Serial rebuild into hash map (see pauli_sorted.h for the sorted-array alternative)
            obs.terms.clear();
            for (int t = 0; t < n_threads; t++) {
                for (const auto& [x, coeff] : all_kept[t]) obs.terms[x] += coeff;
                for (const auto& [x, coeff] : all_partners[t]) obs.terms[x] += coeff;
            }

            // Truncate
            if (mincoeff > 0)
                std::erase_if(obs.terms, [&mincoeff](const auto& term) {
                    return std::abs(term.second) <= mincoeff;
                });
            if (topk > 0) truncate_top_k(obs, topk);
            rot_count++;
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                truncate_x_weight(obs, max_xweight);
        }
    }
    if (pending_clifford) {
        _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    }
    return obs;

#else
    return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#endif
}

}  // namespace pauli_gates
}  // namespace fastfermion
