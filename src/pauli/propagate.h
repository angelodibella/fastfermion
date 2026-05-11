// Pauli propagation: per-gate conjugation strategies, gate batching, and entry points.

#pragma once
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include "common.h"
#include "pauli/gates.h"
#include "pauli/sorted.h"
#include "pauli/truncate.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {

namespace pauli_gates {

// --- Per-gate conjugation: serial hash-map ---

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

// --- Per-gate conjugation: OMP hash-map (parallel emit, serial rebuild) ---

inline void conjugate_omp(PauliPolynomial& obs, const ROT& gate, int maxdegree, int n_threads,
                          std::vector<std::pair<PauliString, ff_complex>>& obs_snap) {
    obs_snap.clear();
    obs_snap.insert(obs_snap.end(), obs.terms.begin(), obs.terms.end());

    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

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

    obs.terms.clear();
    for (int t = 0; t < n_threads; t++) {
        for (const auto& [x, coeff] : all_kept[t]) obs.terms[x] += coeff;
        for (const auto& [x, coeff] : all_partners[t]) obs.terms[x] += coeff;
    }
}

#endif  // FF_OPENMP

// --- Gate batching: group commuting ROTs and apply each batch in one pass ---

inline std::vector<std::vector<ROT>> batch_commuting_gates(const std::vector<ROT>& gates) {
    std::vector<std::vector<ROT>> batches;
    std::vector<ROT> current;

    for (const auto& gate : gates) {
        bool commutes_with_all = true;
        for (const auto& g : current) {
            if (!gate.ps.commutes(g.ps)) {
                commutes_with_all = false;
                break;
            }
        }
        if (commutes_with_all)
            current.push_back(gate);
        else {
            if (!current.empty()) batches.push_back(std::move(current));
            current = {gate};
        }
    }
    if (!current.empty()) batches.push_back(std::move(current));
    return batches;
}

inline void apply_batch(PauliPolynomial& obs, const std::vector<ROT>& batch, int maxdegree) {
    for (const auto& gate : batch) {
        const PauliString& ps = gate.ps;
        const ff_float cos_t = cos(gate.theta);
        const ff_complex isin_t = ff_complex(0, sin(gate.theta));

        std::vector<std::pair<PauliString, ff_complex>> gate_partners;
        gate_partners.reserve(obs.terms.size());

        for (auto& [x, coeff] : obs.terms) {
            if (!x.commutes(ps)) {
                PauliMonomial partner = ps * x;
                if (partner.degree_total() <= maxdegree)
                    gate_partners.emplace_back(partner.pauli_string(),
                                               coeff * isin_t * partner.coefficient());
                coeff *= cos_t;
            }
        }

        for (const auto& [k, c] : gate_partners) obs.terms[k] += c;
    }
}

// --- Propagation loop: shared gate iteration with Clifford batching ---

template <typename Obs, typename ConjugateFn>
PauliPolynomial propagate_loop(const Circuit& circuit, const PauliPolynomial& a,
                               ConjugateFn conjugate_gate, int maxdegree, ff_float mincoeff,
                               int topk, int max_xweight, int xtrunc_period) {
    Obs obs = obs_from_poly<Obs>(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                PauliPolynomial hm = obs_to_poly(obs);
                _apply_clifford_circuit(hm, circuit, i + 1, clifford_begin + 1);
                obs = obs_from_poly<Obs>(hm);
                pending_clifford = false;
            }
            const ROT& gate = std::get<ROT>(circuit[i]);
            conjugate_gate(obs, gate);
            truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
            rot_count++;
        }
    }
    if (pending_clifford) {
        PauliPolynomial hm = obs_to_poly(obs);
        _apply_clifford_circuit(hm, circuit, 0, clifford_begin + 1);
        return hm;
    }
    return obs_to_poly(obs);
}

// --- Public entry points ---

inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                                 const int& maxdegree = 128, const ff_float& mincoeff = 0,
                                 const int topk = 0, const int max_xweight = -1,
                                 const int xtrunc_period = 1) {
    return propagate_loop<PauliPolynomial>(
        circuit, a,
        [maxdegree](PauliPolynomial& obs, const ROT& gate) { conjugate(obs, gate, maxdegree); },
        maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a,
                                 const int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), maxdegree);
}

inline PauliPolynomial propagate_omp(const Circuit& circuit, const PauliPolynomial& a,
                                     int n_threads = 1, const int& maxdegree = 128,
                                     const ff_float& mincoeff = 0, const int topk = 0,
                                     const int max_xweight = -1, const int xtrunc_period = 1) {
    if (n_threads <= 1)
        return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);

#ifdef FF_OPENMP
    // Needs its own loop: obs_snap persists across gates for capacity reuse
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
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
            conjugate_omp(obs, std::get<ROT>(circuit[i]), maxdegree, n_threads, obs_snap);
            truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
            rot_count++;
        }
    }
    if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    return obs;
#else
    return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#endif
}

inline PauliPolynomial propagate_sorted(const Circuit& circuit, const PauliPolynomial& a,
                                        int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                        int max_xweight = -1, int xtrunc_period = 1) {
    return propagate_loop<SortedPauliPoly>(
        circuit, a,
        [maxdegree](SortedPauliPoly& obs, const ROT& gate) { conjugate(obs, gate, maxdegree); },
        maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
}

inline PauliPolynomial propagate_sorted_radix(const Circuit& circuit, const PauliPolynomial& a,
                                              int maxdegree = 128, ff_float mincoeff = 0,
                                              int topk = 0, int max_xweight = -1,
                                              int xtrunc_period = 1) {
    std::vector<PauliString> keys_buf;
    std::vector<ff_complex> coeffs_buf;
    return propagate_loop<SortedPauliPoly>(
        circuit, a,
        [maxdegree, &keys_buf, &coeffs_buf](SortedPauliPoly& obs, const ROT& gate) {
            conjugate_radix(obs, gate, maxdegree, keys_buf, coeffs_buf);
        },
        maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
}

inline PauliPolynomial propagate_sorted_omp(const Circuit& circuit, const PauliPolynomial& a,
                                            int n_threads = 1, int maxdegree = 128,
                                            ff_float mincoeff = 0, int topk = 0,
                                            int max_xweight = -1, int xtrunc_period = 1) {
    if (n_threads <= 1)
        return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);

#ifdef FF_OPENMP
    return propagate_loop<SortedPauliPoly>(
        circuit, a,
        [maxdegree, n_threads](SortedPauliPoly& obs, const ROT& gate) {
            conjugate_omp(obs, gate, maxdegree, n_threads);
        },
        maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#else
    return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#endif
}

// Batched propagation (own loop: groups ROTs into commuting batches)
inline PauliPolynomial propagate_batched(const Circuit& circuit, const PauliPolynomial& a,
                                         int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                         int max_xweight = -1, int xtrunc_period = 1) {
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
    std::vector<ROT> rot_buffer;

    auto flush_rots = [&]() {
        if (rot_buffer.empty()) return;
        auto batches = batch_commuting_gates(rot_buffer);
        for (const auto& batch : batches) {
            apply_batch(obs, batch, maxdegree);
            rot_count += batch.size();
            truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
        }
        rot_buffer.clear();
    };

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            flush_rots();
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
    flush_rots();
    if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    return obs;
}

}  // namespace pauli_gates

}  // namespace fastfermion
