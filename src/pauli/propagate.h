// Pauli propagation entry points.
//
// Four propagation functions, all executing Algorithm 2 of the workbook:
//   propagate           serial, hash-map merge
//   propagate_omp       parallel emission (OpenMP), serial hash-map rebuild
//   propagate_sorted    serial, sort-and-merge on a flat sorted vector
//   propagate_batched   serial, groups commuting gates into batches
//
// Each iterates the circuit in reverse, batching consecutive Clifford gates
// and applying ROT gates one at a time (or in batches for propagate_batched).
// Truncation is applied after each ROT gate via truncate_all (truncate.h).

#pragma once
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

// =========================================================================
// Per-gate conjugation helpers (not public API — called by propagate_*)
// =========================================================================

// Apply one ROT gate to a PauliPolynomial via hash-map insertion.
// For each anticommuting term Q, scales its coefficient by cos(theta) in place
// and inserts the partner P*Q with coefficient i*sin(theta) into the hash map.
// New terms are collected in a temporary vector to avoid mutating the map
// while iterating.
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

// Apply one ROT gate with OpenMP-parallel emission.
// Snapshots the hash map into `snap` (a flat vector, reused across gates to
// avoid reallocation), distributes terms across threads for independent
// emission, then serially rebuilds the hash map from the thread-local buffers.
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

// Partition consecutive ROT gates into maximal batches of mutually commuting
// gates. Gates within a batch can be applied in any order.
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

// Apply all gates in a commuting batch to a PauliPolynomial in one pass.
// Each gate's partners are collected and inserted after the gate loop,
// same as conjugate() but iterated over the batch.
inline void apply_batch(PauliPolynomial& obs, const std::vector<ROT>& batch, int maxdegree) {
    for (const auto& gate : batch) {
        const PauliString& ps = gate.ps;
        const ff_float cos_t = cos(gate.theta);
        const ff_complex isin_t = ff_complex(0, sin(gate.theta));

        std::vector<std::pair<PauliString, ff_complex>> partners;
        partners.reserve(obs.terms.size());

        for (auto& [x, coeff] : obs.terms) {
            if (!x.commutes(ps)) {
                PauliMonomial pq = ps * x;
                if (pq.degree_total() <= maxdegree)
                    partners.emplace_back(pq.pauli_string(), coeff * isin_t * pq.coefficient());
                coeff *= cos_t;
            }
        }
        for (const auto& [k, c] : partners) obs.terms[k] += c;
    }
}

// =========================================================================
// Public entry points
// =========================================================================

// Serial hash-map propagation (baseline).
inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                                 const int& maxdegree = 128, const ff_float& mincoeff = 0,
                                 const int topk = 0, const int max_xweight = -1,
                                 const int xtrunc_period = 1) {
    PauliPolynomial obs(a);
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
                _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                pending_clifford = false;
            }
            conjugate(obs, std::get<ROT>(circuit[i]), maxdegree);
            truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
            rot_count++;
        }
    }
    if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    return obs;
}

// Convenience overload: propagate a single PauliString (wraps it in a PauliPolynomial).
inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a,
                                 const int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), maxdegree);
}

// OpenMP hash-map propagation. Falls back to serial when n_threads <= 1.
// The snapshot vector `snap` persists across gates to reuse allocated capacity.
inline PauliPolynomial propagate_omp(const Circuit& circuit, const PauliPolynomial& a,
                                     int n_threads = 1, const int& maxdegree = 128,
                                     const ff_float& mincoeff = 0, const int topk = 0,
                                     const int max_xweight = -1, const int xtrunc_period = 1) {
    if (n_threads <= 1)
        return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);

#ifdef FF_OPENMP
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
    std::vector<std::pair<PauliString, ff_complex>> snap(obs.terms.begin(), obs.terms.end());

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
    if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
    return obs;
#else
    return propagate(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#endif
}

// Sorted-vector propagation. Converts the hash map to a sorted vector at the
// start, applies gates via sort-and-merge (conjugate_sorted in sorted.h),
// truncates by converting back to hash map each gate, and returns a hash map.
// The `sort_method` parameter selects comparison sort or radix sort for the
// partner stream.
inline PauliPolynomial propagate_sorted(const Circuit& circuit, const PauliPolynomial& a,
                                        int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                        int max_xweight = -1, int xtrunc_period = 1,
                                        SortMethod sort_method = SortMethod::comparison) {
    SortedTerms obs = to_sorted(a);
    SortedTerms sort_buf;
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
                PauliPolynomial hm = to_poly(obs);
                _apply_clifford_circuit(hm, circuit, i + 1, clifford_begin + 1);
                obs = to_sorted(hm);
                pending_clifford = false;
            }
            conjugate_sorted(obs, std::get<ROT>(circuit[i]), maxdegree, sort_method, sort_buf);

            // Truncate via hash map (avoids duplicating truncation logic for sorted vectors)
            PauliPolynomial hm = to_poly(obs);
            truncate_all(hm, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
            obs = to_sorted(hm);
            rot_count++;
        }
    }
    if (pending_clifford) {
        PauliPolynomial hm = to_poly(obs);
        _apply_clifford_circuit(hm, circuit, 0, clifford_begin + 1);
        return hm;
    }
    return to_poly(obs);
}

// Wrappers with fixed sort method (stable Python API).
inline PauliPolynomial propagate_sorted_radix(const Circuit& circuit, const PauliPolynomial& a,
                                              int maxdegree = 128, ff_float mincoeff = 0,
                                              int topk = 0, int max_xweight = -1,
                                              int xtrunc_period = 1) {
    return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period,
                            SortMethod::radix);
}

// Kept for Python API compatibility; no OMP sorted path, falls back to serial.
inline PauliPolynomial propagate_sorted_omp(const Circuit& circuit, const PauliPolynomial& a,
                                            int n_threads = 1, int maxdegree = 128,
                                            ff_float mincoeff = 0, int topk = 0,
                                            int max_xweight = -1, int xtrunc_period = 1) {
    return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
}

// Gate-batched propagation. Buffers consecutive ROT gates, partitions them
// into maximal commuting batches (batch_commuting_gates), and applies each
// batch in a single pass (apply_batch). Truncation is per batch, not per gate.
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
