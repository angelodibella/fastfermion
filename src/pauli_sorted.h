// Sorted-array Pauli propagation.
//
// Replaces the hash-map merge with a two-pointer sweep over sorted arrays,
// giving O(K) cache-friendly merging instead of O(K) random-access hash
// insertions.

#pragma once
#include <algorithm>
#include <vector>

#include "common.h"
#include "pauli_algebra.h"
#include "pauli_propagation.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {

// Sorted-array representation of a Pauli polynomial.
struct SortedPauliPoly {
    std::vector<PauliString> keys;
    std::vector<ff_complex> coeffs;

    SortedPauliPoly() = default;

    static SortedPauliPoly from_poly(const PauliPolynomial& p) {
        std::vector<std::pair<PauliString, ff_complex>> entries(p.terms.begin(), p.terms.end());
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        SortedPauliPoly result;
        result.keys.reserve(entries.size());
        result.coeffs.reserve(entries.size());
        for (const auto& [k, c] : entries) {
            result.keys.push_back(k);
            result.coeffs.push_back(c);
        }
        return result;
    }

    PauliPolynomial to_poly() const { return PauliPolynomial(keys, coeffs); }

    size_t size() const { return keys.size(); }

    // Merge two sorted polynomials, summing coefficients on matching keys.
    static SortedPauliPoly merge(const SortedPauliPoly& a, const SortedPauliPoly& b) {
        SortedPauliPoly result;
        result.keys.reserve(a.size() + b.size());
        result.coeffs.reserve(a.size() + b.size());

        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a.keys[i] < b.keys[j]) {
                result.keys.push_back(a.keys[i]);
                result.coeffs.push_back(a.coeffs[i++]);
            } else if (b.keys[j] < a.keys[i]) {
                result.keys.push_back(b.keys[j]);
                result.coeffs.push_back(b.coeffs[j++]);
            } else {
                result.keys.push_back(a.keys[i]);
                result.coeffs.push_back(a.coeffs[i++] + b.coeffs[j++]);
            }
        }
        for (; i < a.size(); i++) {
            result.keys.push_back(a.keys[i]);
            result.coeffs.push_back(a.coeffs[i]);
        }
        for (; j < b.size(); j++) {
            result.keys.push_back(b.keys[j]);
            result.coeffs.push_back(b.coeffs[j]);
        }
        return result;
    }
};

// Apply one Pauli rotation to a sorted polynomial:
//   Q -> Q                              if [P, Q] = 0
//   Q -> cos(t) Q + i sin(t) (P*Q)     if {P, Q} = 0
SortedPauliPoly sorted_conjugate(const SortedPauliPoly& in, const PauliString& ps, ff_float theta,
                                 int maxdegree) {
    const ff_float cos_t = cos(theta);
    const ff_complex isin_t = ff_complex(0, sin(theta));

    // Emit kept (sorted) and partners (unsorted)
    SortedPauliPoly kept;
    std::vector<std::pair<PauliString, ff_complex>> partner_pairs;
    kept.keys.reserve(in.size());
    kept.coeffs.reserve(in.size());
    partner_pairs.reserve(in.size());

    for (size_t i = 0; i < in.size(); i++) {
        const PauliString& q = in.keys[i];
        const ff_complex& c = in.coeffs[i];

        if (q.commutes(ps)) {
            kept.keys.push_back(q);
            kept.coeffs.push_back(c);
        } else {
            kept.keys.push_back(q);
            kept.coeffs.push_back(c * cos_t);

            PauliMonomial pq = ps * q;
            if (pq.degree_total() <= maxdegree)
                partner_pairs.emplace_back(pq.pauli_string(), c * isin_t * pq.coefficient());
        }
    }

    // Sort partners by key
    std::sort(partner_pairs.begin(), partner_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    SortedPauliPoly partners;
    partners.keys.reserve(partner_pairs.size());
    partners.coeffs.reserve(partner_pairs.size());
    for (const auto& [k, c] : partner_pairs) {
        partners.keys.push_back(k);
        partners.coeffs.push_back(c);
    }

    // Step 3: merge the two sorted streams
    return SortedPauliPoly::merge(kept, partners);
}

#ifdef FF_OPENMP

// Parallel merge of two sorted SortedPauliPolys.
//
// Splits both arrays into n_chunks aligned segments via binary search,
// merges each segment independently on a separate thread.
SortedPauliPoly sorted_merge_omp(const SortedPauliPoly& a, const SortedPauliPoly& b,
                                 int n_threads) {
    if (a.size() == 0) return b;
    if (b.size() == 0) return a;

    // Choose split points from the larger array, find matching positions in the smaller
    const SortedPauliPoly& big = (a.size() >= b.size()) ? a : b;
    const SortedPauliPoly& small = (a.size() >= b.size()) ? b : a;

    // Divide the big array into n_threads chunks
    std::vector<size_t> big_splits(n_threads + 1);
    std::vector<size_t> small_splits(n_threads + 1);
    big_splits[0] = 0;
    big_splits[n_threads] = big.size();
    small_splits[0] = 0;
    small_splits[n_threads] = small.size();

    for (int t = 1; t < n_threads; t++) {
        big_splits[t] = (big.size() * t) / n_threads;
        // Find where big's split key falls in small's array
        const PauliString& split_key = big.keys[big_splits[t]];
        small_splits[t] =
            std::lower_bound(small.keys.begin(), small.keys.end(), split_key) - small.keys.begin();
    }

    // Each thread merges its chunk into a local result
    std::vector<SortedPauliPoly> chunks(n_threads);

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int t = 0; t < n_threads; t++) {
        size_t bi = big_splits[t], b_end = big_splits[t + 1];
        size_t si = small_splits[t], s_end = small_splits[t + 1];

        auto& chunk = chunks[t];
        chunk.keys.reserve((b_end - bi) + (s_end - si));
        chunk.coeffs.reserve((b_end - bi) + (s_end - si));

        while (bi < b_end && si < s_end) {
            if (big.keys[bi] < small.keys[si]) {
                chunk.keys.push_back(big.keys[bi]);
                chunk.coeffs.push_back(big.coeffs[bi++]);
            } else if (small.keys[si] < big.keys[bi]) {
                chunk.keys.push_back(small.keys[si]);
                chunk.coeffs.push_back(small.coeffs[si++]);
            } else {
                chunk.keys.push_back(big.keys[bi]);
                chunk.coeffs.push_back(big.coeffs[bi++] + small.coeffs[si++]);
            }
        }
        for (; bi < b_end; bi++) {
            chunk.keys.push_back(big.keys[bi]);
            chunk.coeffs.push_back(big.coeffs[bi]);
        }
        for (; si < s_end; si++) {
            chunk.keys.push_back(small.keys[si]);
            chunk.coeffs.push_back(small.coeffs[si]);
        }
    }

    // Concatenate chunks (already in order)
    size_t total = 0;
    for (const auto& chunk : chunks) total += chunk.size();

    SortedPauliPoly result;
    result.keys.reserve(total);
    result.coeffs.reserve(total);
    for (const auto& chunk : chunks) {
        result.keys.insert(result.keys.end(), chunk.keys.begin(), chunk.keys.end());
        result.coeffs.insert(result.coeffs.end(), chunk.coeffs.begin(), chunk.coeffs.end());
    }
    return result;
}

// Parallel emit, sort, and merge.
SortedPauliPoly sorted_conjugate_omp(const SortedPauliPoly& in, const PauliString& ps,
                                     ff_float theta, int maxdegree, int n_threads) {
    const ff_float cos_t = cos(theta);
    const ff_complex isin_t = ff_complex(0, sin(theta));
    const int K = (int)in.size();

    // Parallel emit into thread-local buffers
    std::vector<SortedPauliPoly> kept_parts(n_threads);
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> partner_parts(n_threads);

    int chunk = (K + n_threads - 1) / n_threads;

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& my_kept = kept_parts[tid];
        auto& my_partners = partner_parts[tid];
        my_kept.keys.reserve(chunk);
        my_kept.coeffs.reserve(chunk);
        my_partners.reserve(chunk);

#pragma omp for schedule(static)
        for (int j = 0; j < K; j++) {
            const PauliString& q = in.keys[j];
            const ff_complex& c = in.coeffs[j];

            if (q.commutes(ps)) {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c);
            } else {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c * cos_t);

                PauliMonomial pq = ps * q;
                if (pq.degree_total() <= maxdegree)
                    my_partners.emplace_back(pq.pauli_string(), c * isin_t * pq.coefficient());
            }
        }
    }

    // Concatenate kept parts (each is sorted since input was sorted and
    // static scheduling preserves order within each chunk)
    SortedPauliPoly kept;
    {
        size_t total_kept = 0;
        for (const auto& part : kept_parts) total_kept += part.size();
        kept.keys.reserve(total_kept);
        kept.coeffs.reserve(total_kept);
        for (const auto& part : kept_parts) {
            kept.keys.insert(kept.keys.end(), part.keys.begin(), part.keys.end());
            kept.coeffs.insert(kept.coeffs.end(), part.coeffs.begin(), part.coeffs.end());
        }
    }

    // Concatenate partner pairs, then sort
    std::vector<std::pair<PauliString, ff_complex>> all_partners;
    {
        size_t total_partners = 0;
        for (const auto& part : partner_parts) total_partners += part.size();
        all_partners.reserve(total_partners);
        for (auto& part : partner_parts)
            all_partners.insert(all_partners.end(), part.begin(), part.end());
    }

    // Sort partners
    std::sort(all_partners.begin(), all_partners.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    SortedPauliPoly partners;
    partners.keys.reserve(all_partners.size());
    partners.coeffs.reserve(all_partners.size());
    for (const auto& [k, c] : all_partners) {
        partners.keys.push_back(k);
        partners.coeffs.push_back(c);
    }

    return sorted_merge_omp(kept, partners, n_threads);
}

#endif  // FF_OPENMP

// Remove entries with |coeff| <= threshold, preserving sorted order.
SortedPauliPoly sorted_truncate(const SortedPauliPoly& in, ff_float mincoeff) {
    SortedPauliPoly result;
    result.keys.reserve(in.size());
    result.coeffs.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        if (std::abs(in.coeffs[i]) > mincoeff) {
            result.keys.push_back(in.keys[i]);
            result.coeffs.push_back(in.coeffs[i]);
        }
    }
    return result;
}

namespace pauli_gates {

// Serial sorted-array propagation.
PauliPolynomial propagate_sorted(const Circuit& circuit, const PauliPolynomial& a,
                                 int maxdegree = 128, ff_float mincoeff = 0) {
    SortedPauliPoly obs = SortedPauliPoly::from_poly(a);

    int clifford_begin;
    bool pending_clifford = false;
    PauliPolynomial obs_hm;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                obs_hm = obs.to_poly();
                _apply_clifford_circuit(obs_hm, circuit, i + 1, clifford_begin + 1);
                obs = SortedPauliPoly::from_poly(obs_hm);
                pending_clifford = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);
            obs = sorted_conjugate(obs, gate.ps, gate.theta, maxdegree);

            if (mincoeff > 0) obs = sorted_truncate(obs, mincoeff);
        }
    }
    if (pending_clifford) {
        obs_hm = obs.to_poly();
        _apply_clifford_circuit(obs_hm, circuit, 0, clifford_begin + 1);
        return obs_hm;
    }
    return obs.to_poly();
}

// OpenMP-parallel sorted-array propagation.
//
// Falls back to serial sorted path when n_threads <= 1 or FF_OPENMP is not defined.
PauliPolynomial propagate_sorted_omp(const Circuit& circuit, const PauliPolynomial& a,
                                     int n_threads = 1, int maxdegree = 128,
                                     ff_float mincoeff = 0) {
    if (n_threads <= 1) return propagate_sorted(circuit, a, maxdegree, mincoeff);

#ifdef FF_OPENMP
    SortedPauliPoly obs = SortedPauliPoly::from_poly(a);

    int clifford_begin;
    bool pending_clifford = false;
    PauliPolynomial obs_hm;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                obs_hm = obs.to_poly();
                _apply_clifford_circuit(obs_hm, circuit, i + 1, clifford_begin + 1);
                obs = SortedPauliPoly::from_poly(obs_hm);
                pending_clifford = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);
            obs = sorted_conjugate_omp(obs, gate.ps, gate.theta, maxdegree, n_threads);

            if (mincoeff > 0) obs = sorted_truncate(obs, mincoeff);
        }
    }
    if (pending_clifford) {
        obs_hm = obs.to_poly();
        _apply_clifford_circuit(obs_hm, circuit, 0, clifford_begin + 1);
        return obs_hm;
    }
    return obs.to_poly();
#else
    return propagate_sorted(circuit, a, maxdegree, mincoeff);
#endif
}

}  // namespace pauli_gates
}  // namespace fastfermion
