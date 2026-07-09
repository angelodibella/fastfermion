// Truncation rules for PauliPolynomial.
// All truncation operates on PauliPolynomial (hash-map representation).
// The sorted path converts to PauliPolynomial, truncates here, and converts back.

#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"
#include "pauli/algebra.h"

namespace fastfermion {
namespace pauli_gates {

// =========================================================================
// Per-run truncation statistics (always on; a handful of scalar updates per
// truncation event, negligible next to the event itself). cert_tau is the
// running certificate Sum_e delta_e of the coefficient threshold: delta_e is
// the Hilbert-Schmidt norm of the terms discarded at event e (Pauli strings
// are orthonormal, so delta_e = sqrt of the discarded |c|^2 sum), and the
// sum bounds the drift from the weight-only reference for ANY threshold
// schedule (workbook cor:schedule-certificate). Written only from the driver
// thread; parallel backends reduce their per-shard discards first.
// =========================================================================
struct TruncStats {
    double cert_tau = 0;                       // Sum_e ||discard_e|| over threshold events
    long long n_tau_events = 0, n_w_events = 0;  // scheduled rule firings this run
    std::size_t peak_terms = 0;                // high-water term count seen at events
    void reset() {
        cert_tau = 0;
        n_tau_events = n_w_events = 0;
        peak_terms = 0;
    }
};
inline TruncStats& trunc_stats() {
    static TruncStats s;
    return s;
}

// A period-p rule fires when the gate window [first, last] contains a gate
// index g with (g+1) % p == 0, i.e. when the window crosses a period
// boundary. Windows are one gate wide unbatched and one commuting batch wide
// batched, so with p = 1 this is always true and the cadence reduces to the
// historical per-gate / per-batch behavior.
inline bool period_crossed(int first, int last, int period) {
    if (period <= 1) return true;
    return (last + 1) / period > first / period;
}

// Discard terms with |c_Q| <= mincoeff; return the discarded |c|^2 sum
// (the squared HS norm of the discard, feeding the run certificate).
inline double truncate_threshold(PauliPolynomial& p, ff_float mincoeff) {
    double disc2 = 0;
    std::erase_if(p.terms, [mincoeff, &disc2](const auto& t) {
        const double m = std::abs(t.second);
        if (m <= mincoeff) {
            disc2 += m * m;
            return true;
        }
        return false;
    });
    return disc2;
}

// Discard terms whose full Pauli weight exceeds maxdegree. Only used by the
// deferred schedule (maxdegree_period > 1); at period 1 the emission filter
// in conjugate() enforces the cutoff and this pass never runs.
inline void truncate_weight(PauliPolynomial& p, int maxdegree) {
    std::erase_if(p.terms,
                  [maxdegree](const auto& t) { return t.first.degree_total() > maxdegree; });
}

// Keep only the K terms with largest |c_Q|. Uses partial sort (O(K) average).
inline void truncate_top_k(PauliPolynomial& p, int k) {
    if (k <= 0 || (int)p.terms.size() <= k) return;
    std::vector<ff_float> mags;
    mags.reserve(p.terms.size());
    for (const auto& [_, c] : p.terms) mags.push_back(std::abs(c));
    std::nth_element(mags.begin(), mags.begin() + k, mags.end(), std::greater<ff_float>());
    ff_float cutoff = mags[k];
    std::erase_if(p.terms, [cutoff](const auto& t) { return std::abs(t.second) < cutoff; });
    // Trim exact ties at the boundary to enforce |terms| <= k
    while ((int)p.terms.size() > k) {
        auto it = p.terms.begin();
        while (it != p.terms.end() && std::abs(it->second) > cutoff) ++it;
        if (it != p.terms.end())
            p.terms.erase(it);
        else
            break;
    }
}

// Discard terms whose X/Y-weight exceeds max_xweight (xSPD truncation).
inline void truncate_x_weight(PauliPolynomial& p, int max_xweight) {
    if (max_xweight < 0) return;
    std::erase_if(p.terms, [max_xweight](const auto& t) {
        return t.first.degree_x() + t.first.degree_y() > max_xweight;
    });
}

// Apply all active truncation rules after one gate window (a single gate
// unbatched, a commuting batch batched; [rot_first, rot_last] are the window's
// gate indices). The individual rules are no-ops at their defaults
// (mincoeff=0, topk=0, max_xweight=-1, periods=1). The threshold and the
// deferred weight cutoff fire on their own periods; top-K keeps its
// historical every-window cadence, and x-weight keeps its historical
// xtrunc_period semantics (evaluated on rot_last), unchanged.
inline void truncate_all(PauliPolynomial& obs, ff_float mincoeff, int topk, int max_xweight,
                         int xtrunc_period, int rot_last, int maxdegree = 128,
                         int maxdegree_period = 1, int mincoeff_period = 1, int rot_first = -1) {
    if (rot_first < 0) rot_first = rot_last;
    auto& stats = trunc_stats();
    stats.peak_terms = std::max(stats.peak_terms, obs.terms.size());
    // Weight event before the threshold event: delta_e then counts only terms
    // the weight rule keeps, matching the certificate's weight-only reference.
    if (maxdegree_period > 1 && period_crossed(rot_first, rot_last, maxdegree_period)) {
        truncate_weight(obs, maxdegree);
        stats.n_w_events++;
    }
    if (mincoeff > 0 && period_crossed(rot_first, rot_last, mincoeff_period)) {
        stats.cert_tau += std::sqrt(truncate_threshold(obs, mincoeff));
        stats.n_tau_events++;
    }
    if (topk > 0) truncate_top_k(obs, topk);
    if (max_xweight >= 0 && xtrunc_period > 0 && (rot_last + 1) % xtrunc_period == 0)
        truncate_x_weight(obs, max_xweight);
}

}  // namespace pauli_gates
}  // namespace fastfermion
