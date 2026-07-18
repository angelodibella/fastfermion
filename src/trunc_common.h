// Basis-independent truncation infrastructure shared by the Pauli and
// Majorana propagations: the per-run statistics singleton (one object per
// process -- the certificate a run reports must accumulate in one place
// whichever basis produced it), the period-cadence test, and the two
// coefficient-reading rules (threshold, top-K), which never inspect the
// basis element and are therefore templates over the polynomial type.
// Structural rules (Pauli weight, x-weight, Majorana degree, unpaired
// weight) read the key and live with their basis in pauli/truncate.h and
// majorana/truncate.h.

#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"

namespace fastfermion {

// =========================================================================
// Per-run truncation statistics (always on; a handful of scalar updates per
// truncation event, negligible next to the event itself). cert_tau is the
// running certificate Sum_e delta_e of the coefficient threshold: delta_e is
// the norm of the terms discarded at event e (basis elements are
// orthonormal, so delta_e = sqrt of the discarded |c|^2 sum), and the sum
// bounds the drift from the structural-rule-only reference for ANY threshold
// schedule (the truncated gate map is an isometry followed by an orthogonal
// projection, so per-event discards accumulate additively). Written only from the driver thread; parallel backends
// reduce their per-shard discards first.
// =========================================================================
struct TruncStats {
    double cert_tau = 0;                         // Sum_e ||discard_e|| over threshold events
    long long n_tau_events = 0, n_w_events = 0;  // scheduled rule firings this run
    std::size_t peak_terms = 0;                  // high-water term count seen at events
    std::size_t peak_device_bytes = 0;  // GPU runs: allocator high-water (0 on CPU paths)
    void reset() {
        cert_tau = 0;
        n_tau_events = n_w_events = 0;
        peak_terms = 0;
        peak_device_bytes = 0;
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

// Discard terms with |c| <= mincoeff; return the discarded |c|^2 sum
// (the squared norm of the discard, feeding the run certificate).
template <class PolyT>
inline double truncate_threshold(PolyT& p, ff_float mincoeff) {
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

// Keep only the K terms with largest |c|. Uses partial sort (O(K) average).
template <class PolyT>
inline void truncate_top_k(PolyT& p, int k) {
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

}  // namespace fastfermion
