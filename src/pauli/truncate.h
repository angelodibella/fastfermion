// Truncation rules for PauliPolynomial.
// All truncation operates on PauliPolynomial (hash-map representation).
// The sorted path converts to PauliPolynomial, truncates here, and converts back.

#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"
#include "pauli/algebra.h"
#include "trunc_common.h"

namespace fastfermion {
namespace pauli_gates {

// The stats singleton, cadence test, and the coefficient-reading rules
// (threshold, top-K) are basis-independent and shared with the Majorana
// propagation; see trunc_common.h. Re-exported here so existing callers
// keep their pauli_gates:: qualification.
using fastfermion::TruncStats;
using fastfermion::trunc_stats;
using fastfermion::period_crossed;
using fastfermion::truncate_threshold;
using fastfermion::truncate_top_k;

// Discard terms whose full Pauli weight exceeds maxdegree. Only used by the
// deferred schedule (maxdegree_period > 1); at period 1 the emission filter
// in conjugate() enforces the cutoff and this pass never runs.
inline void truncate_weight(PauliPolynomial& p, int maxdegree) {
    std::erase_if(p.terms,
                  [maxdegree](const auto& t) { return t.first.degree_total() > maxdegree; });
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
