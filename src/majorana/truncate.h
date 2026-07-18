// Truncation rules for MajoranaPolynomial (hash-map representation).
// The coefficient-reading rules (threshold, top-K), the cadence test, and
// the per-run stats singleton are shared with the Pauli propagation in
// trunc_common.h; only the structural, key-reading rules live here.

#pragma once
#include <algorithm>
#include <cmath>

#include "common.h"
#include "majorana/algebra.h"
#include "trunc_common.h"

namespace fastfermion {
namespace majorana_gates {

using fastfermion::TruncStats;
using fastfermion::trunc_stats;
using fastfermion::period_crossed;
using fastfermion::truncate_threshold;
using fastfermion::truncate_top_k;

// Discard terms whose Majorana degree (number of factors) exceeds maxdegree
// -- the structural cutoff trunc_d (workbook sec:majorana-transition). Only
// used by the deferred schedule (maxdegree_period > 1); at period 1 the
// emission filter in MROT::apply_inplace enforces the cutoff and this pass
// never runs.
inline void truncate_degree(MajoranaPolynomial& p, int maxdegree) {
    std::erase_if(p.terms, [maxdegree](const auto& t) { return t.first.degree() > maxdegree; });
}

// Apply all active truncation rules after one gate window (a single gate
// unbatched, a commuting batch batched; [rot_first, rot_last] are the
// window's gate indices). Mirrors pauli_gates::truncate_all rule for rule so
// the two propagations share their schedule semantics and certificate;
// the x-weight rule has no counterpart here (its fermionic analogue, the
// unpaired-weight cutoff, is a planned second filter, not a v1 rule).
inline void truncate_all(MajoranaPolynomial& obs, ff_float mincoeff, int topk, int rot_last,
                         int maxdegree = INT_MAX, int maxdegree_period = 1,
                         int mincoeff_period = 1, int rot_first = -1) {
    if (rot_first < 0) rot_first = rot_last;
    auto& stats = trunc_stats();
    stats.peak_terms = std::max(stats.peak_terms, obs.terms.size());
    // Degree event before the threshold event: delta_e then counts only terms
    // the degree rule keeps, matching the certificate's structural reference.
    if (maxdegree_period > 1 && period_crossed(rot_first, rot_last, maxdegree_period)) {
        truncate_degree(obs, maxdegree);
        stats.n_w_events++;
    }
    if (mincoeff > 0 && period_crossed(rot_first, rot_last, mincoeff_period)) {
        stats.cert_tau += std::sqrt(truncate_threshold(obs, mincoeff));
        stats.n_tau_events++;
    }
    if (topk > 0) truncate_top_k(obs, topk);
}

}  // namespace majorana_gates
}  // namespace fastfermion
