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
// -- the structural cutoff, whose retained set is fixed a priori. Only
// used by the deferred schedule (maxdegree_period > 1); at period 1 the
// emission filter in MROT::apply_inplace enforces the cutoff and this pass
// never runs.
inline void truncate_degree(MajoranaPolynomial& p, int maxdegree) {
    std::erase_if(p.terms, [maxdegree](const auto& t) { return t.first.degree() > maxdegree; });
}

// Discard terms with more than max_unpaired unpaired modes -- the fermionic
// analogue of the Pauli x-weight rule: it grades distance from the Fock
// readout sector (a monomial with an unpaired mode has zero expectation in
// every Fock state). Key-only, hence schedule-free at a fixed cadence by
// compaction invariance; unlike the degree cutoff its retained set carries
// no polynomial a-priori bound (the fully paired sector alone holds 2^M
// monomials), so pair it with a threshold or top-K budget in practice.
inline void truncate_unpaired(MajoranaPolynomial& p, int max_unpaired) {
    if (max_unpaired < 0) return;
    std::erase_if(p.terms,
                  [max_unpaired](const auto& t) { return t.first.unpaired() > max_unpaired; });
}

// Apply all active truncation rules after one gate window (a single gate
// unbatched, a commuting batch batched; [rot_first, rot_last] are the
// window's gate indices). Mirrors pauli_gates::truncate_all rule for rule so
// the two propagations share their schedule semantics and certificate; the
// unpaired-mode rule is the counterpart of the Pauli x-weight rule (under
// Jordan-Wigner the unpaired count equals the image string's x-weight).
inline void truncate_all(MajoranaPolynomial& obs, ff_float mincoeff, int topk, int rot_last,
                         int maxdegree = INT_MAX, int maxdegree_period = 1,
                         int mincoeff_period = 1, int rot_first = -1, int max_unpaired = -1,
                         int unpaired_period = 1) {
    if (rot_first < 0) rot_first = rot_last;
    auto& stats = trunc_stats();
    stats.peak_terms = std::max(stats.peak_terms, obs.terms.size());
    // Degree event before the threshold event: delta_e then counts only terms
    // the degree rule keeps, matching the certificate's structural reference.
    if (maxdegree_period > 1 && period_crossed(rot_first, rot_last, maxdegree_period)) {
        truncate_degree(obs, maxdegree);
        stats.n_w_events++;
    }
    // Key-only like the degree rule, so it precedes the threshold event: the
    // certificate's delta_e counts only terms the structural rules keep.
    if (max_unpaired >= 0 && period_crossed(rot_first, rot_last, unpaired_period))
        truncate_unpaired(obs, max_unpaired);
    if (mincoeff > 0 && period_crossed(rot_first, rot_last, mincoeff_period)) {
        stats.cert_tau += std::sqrt(truncate_threshold(obs, mincoeff));
        stats.n_tau_events++;
    }
    if (topk > 0) truncate_top_k(obs, topk);
}

}  // namespace majorana_gates
}  // namespace fastfermion
