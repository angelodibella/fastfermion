/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "common.h"

namespace fastfermion {

// Statistics of the last call to propagate() in the calling thread, in any basis. cert_tau bounds
// the error introduced by the coefficient threshold alone: the terms discarded by each threshold
// event have norm delta_e (basis elements are orthonormal) and gates act isometrically, so the
// propagated polynomial is within sum_e delta_e of the one propagated with the same other rules
// but no threshold.
struct TruncStats {
    double cert_tau = 0;         // sum over threshold events of the norm of the discarded terms
    long long n_tau_events = 0;  // number of threshold events
    long long n_w_events = 0;    // number of scheduled degree events (maxdegree_period > 1)
    std::size_t peak_terms = 0;  // largest number of terms held before a truncation event
};

inline TruncStats& trunc_stats() {
    thread_local TruncStats stats;
    return stats;
}

// A rule with period p fires after the window of gates [first, last] (gates numbered in the
// order they are applied) if the window contains a gate g with (g + 1) % p == 0. A window is
// a single gate, or a batch of mutually commuting gates when batching is on.
inline bool period_crossed(int first, int last, int period) {
    return period <= 1 || (last + 1) / period > first / period;
}

// Removes the terms with |coefficient| <= mincoeff; returns the sum of their squared magnitudes
template <class Map>
double truncate_threshold(Map& terms, ff_float mincoeff) {
    double discarded = 0;
    std::erase_if(terms, [&](const auto& t) {
        const double m = std::abs(t.second);
        if (m > mincoeff) return false;
        discarded += m * m;
        return true;
    });
    return discarded;
}

// Keeps the k terms of largest magnitude; ties at the k-th magnitude are broken in favor of the
// smallest basis elements, so that the result does not depend on the order of the terms in the map
template <class Map>
void truncate_top_k(Map& terms, int k) {
    if (k <= 0 || terms.size() <= std::size_t(k)) return;
    std::vector<ff_float> mags;
    mags.reserve(terms.size());
    for (const auto& [x, c] : terms) mags.push_back(std::abs(c));
    std::nth_element(mags.begin(), mags.begin() + (k - 1), mags.end(), std::greater<ff_float>());
    const ff_float kth = mags[k - 1];
    std::erase_if(terms, [kth](const auto& t) { return std::abs(t.second) < kth; });
    if (terms.size() > std::size_t(k)) {
        std::vector<typename Map::key_type> ties;
        for (const auto& [x, c] : terms) {
            if (std::abs(c) == kth) ties.push_back(x);
        }
        std::sort(ties.begin(), ties.end());
        for (std::size_t i = ties.size() - (terms.size() - k); i < ties.size(); i++) terms.erase(ties[i]);
    }
}

// Removes the terms whose basis element satisfies pred
template <class Map, class Pred>
void truncate_keys(Map& terms, Pred pred) {
    std::erase_if(terms, [&](const auto& t) { return pred(t.first); });
}

// The truncation rules of a propagation and their schedule, shared by the Pauli and Majorana
// bases, which add their own rules on the basis element (see admits and apply_key_rules there).
// A rule on the basis element with period 1 is enforced when the new terms of a gate are created,
// so that no term violating it is ever stored; with a period p > 1 it fires after every p-th
// gate instead, and terms violating it exist in between and may be rotated back. The coefficient
// threshold fires after every mincoeff_period-th gate, top-k after every window.
struct Truncation {
    int maxdegree = INT_MAX;
    int maxdegree_period = 1;
    ff_float mincoeff = 0;
    int mincoeff_period = 1;
    int topk = 0;  // 0 = off

    // Degree above which the new terms of a gate are discarded
    int emission_degree() const { return maxdegree_period == 1 ? maxdegree : INT_MAX; }
    bool degree_due(int first, int last) const {
        return maxdegree_period > 1 && period_crossed(first, last, maxdegree_period);
    }
    bool threshold_due(int first, int last) const {
        return mincoeff > 0 && period_crossed(first, last, mincoeff_period);
    }
    // Records the events of a window: n_terms is the term count before its truncation and
    // discarded the squared norm dropped by the threshold
    void record(int first, int last, std::size_t n_terms, double discarded) const {
        TruncStats& stats = trunc_stats();
        stats.peak_terms = std::max(stats.peak_terms, n_terms);
        if (degree_due(first, last)) stats.n_w_events++;
        if (threshold_due(first, last)) {
            stats.cert_tau += std::sqrt(discarded);
            stats.n_tau_events++;
        }
    }
};

// Applies the rules on the basis element and the threshold due after the window [first, last] to
// a term map, which may be a part of the polynomial; returns the squared norm the threshold
// discarded. rules is a Truncation extended with the basis-specific apply_key_rules.
template <class Map, class Rules>
double truncate_terms(Map& terms, const Rules& rules, int first, int last) {
    rules.apply_key_rules(terms, first, last);
    return rules.threshold_due(first, last) ? truncate_threshold(terms, rules.mincoeff) : 0;
}

// Applies all the rules due after the window [first, last] to the whole polynomial and records
// the event
template <class Map, class Rules>
void truncate_window(Map& terms, const Rules& rules, int first, int last) {
    const std::size_t n_terms = terms.size();
    const double discarded = truncate_terms(terms, rules, first, last);
    if (rules.topk > 0) truncate_top_k(terms, rules.topk);
    rules.record(first, last, n_terms, discarded);
}

}  // namespace fastfermion
