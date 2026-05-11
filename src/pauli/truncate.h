// Truncation rules for PauliPolynomial.
// All truncation operates on PauliPolynomial (hash-map representation).
// The sorted path converts to PauliPolynomial, truncates here, and converts back.

#pragma once
#include <algorithm>
#include <vector>

#include "common.h"
#include "pauli/algebra.h"

namespace fastfermion {
namespace pauli_gates {

// Discard terms with |c_Q| <= mincoeff.
inline void truncate_threshold(PauliPolynomial& p, ff_float mincoeff) {
    std::erase_if(p.terms, [mincoeff](const auto& t) { return std::abs(t.second) <= mincoeff; });
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

// Apply all active truncation rules after one gate (or batch).
// Called by every propagation loop; the individual rules are no-ops
// when their parameter is at the default (mincoeff=0, topk=0, max_xweight=-1).
inline void truncate_all(PauliPolynomial& obs, ff_float mincoeff, int topk, int max_xweight,
                         int xtrunc_period, int rot_count) {
    if (mincoeff > 0) truncate_threshold(obs, mincoeff);
    if (topk > 0) truncate_top_k(obs, topk);
    if (max_xweight >= 0 && xtrunc_period > 0 && (rot_count + 1) % xtrunc_period == 0)
        truncate_x_weight(obs, max_xweight);
}

}  // namespace pauli_gates
}  // namespace fastfermion
