/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "pauli/algebra.h"
#include "truncate.h"

namespace fastfermion {

namespace pauli_gates {

// Truncation rules for Pauli polynomials: the shared rules plus a cutoff on the number of X and
// Y factors of a string (its x-weight), with period xweight_period
struct PauliTruncation : Truncation {
    int max_xweight = -1;  // -1 = off
    int xweight_period = 1;

    // Whether a term created by a gate is kept: the rules on the string with period 1
    bool admits(const PauliString& s) const {
        return s.degree_total() <= emission_degree() &&
               (max_xweight < 0 || xweight_period > 1 || s.degree_x() + s.degree_y() <= max_xweight);
    }
    bool xweight_due(int first, int last) const {
        return max_xweight >= 0 && xweight_period > 1 && period_crossed(first, last, xweight_period);
    }
    // The rules on the string due after the window [first, last], applied term by term
    void apply_key_rules(PauliPolynomialMap& terms, int first, int last) const {
        if (degree_due(first, last)) {
            truncate_keys(terms, [this](const PauliString& s) { return s.degree_total() > maxdegree; });
        }
        if (xweight_due(first, last)) {
            truncate_keys(terms, [this](const PauliString& s) { return s.degree_x() + s.degree_y() > max_xweight; });
        }
    }
};

}  // namespace pauli_gates

}  // namespace fastfermion
