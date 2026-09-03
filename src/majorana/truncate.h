/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "majorana/algebra.h"
#include "truncate.h"

namespace fastfermion {

namespace majorana_gates {

// Truncation rules for Majorana polynomials: the shared rules plus a cutoff on the number of
// unpaired modes of a monomial (modes contributing exactly one of their two Majorana operators),
// with period unpaired_period. A monomial with an unpaired mode has zero expectation in every Fock
// state; under the Jordan-Wigner transform its number of unpaired modes is the x-weight of its
// image Pauli string.
struct MajoranaTruncation : Truncation {
    int max_unpaired = -1;  // -1 = off
    int unpaired_period = 1;

    // Whether a term created by a gate is kept: the rules on the string with period 1
    bool admits(const MajoranaString& s) const {
        return s.degree() <= emission_degree() &&
               (max_unpaired < 0 || unpaired_period > 1 || s.unpaired() <= max_unpaired);
    }
    bool unpaired_due(int first, int last) const {
        return max_unpaired >= 0 && unpaired_period > 1 && period_crossed(first, last, unpaired_period);
    }
    // The rules on the string due after the window [first, last], applied term by term
    void apply_key_rules(MajoranaPolynomialMap& terms, int first, int last) const {
        if (degree_due(first, last)) {
            truncate_keys(terms, [this](const MajoranaString& s) { return s.degree() > maxdegree; });
        }
        if (unpaired_due(first, last)) {
            truncate_keys(terms, [this](const MajoranaString& s) { return s.unpaired() > max_unpaired; });
        }
    }
};

}  // namespace majorana_gates

}  // namespace fastfermion
