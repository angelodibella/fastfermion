/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// Heisenberg-picture backpropagation of Pauli observables through quantum
// circuits (Sparse Pauli Dynamics). Observables are propagated in reverse
// through Clifford gates and Pauli rotations, accumulating terms in a
// PauliPolynomial.

#pragma once

#include <functional>
#include <variant>

#include "pauli_gates.h"

namespace fastfermion {

namespace pauli_gates {

// A CliffordGate is either a H, or S, CNOT, SWAP, CZ
using CliffordGate = std::variant<H, S, CNOT, SWAP, CZ>;

// A Clifford circuit is a sequence of Clifford gates
using CliffordCircuit = std::vector<CliffordGate>;

// A gate is either a CliffordGate or a Pauli Rotation
using Gate = std::variant<CliffordGate, ROT>;

// A circuit is a sequence of gates
using Circuit = std::vector<Gate>;

std::pair<PauliString, ff_complex> propagate_clifford(const CliffordCircuit& circuit,
                                                      const PauliString& a) {
    ff_complex coeff = 1;
    PauliString res = a;
    for (int i = circuit.size() - 1; i >= 0; i--) {
        // Dispatch via std::visit over the CliffordGate variant
        std::visit([&res, &coeff](const auto& gate) { gate.apply_inplace(res, coeff); },
                   circuit[i]);
    }
    return std::make_pair(res, coeff);
}

void _apply_clifford_circuit(PauliPolynomial& obs, const Circuit& circuit, int begin, int end) {
    // TODO: avoid extracting into a temporary CliffordCircuit; propagate directly
    CliffordCircuit cc(end - begin);
    for (int j = begin; j < end; ++j) {
        try {
            cc[j - begin] = std::get<CliffordGate>(circuit[j]);
        } catch (const std::bad_variant_access& err) {
            throw_error("Internal error: circuit[begin:end] contains non-Clifford gates");
        }
    }
    PauliPolynomial result;
    for (const auto& [x, coeff] : obs.terms) {
        const auto& [y, mult] = propagate_clifford(cc, x);
        result.terms[y] += mult * coeff;
    }
    obs.terms.swap(result.terms);
}

PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                          const int& maxdegree = 128, const ff_float& mincoeff = 0) {
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford_operations = false;
    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            // Clifford gate -- batch consecutive Cliffords
            if (!pending_clifford_operations) {
                clifford_begin = i;
                pending_clifford_operations = true;
            }
        } else if (circuit[i].index() == 1) {
            // Flush any batched Clifford gates before the rotation
            if (pending_clifford_operations) {
                _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                pending_clifford_operations = false;
            }

            // Apply Pauli rotation gate
            const ROT& gate = std::get<ROT>(circuit[i]);

            // Conjugation rule for Pauli rotations:
            //   ROT_{P,t}(Q) = Q                             if [P, Q] = 0
            //                = cos(t)*Q + i*sin(t)*P*Q       otherwise

            std::vector<std::pair<PauliString, ff_complex>> new_terms;
            new_terms.reserve(obs.terms.size());

            const PauliString& ps = gate.ps;
            const ff_float& theta = gate.theta;
            const ff_float costheta = cos(theta);
            const ff_complex isintheta = ff_complex(0, sin(theta));

            for (auto& [x, coeff] : obs.terms) {
                if (!x.commutes(ps)) {
                    PauliMonomial partner = ps * x;
                    if (partner.degree_total() <= maxdegree) {
                        new_terms.emplace_back(partner.pauli_string(),
                                               coeff * isintheta * partner.coefficient());
                    }
                    coeff *= costheta;
                }
            }

            for (const auto& [x, coeff] : new_terms) {
                obs.terms[x] += coeff;
            }

            if (mincoeff > 0) {
                std::erase_if(obs.terms, [&mincoeff](const auto& term) {
                    return std::abs(term.second) <= mincoeff;
                });
            }
        }
    }
    if (pending_clifford_operations) {
        _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        pending_clifford_operations = false;
    }
    return obs;
}

PauliPolynomial propagate(const Circuit& circuit, const PauliString& a, const int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), maxdegree);
}

}  // namespace pauli_gates

}  // namespace fastfermion
