/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

// Heisenberg-picture Pauli propagation (Sparse Pauli Dynamics).

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

// X-truncation (xSPD): discard strings with more than max_xweight X/Y factors.
void truncate_x_weight(PauliPolynomial& p, int max_xweight) {
    if (max_xweight < 0) return;
    std::erase_if(p.terms, [max_xweight](const auto& term) {
        return term.first.degree_x() + term.first.degree_y() > max_xweight;
    });
}

// Top-K truncation: keep only the K entries with largest |coefficient|.
void truncate_top_k(PauliPolynomial& p, int k) {
    if (k <= 0 || (int)p.terms.size() <= k) return;
    std::vector<ff_float> mags;
    mags.reserve(p.terms.size());
    for (const auto& [_, c] : p.terms) mags.push_back(std::abs(c));
    std::nth_element(mags.begin(), mags.begin() + k, mags.end(), std::greater<ff_float>());
    ff_float cutoff = mags[k];
    std::erase_if(p.terms, [cutoff](const auto& term) { return std::abs(term.second) < cutoff; });
    // Trim ties at the boundary
    while ((int)p.terms.size() > k) {
        auto it = p.terms.begin();
        while (it != p.terms.end() && std::abs(it->second) > cutoff) ++it;
        if (it != p.terms.end())
            p.terms.erase(it);
        else
            break;
    }
}

PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                          const int& maxdegree = 128, const ff_float& mincoeff = 0,
                          const int topk = 0, const int max_xweight = -1,
                          const int xtrunc_period = 1) {
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford_operations = false;
    int rot_count = 0;
    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            // Clifford gate — batch
            if (!pending_clifford_operations) {
                clifford_begin = i;
                pending_clifford_operations = true;
            }
        } else if (circuit[i].index() == 1) {
            // Flush batched Cliffords
            if (pending_clifford_operations) {
                _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                pending_clifford_operations = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);

            // Q -> Q if [P,Q]=0, else cos(t)Q + i sin(t) PQ

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
            if (topk > 0) truncate_top_k(obs, topk);
            rot_count++;
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                truncate_x_weight(obs, max_xweight);
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
