// Gate-batched Pauli propagation.
// Groups commuting ROT gates and applies each batch in a single pass,
// reducing the number of emit-merge-truncate cycles.

#pragma once
#include <vector>

#include "common.h"
#include "pauli_algebra.h"
#include "pauli_propagation.h"

namespace fastfermion {
namespace pauli_gates {

// Partition a sequence of ROT gates into maximal batches of mutually commuting gates.
// Gates within each batch can be applied in any order (or simultaneously).
std::vector<std::vector<ROT>> batch_commuting_gates(const std::vector<ROT>& gates) {
    std::vector<std::vector<ROT>> batches;
    std::vector<ROT> current;

    for (const auto& gate : gates) {
        // Check if gate commutes with everything in the current batch
        bool commutes_with_all = true;
        for (const auto& g : current) {
            if (!gate.ps.commutes(g.ps)) {
                commutes_with_all = false;
                break;
            }
        }
        if (commutes_with_all) {
            current.push_back(gate);
        } else {
            if (!current.empty()) batches.push_back(std::move(current));
            current = {gate};
        }
    }
    if (!current.empty()) batches.push_back(std::move(current));
    return batches;
}

// Apply a batch of mutually commuting ROT gates to a polynomial in one pass.
// Each input term is conjugated by all gates in the batch before merging.
void apply_batch(PauliPolynomial& obs, const std::vector<ROT>& batch, int maxdegree) {
    // Collect all new terms from the entire batch
    std::vector<std::pair<PauliString, ff_complex>> new_terms;
    new_terms.reserve(obs.terms.size() * batch.size());

    for (const auto& gate : batch) {
        const PauliString& ps = gate.ps;
        const ff_float cos_t = cos(gate.theta);
        const ff_complex isin_t = ff_complex(0, sin(gate.theta));

        // Temporary storage for partners from this gate
        std::vector<std::pair<PauliString, ff_complex>> gate_partners;
        gate_partners.reserve(obs.terms.size());

        for (auto& [x, coeff] : obs.terms) {
            if (!x.commutes(ps)) {
                PauliMonomial partner = ps * x;
                if (partner.degree_total() <= maxdegree)
                    gate_partners.emplace_back(partner.pauli_string(),
                                               coeff * isin_t * partner.coefficient());
                coeff *= cos_t;
            }
        }

        // Insert this gate's partners into the observable
        for (const auto& [k, c] : gate_partners)
            obs.terms[k] += c;
    }
}

// Propagation with gate batching.
// Extracts ROT gates from the circuit, groups commuting ones into batches,
// applies each batch in a single pass.
PauliPolynomial propagate_batched(const Circuit& circuit, const PauliPolynomial& a,
                                   int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                   int max_xweight = -1, int xtrunc_period = 1) {
    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    // Collect consecutive ROT gates, batch them when a Clifford or end is reached
    std::vector<ROT> rot_buffer;

    auto flush_rots = [&]() {
        if (rot_buffer.empty()) return;
        auto batches = batch_commuting_gates(rot_buffer);
        for (const auto& batch : batches) {
            apply_batch(obs, batch, maxdegree);
            rot_count += batch.size();
            if (mincoeff > 0)
                std::erase_if(obs.terms, [mincoeff](const auto& t) {
                    return std::abs(t.second) <= mincoeff;
                });
            if (topk > 0) truncate_top_k(obs, topk);
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                truncate_x_weight(obs, max_xweight);
        }
        rot_buffer.clear();
    };

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            // Clifford gate — flush any pending ROTs, then batch Cliffords
            flush_rots();
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                pending_clifford = false;
            }
            rot_buffer.push_back(std::get<ROT>(circuit[i]));
        }
    }
    flush_rots();
    if (pending_clifford)
        _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);

    return obs;
}

}  // namespace pauli_gates
}  // namespace fastfermion
