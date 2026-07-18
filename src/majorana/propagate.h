// Sparse Majorana dynamics: Heisenberg propagation of a MajoranaPolynomial
// through a circuit of Majorana rotations, with the same truncation rules,
// schedules, and per-run certificate as the Pauli driver (pauli/propagate.h)
// -- the transfer is licensed by workbook rem:keyed-systems: monomials are a
// keyed orthonormal basis, gates act entrywise and isometrically, and the
// structural cutoff reads only the key. v1 ships the serial backend with the
// full knob set; the parallel (omp/sharded) and GPU backends follow the
// Pauli implementations and land next.

#pragma once
#include <climits>

#include "common.h"
#include "majorana/gates.h"
#include "majorana/truncate.h"

namespace fastfermion {

namespace majorana_gates {

// For now, a MajoranaCircuit is simply a sequence of Majorana Rotations
using MajoranaCircuit = std::vector<MROT>;

// Batching: a maximal run of consecutive gates whose generators pairwise
// commute forms one window; rules fire once per window on the cadence of
// truncate_all. Under the structural cutoff the batched and per-gate
// retained sets are identical (compaction invariance); under coefficient
// rules the window is part of the rule's schedule, as on the Pauli side.
inline bool _commutes_with_batch(const std::vector<const MROT*>& batch, const MROT& g) {
    for (const MROT* b : batch)
        if (!b->ms.commutes(g.ms)) return false;
    return true;
}

// Full-featured serial propagation. The circuit is applied in Heisenberg
// order (last gate first); gate indices reported to the scheduler count
// applied rotations, so period semantics match the Pauli driver exactly.
inline MajoranaPolynomial propagate_serial(const MajoranaCircuit& circuit,
                                           const MajoranaPolynomial& obs, int maxdegree,
                                           ff_float mincoeff, int topk, int maxdegree_period,
                                           int mincoeff_period, bool batched) {
    MajoranaPolynomial ret(obs);
    const int L = (int)circuit.size();
    // Emission filter: the structural cutoff acts at emission when its period
    // is 1 (the controlled default); a deferred cutoff lifts the filter and
    // prunes at scheduled events instead (a different rule -- see workbook).
    const int emit_deg = (maxdegree_period <= 1) ? maxdegree : INT_MAX;
    auto emit_ok = [emit_deg](const MajoranaString& s) { return s.degree() <= emit_deg; };

    int applied = 0;  // rotations applied so far (window bookkeeping)
    int i = L - 1;
    while (i >= 0) {
        int first = applied;
        if (batched) {
            std::vector<const MROT*> batch;
            batch.push_back(&circuit[i]);
            circuit[i].apply_inplace(ret, emit_ok);
            applied++;
            int j = i - 1;
            while (j >= 0 && _commutes_with_batch(batch, circuit[j])) {
                batch.push_back(&circuit[j]);
                circuit[j].apply_inplace(ret, emit_ok);
                applied++;
                j--;
            }
            i = j;
        } else {
            circuit[i].apply_inplace(ret, emit_ok);
            applied++;
            i--;
        }
        truncate_all(ret, mincoeff, topk, applied - 1, maxdegree, maxdegree_period,
                     mincoeff_period, first);
    }
    return ret;
}

// Public driver: knob set mirrors pauli_gates::propagate one-for-one (minus
// the spin-only x-weight pair); parallel backend selection arrives with the
// omp/sharded ports. Stats reset here so ff.trunc_stats() reports this run.
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs,
                                    int maxdegree = INT_MAX, ff_float mincoeff = 0, int topk = 0,
                                    int maxdegree_period = 1, int mincoeff_period = 1,
                                    bool batched = true) {
    trunc_stats().reset();
    return propagate_serial(circuit, obs, maxdegree, mincoeff, topk, maxdegree_period,
                            mincoeff_period, batched);
}

// Back-compatible overloads (the pre-port public surface).
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs,
                                    const ff_float& mincoeff) {
    return propagate(circuit, obs, INT_MAX, mincoeff);
}
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs,
                                    const ff_float& mincoeff = 0) {
    return propagate(circuit, MajoranaPolynomial(obs), INT_MAX, mincoeff);
}
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs,
                                    const int& maxdegree, const ff_float& mincoeff = 0) {
    return propagate(circuit, MajoranaPolynomial(obs), maxdegree, mincoeff);
}

}  // namespace majorana_gates

}  // namespace fastfermion
