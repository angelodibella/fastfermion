/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "backends.h"
#include "pauli/gates.h"
#include "pauli/truncate.h"

#include <string>
#include <variant>
#include <functional>

namespace fastfermion {

namespace pauli_gates {

// A CliffordGate is either a H, or S, CNOT, SWAP, CZ
using CliffordGate = std::variant<H,S,CNOT,SWAP,CZ>;

// A Clifford circuit is a sequence of Clifford gates
using CliffordCircuit = std::vector<CliffordGate>;

// A gate is either a CliffordGate or a Pauli Rotation
using Gate = std::variant<CliffordGate,ROT>;

// A circuit is a sequence of gates
using Circuit = std::vector<Gate>;

inline std::pair<PauliString, ff_complex> propagate_clifford(const CliffordCircuit& circuit, const PauliString& a) {
    ff_complex coeff = 1;
    PauliString res = a;
    for(int i=circuit.size()-1; i>=0; i--) {
        // Call Clifford gate
        // All CliffordGate structs should have a method:
        //   apply_inplace(PauliString& a, ff_complex& coeff)
        // which applies the gate to coeff*a, and modifies a and coeff in-place
        //
        // I should call circuit[i].apply_inplace(res,coeff) however this
        // raises a compilation error because there is no method called
        // apply_inplace for std::variant<H,S,CNOT>.
        // The way to do this is to use the visit function
        // https://en.cppreference.com/w/cpp/utility/variant/visit2.html
        // See e.g.,
        // https://www.cppstories.com/2020/04/variant-virtual-polymorphism.html/
        // The visit function is essentially equivalent to a switch statement
        // on number of variants (=number of possible Clifford gates)
        std::visit(
            [&res, &coeff](const auto& gate) { gate.apply_inplace(res,coeff); }
            , circuit[i]
        );
    }
    return std::make_pair(res,coeff);
}

// Applies the Clifford gates circuit[begin:end] (last gate first) to all the terms of poly
inline void _apply_clifford_circuit(PauliPolynomial& poly, const Circuit& circuit, int begin, int end) {
    PauliPolynomial poly2;
    for(const auto& [x,val] : poly.terms) {
        PauliString y = x;
        ff_complex mult = 1;
        for(int j=end-1; j>=begin; j--) {
            std::visit([&y, &mult](const auto& gate) { gate.apply_inplace(y,mult); }, std::get<CliffordGate>(circuit[j]));
        }
        poly2.terms[y] += mult*val;
    }
    poly.terms.swap(poly2.terms);
}

// Conjugation constants of a Pauli rotation e^{-i theta/2 P} (see conjugate in backends.h)
struct Rotation {
    const PauliString& axis;
    ff_float cos_t;
    ff_complex isin_t;
    explicit Rotation(const ROT& gate)
        : axis(gate.ps), cos_t(std::cos(gate.theta)), isin_t(0, std::sin(gate.theta)) {}
    static int degree(const PauliString& s) { return s.degree_total(); }
};

// Whether the rotation gates circuit[j..i] all commute with ps
inline bool _commute_with(const Circuit& circuit, int j, int i, const PauliString& ps) {
    for(int g=j; g<=i; g++) {
        if(!std::get<ROT>(circuit[g]).ps.commutes(ps)) return false;
    }
    return true;
}

// Propagates the observable held by the backend through the circuit, last gate first. Rotation
// gates are applied one window at a time -- a single gate, or a maximal run of mutually
// commuting gates when batching is on -- and the truncation rules fire after each window. Runs
// of Clifford gates are applied to all the terms at once.
template <class Backend>
PauliPolynomial run(const Circuit& circuit, Backend& backend, bool batched) {
    int applied = 0;  // rotation gates applied so far, i.e., the gate index of the schedule
    int i = int(circuit.size()) - 1;
    while(i >= 0) {
        int j = i;  // the window is circuit[j..i]
        if(circuit[i].index() == 0) {
            while(j > 0 && circuit[j-1].index() == 0) j--;
            PauliPolynomial obs = backend.take();
            _apply_clifford_circuit(obs, circuit, j, i+1);
            if(j == 0) return obs;
            backend.load(std::move(obs));
        } else {
            if(batched) {
                while(j > 0 && circuit[j-1].index() == 1 && _commute_with(circuit, j, i, std::get<ROT>(circuit[j-1]).ps)) j--;
            }
            const int first = applied;
            for(int g=i; g>=j; g--, applied++) backend.conjugate(Rotation(std::get<ROT>(circuit[g])));
            backend.truncate(first, applied-1);
        }
        i = j-1;
    }
    return backend.take();
}

// The main Pauli propagation function: Heisenberg evolution of a through the circuit with the
// given truncation rules (see Truncation and PauliTruncation), on the backend named by parallel
// (see select_backend in backends.h) with n_threads OpenMP threads
inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a, const PauliTruncation& truncation, bool batched=false, int n_threads=1, const std::string& parallel="auto") {
    if(truncation.maxdegree_period < 1 || truncation.mincoeff_period < 1 || truncation.xweight_period < 1) {
        throw_error("Truncation periods must be >= 1");
    }
    trunc_stats() = TruncStats();
    switch(select_backend(parallel, n_threads)) {
#ifdef FF_OPENMP
        case Backend::sharded: {
            ShardedBackend<PauliPolynomial, PauliTruncation> backend(a, truncation, n_threads);
            return run(circuit, backend, batched);
        }
#endif
        default: {
            SerialBackend<PauliPolynomial, PauliTruncation> backend(a, truncation);
            return run(circuit, backend, batched);
        }
    }
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a, const int& maxdegree=128, const ff_float& mincoeff=0) {
    PauliTruncation truncation;
    truncation.maxdegree = maxdegree;
    truncation.mincoeff = mincoeff;
    return propagate(circuit, a, truncation);
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a, const int maxdegree=128) {
    return propagate(circuit, PauliPolynomial(a), maxdegree);
}

}

}
