/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "backends.h"
#include "majorana/gates.h"
#include "majorana/truncate.h"

#include <string>

namespace fastfermion {

namespace majorana_gates {

// For now, a MajoranaCircuit is simply a sequence of Majorana Rotations
using MajoranaCircuit = std::vector<MROT>;

// Conjugation constants of a Majorana rotation e^{-i theta/2 M} with M = i^r P (see MROT and
// conjugate in backends.h): the partner of a term x is (i i^r sin(theta)) P x
struct Rotation {
    const MajoranaString& axis;
    ff_float cos_t;
    ff_complex isin_t;
    explicit Rotation(const MROT& gate)
        : axis(gate.ms), cos_t(std::cos(gate.theta)),
          isin_t(gate._r ? ff_complex(-std::sin(gate.theta), 0) : ff_complex(0, std::sin(gate.theta))) {}
    static int degree(const MajoranaString& s) { return s.degree(); }
};

// Whether the gates circuit[j..i] all commute with ms
inline bool _commute_with(const MajoranaCircuit& circuit, int j, int i, const MajoranaString& ms) {
    for(int g=j; g<=i; g++) {
        if(!circuit[g].ms.commutes(ms)) return false;
    }
    return true;
}

// Propagates the observable held by the backend through the circuit, last gate first, one window
// at a time -- a single gate, or a maximal run of mutually commuting gates when batching is on --
// with the truncation rules firing after each window
template <class Backend>
MajoranaPolynomial run(const MajoranaCircuit& circuit, Backend& backend, bool batched) {
    int applied = 0;  // gates applied so far, i.e., the gate index of the schedule
    int i = int(circuit.size()) - 1;
    while(i >= 0) {
        int j = i;  // the window is circuit[j..i]
        if(batched) {
            while(j > 0 && _commute_with(circuit, j, i, circuit[j-1].ms)) j--;
        }
        const int first = applied;
        for(int g=i; g>=j; g--, applied++) backend.conjugate(Rotation(circuit[g]));
        backend.truncate(first, applied-1);
        i = j-1;
    }
    return backend.take();
}

// The main Majorana propagation function: Heisenberg evolution of obs through the circuit with
// the given truncation rules (see Truncation and MajoranaTruncation), on the backend named by
// parallel (see select_backend in backends.h) with n_threads OpenMP threads
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const MajoranaTruncation& truncation, bool batched=false, int n_threads=1, const std::string& parallel="auto") {
    if(truncation.maxdegree_period < 1 || truncation.mincoeff_period < 1 || truncation.unpaired_period < 1) {
        throw_error("Truncation periods must be >= 1");
    }
    trunc_stats() = TruncStats();
    switch(select_backend(parallel, n_threads)) {
#ifdef FF_OPENMP
        case Backend::sharded: {
            ShardedBackend<MajoranaPolynomial, MajoranaTruncation> backend(obs, truncation, n_threads);
            return run(circuit, backend, batched);
        }
#endif
        default: {
            SerialBackend<MajoranaPolynomial, MajoranaTruncation> backend(obs, truncation);
            return run(circuit, backend, batched);
        }
    }
}

inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const ff_float& mincoeff=0) {
    MajoranaTruncation truncation;
    truncation.mincoeff = mincoeff;
    return propagate(circuit, obs, truncation);
}

inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const int& maxdegree, const ff_float& mincoeff=0) {
    MajoranaTruncation truncation;
    truncation.maxdegree = maxdegree;
    truncation.mincoeff = mincoeff;
    return propagate(circuit, obs, truncation);
}

inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs, const ff_float& mincoeff=0) {
    return propagate(circuit, MajoranaPolynomial(obs), mincoeff);
}

inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaString& obs, const int& maxdegree, const ff_float& mincoeff=0) {
    return propagate(circuit, MajoranaPolynomial(obs), maxdegree, mincoeff);
}

}

}
