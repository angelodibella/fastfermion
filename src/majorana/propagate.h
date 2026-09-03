/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "backends.h"
#include "gpu/backend.h"
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
    ff_float theta;
    ff_float cos_t;
    ff_complex isin_t;
    explicit Rotation(const MROT& gate)
        : axis(gate.ms), theta(gate.theta), cos_t(std::cos(gate.theta)),
          isin_t(gate._r ? ff_complex(-std::sin(gate.theta), 0) : ff_complex(0, std::sin(gate.theta))) {}
    static int degree(const MajoranaString& s) { return s.degree(); }
};

#ifdef FF_GPU

// Number of subsets of size <= d of n Majorana operators, i.e., sum_{k<=d} C(n,k), saturating at 4e9
inline double _count_monomials(int n, int d) {
    double total = 0, binom = 1;
    for(int k=0; k<=MIN(d,n); k++) {
        total += binom;
        if(total > 4e9) return 4e9;
        binom = binom*(n-k)/(k+1);
    }
    return total;
}

// Conversion between Majorana strings and the flat keys of the GPU engine, and the configuration
// of the engine for a circuit (see GpuBackend in gpu/backend.h): the key of a monomial is its
// index mask, in 2*words words. Coefficients cross as those of the Hermitian monomials
// i^{m(k)} gamma_S, m(k) = k(k-1)/2, so that gates act with real factors on the device.
struct GpuCodec {
    int words;
    int key_format = -1;
    std::size_t reserve = 0;
    bool reserve_hard = false;
    double beta;

    GpuCodec(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const MajoranaTruncation& rules, const GpuOptions& options) : beta(options.beta) {
        int n = 1;  // number of Majorana operators
        for(const auto& [x,c] : obs.terms) n = MAX(n, x.extent());
        for(const auto& g : circuit) n = MAX(n, g.ms.extent());
        words = (n + 2*WORD_LENGTH - 1) / (2*WORD_LENGTH);
        if(words > 2) throw_error("The gpu backend supports at most 128 modes");
        if(options.key != "auto" && options.key != "dense") throw_error("gpu_key must be \"auto\" or \"dense\" for Majorana polynomials");
        // As for Pauli strings: the number of monomials of degree <= maxdegree bounds the
        // deduplicated terms when the cutoff is enforced at emission
        if(options.reserve_terms > 0) {
            reserve = 2*std::size_t(MIN(options.reserve_terms, 1LL << 31));
            reserve_hard = true;
        } else if(options.reserve_terms < 0 && rules.maxdegree_period == 1) {
            const double count = _count_monomials(n, rules.maxdegree);
            if(count < 2e9) reserve = 2*std::size_t(count) + 64;
        }
    }
    void flatten(const MajoranaString& s, std::uint64_t* out) const {
        for(int k=0; k<2*words; k++) out[k] = k < int(s.alpha.words.size()) ? s.alpha.words[k] : 0;
    }
    MajoranaString unflatten(const std::uint64_t* in) const {
        MajoranaString s;
        for(int k=0; k<2*words && k<int(s.alpha.words.size()); k++) s.alpha.words[k] = in[k];
        return s;
    }
    static bool _phase(int degree) { return (degree*(degree-1)/2) & 1; }  // m(k)
    double coeff_to_device(const MajoranaString& s, const ff_complex& c) const {
        const ff_complex h = _phase(s.degree()) ? c*ff_complex(0,-1) : c;
        if(std::abs(h.imag()) > 1e-12*(1+std::abs(h.real()))) throw_error("The gpu backend requires a Hermitian observable");
        return h.real();
    }
    ff_complex coeff_from_device(const MajoranaString& s, double c) const {
        return _phase(s.degree()) ? ff_complex(0,c) : ff_complex(c,0);
    }
};

#endif // FF_GPU

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
// parallel (see select_backend in backends.h) with n_threads OpenMP threads, or on the GPU (see
// GpuOptions)
inline MajoranaPolynomial propagate(const MajoranaCircuit& circuit, const MajoranaPolynomial& obs, const MajoranaTruncation& truncation, bool batched=false, int n_threads=1, const std::string& parallel="auto", const GpuOptions& gpu_options=GpuOptions()) {
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
#ifdef FF_GPU
        case Backend::gpu: {
            if(truncation.topk > 0 || truncation.max_unpaired >= 0) throw_error("The gpu backend does not support topk and max_unpaired");
            GpuBackend<MajoranaPolynomial, MajoranaTruncation, GpuCodec> backend(obs, truncation, GpuCodec(circuit, obs, truncation, gpu_options));
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
