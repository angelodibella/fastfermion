/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include "backends.h"
#include "gpu/backend.h"
#include "pauli/gates.h"
#include "pauli/truncate.h"

#include <string>
#include <variant>
#include <functional>
#include <random>

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
    ff_float theta;
    ff_float cos_t;
    ff_complex isin_t;
    explicit Rotation(const ROT& gate)
        : axis(gate.ps), theta(gate.theta), cos_t(std::cos(gate.theta)), isin_t(0, std::sin(gate.theta)) {}
    static int degree(const PauliString& s) { return s.degree_total(); }
};

#ifdef FF_GPU

// Number of qubits touched by the circuit and the observable
inline int _extent(const Circuit& circuit, const PauliPolynomial& a) {
    int n = 1;
    for(const auto& [x,c] : a.terms) n = MAX(n, x.extent());
    for(const auto& g : circuit) {
        if(g.index() == 1) {
            n = MAX(n, std::get<ROT>(g).ps.extent());
        } else {
            std::visit([&n](const auto& gate) {
                n = MAX(n, gate.i + 1);
                if constexpr (requires { gate.j; }) n = MAX(n, gate.j + 1);
            }, std::get<CliffordGate>(g));
        }
    }
    return n;
}

// Number of Pauli strings on n qubits of degree <= w, i.e., sum_{r<=w} C(n,r) 3^r, saturating at 4e9
inline double _count_strings(int n, int w) {
    double total = 1, binom = 1, pow3 = 1;
    for(int r=1; r<=MIN(w,n); r++) {
        binom *= double(n-r+1)/r;
        pow3 *= 3;
        total += binom*pow3;
        if(total > 4e9) return 4e9;
    }
    return total;
}

// Conversion between Pauli strings and the flat keys of the GPU engine, and the configuration of
// the engine for a circuit (see GpuBackend in gpu/backend.h): a key is words words of xory bits
// followed by words words of yorz bits
struct GpuCodec {
    int words;
    int key_format = 0;
    std::size_t reserve = 0;
    bool reserve_hard = false;
    double beta;

    GpuCodec(const Circuit& circuit, const PauliPolynomial& a, const PauliTruncation& rules, const GpuOptions& options) : beta(options.beta) {
        const int n = _extent(circuit, a);
        words = (n + WORD_LENGTH - 1) / WORD_LENGTH;
        if(words > 2) throw_error("The gpu backend supports at most 128 qubits");
        if(options.key != "auto" && options.key != "dense" && options.key != "support") {
            throw_error("gpu_key must be \"auto\", \"dense\" or \"support\"");
        }
        // The support-list key is valid when the degree cutoff is enforced at emission and no Clifford
        // gate can raise degrees above it, for degrees <= 14 (7 slots per word), sites <= 126 and
        // rotations on <= 2 sites, and worthwhile when smaller than the dense key
        if(options.key != "dense") {
            int degree = rules.maxdegree, axis = 0;
            bool clifford = false;
            for(const auto& [x,c] : a.terms) degree = MAX(degree, x.degree_total());
            for(const auto& g : circuit) {
                if(g.index() == 1) axis = MAX(axis, std::get<ROT>(g).ps.degree_total());
                else clifford = true;
            }
            if(rules.maxdegree_period == 1 && !clifford && n <= 127 && degree <= 14 && axis <= 2 && (degree <= 7 ? 1 : 2) < 2*words) {
                key_format = degree <= 7 ? 1 : 2;
            } else if(options.key == "support") {
                throw_error("gpu_key=\"support\" requires maxdegree_period=1, no Clifford gate, degrees <= 14, rotations on <= 2 sites, sites <= 126, and a key smaller than the dense one");
            }
        }
        // With the degree cutoff enforced at emission the deduplicated terms never exceed the number
        // of strings of degree <= maxdegree, and the engine deduplicates before the new terms can
        // exceed the deduplicated ones, so twice that number never needs to grow
        if(options.reserve_terms > 0) {
            reserve = 2*std::size_t(MIN(options.reserve_terms, 1LL << 31));
            reserve_hard = true;
        } else if(options.reserve_terms < 0 && rules.maxdegree_period == 1) {
            const double count = _count_strings(n, rules.maxdegree);
            if(count < 2e9) reserve = 2*std::size_t(count) + 64;
        }
    }
    void flatten(const PauliString& s, std::uint64_t* out) const {
        for(int i=0; i<words; i++) {
            out[i] = i < SYS_NUM_ULONG ? s.xory.words[i] : 0;
            out[words+i] = i < SYS_NUM_ULONG ? s.yorz.words[i] : 0;
        }
    }
    PauliString unflatten(const std::uint64_t* in) const {
        ff_ulong xory(0), yorz(0);
        for(int i=0; i<MIN(words,SYS_NUM_ULONG); i++) {
            xory.words[i] = in[i];
            yorz.words[i] = in[words+i];
        }
        return PauliString(xory, yorz);
    }
    // The engine holds real coefficients: those of a Hermitian observable stay real under rotations
    double coeff_to_device(const PauliString& s, const ff_complex& c) const {
        if(std::abs(c.imag()) > 1e-12*(1+std::abs(c.real()))) throw_error("The gpu backend requires a Hermitian observable (real coefficients)");
        return c.real();
    }
    ff_complex coeff_from_device(const PauliString& s, double c) const { return c; }
};

// Checks the phase rule of the engine against pauli_string_multiply on n_pairs random string pairs
inline bool gpu_phase_check(int n_pairs, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int words = SYS_NUM_ULONG;
    std::vector<std::uint64_t> a(2*std::size_t(words)*n_pairs), b(2*std::size_t(words)*n_pairs);
    std::vector<int> expected(n_pairs);
    for(int i=0; i<n_pairs; i++) {
        ff_ulong ax(0), az(0), bx(0), bz(0);
        for(int w=0; w<words; w++) {
            ax.words[w] = rng(), az.words[w] = rng();
            bx.words[w] = rng(), bz.words[w] = rng();
        }
        PauliString p(ax, az), q(bx, bz);
        for(int w=0; w<words; w++) {
            a[2*std::size_t(words)*i+w] = ax.words[w], a[2*std::size_t(words)*i+words+w] = az.words[w];
            b[2*std::size_t(words)*i+w] = bx.words[w], b[2*std::size_t(words)*i+words+w] = bz.words[w];
        }
        int jpow = 0;
        pauli_string_multiply(p, jpow, q);
        expected[i] = ((jpow % 4) + 4) % 4;
    }
    return gpu::phase_check(a.data(), b.data(), expected.data(), n_pairs, words);
}

#endif // FF_GPU

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
// (see select_backend in backends.h) with n_threads OpenMP threads, or on the GPU (see GpuOptions)
inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a, const PauliTruncation& truncation, bool batched=false, int n_threads=1, const std::string& parallel="auto", const GpuOptions& gpu_options=GpuOptions()) {
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
#ifdef FF_GPU
        case Backend::gpu: {
            if(truncation.topk > 0 || truncation.max_xweight >= 0) throw_error("The gpu backend does not support topk and max_xweight");
            GpuBackend<PauliPolynomial, PauliTruncation, GpuCodec> backend(a, truncation, GpuCodec(circuit, a, truncation, gpu_options));
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
