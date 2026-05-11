// Clifford gates, Pauli rotation, and circuit types for Heisenberg-picture propagation.

#pragma once

#include <functional>
#include <variant>

#include "bits_utils.h"
#include "pauli/algebra.h"

namespace fastfermion {

namespace pauli_gates {

void H_impl(const int& i, PauliString& a, ff_complex& coeff) {
    // X -> Z,  Z -> X,  Y -> -Y
    // out.xory = in.yorz
    // out.yorz = in.xory
    // out.phase = (-1)^{in.xory & in.yorz}
    ff_ulong mask = ff_ulong::singleton(i);
    ff_ulong xory_i = a.xory & mask;
    ff_ulong yorz_i = a.yorz & mask;
    a.xory = (a.xory & (~mask)) | yorz_i;
    a.yorz = (a.yorz & (~mask)) | xory_i;
    if (!((xory_i & yorz_i) == 0)) coeff *= -1;
}

void S_impl(const int& i, PauliString& a, ff_complex& coeff) {
    // X -> -Y,  Z -> Z,  Y -> X
    // out.xory = in.xory
    // out.yorz = in.xory ^ in.yorz (=in.xorz)
    // out.phase = (-1)^{in.xory & in.yorz}
    ff_ulong mask = ff_ulong::singleton(i);
    ff_ulong xory_i = a.xory & mask;
    ff_ulong yorz_i = a.yorz & mask;
    a.yorz = (a.yorz & (~mask)) | (xory_i ^ yorz_i);
    if (!((xory_i & (~yorz_i)) == 0)) coeff *= -1;
}

void CNOT_impl(const int& i, const int& j, PauliString& a, ff_complex& coeff) {
    // Generating set:   XI -> XX,  ZI -> ZI,  IX -> IX,  IZ -> ZZ
    ff_ulong mask_i = ff_ulong::singleton(i);
    ff_ulong mask_j = ff_ulong::singleton(j);

    ff_ulong xory_j = a.xory & mask_j;
    ff_ulong xory_i_at_j = ((a.xory & mask_i) >> i) << j;
    ff_ulong yorz_i = a.yorz & mask_i;
    ff_ulong yorz_j_at_i = ((a.yorz & mask_j) >> j) << i;
    a.xory = (a.xory & ~mask_j) | (xory_i_at_j ^ xory_j);
    a.yorz = (a.yorz & ~mask_i) | (yorz_i ^ yorz_j_at_i);

    if (!(xory_i_at_j == 0) && !(yorz_j_at_i == 0) && !((yorz_i == 0) ^ (xory_j == 0))) coeff *= -1;
}

void SWAP_impl(int i, int j, PauliString& a, ff_complex& coeff) {
    // XI -> IX,  ZI -> IZ,  IX -> XI,  IZ -> ZI
    swap_bits_inplace(a.xory, i, j);
    swap_bits_inplace(a.yorz, i, j);
}

void CZ_impl(int i, int j, PauliString& a, ff_complex& coeff) {
    // Generating set:   XI -> XZ,  ZI -> ZI,  IX -> ZX,  IZ -> IZ
    ff_ulong mask_i = ff_ulong::singleton(i);
    ff_ulong mask_j = ff_ulong::singleton(j);

    if (a.xory.at(i) && a.xory.at(j) && (a.yorz.at(i) ^ a.yorz.at(j))) coeff *= -1;

    a.yorz ^= (((a.xory & mask_j) >> j) << i) | (((a.xory & mask_i) >> i) << j);
}

struct H {
    int i;
    void apply_inplace(PauliString& a, ff_complex& coeff) const { H_impl(i, a, coeff); }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res, c);
        return std::make_pair(res, c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for (const auto& [x, v] : p.terms) {
            const auto [y, w] = (*this)(x);
            res.terms[y] += w * v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString(std::vector<int>{i}, std::vector<char>{'X'})] = 1 / std::sqrt(2);
        poly.terms[PauliString(std::vector<int>{i}, std::vector<char>{'Z'})] = 1 / std::sqrt(2);
        return poly;
    }
    std::string to_string() const { return "H(" + std::to_string(i) + ")"; }
};

struct S {
    int i;
    void apply_inplace(PauliString& a, ff_complex& coeff) const { S_impl(i, a, coeff); }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res, c);
        return std::make_pair(res, c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for (const auto& [x, v] : p.terms) {
            const auto [y, w] = (*this)(x);
            res.terms[y] += w * v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString()] = ff_complex(.5, .5);
        poly.terms[PauliString(std::vector<int>{i}, std::vector<char>{'Z'})] = ff_complex(.5, -.5);
        return poly;
    }
    std::string to_string() const { return "S(" + std::to_string(i) + ")"; }
};

struct CNOT {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const { CNOT_impl(i, j, a, coeff); }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res, c);
        return std::make_pair(res, c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for (const auto& [x, v] : p.terms) {
            const auto [y, w] = (*this)(x);
            res.terms[y] += w * v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i}, std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{j}, std::vector<char>{'X'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i, j}, std::vector<char>{'Z', 'X'})] = -0.5;
        return poly;
    }
    std::string to_string() const {
        return "CNOT(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

struct SWAP {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const { SWAP_impl(i, j, a, coeff); }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res, c);
        return std::make_pair(res, c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for (const auto& [x, v] : p.terms) {
            const auto [y, w] = (*this)(x);
            res.terms[y] += w * v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i, j}, std::vector<char>{'X', 'X'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i, j}, std::vector<char>{'Y', 'Y'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i, j}, std::vector<char>{'Z', 'Z'})] = 0.5;
        return poly;
    }
    std::string to_string() const {
        return "SWAP(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

struct CZ {
    int i;
    int j;
    void apply_inplace(PauliString& a, ff_complex& coeff) const { CZ_impl(i, j, a, coeff); }
    std::pair<PauliString, ff_complex> operator()(const PauliString& a) const {
        PauliString res(a);
        ff_complex c = 1;
        apply_inplace(res, c);
        return std::make_pair(res, c);
    }
    PauliPolynomial operator()(const PauliPolynomial& p) const {
        PauliPolynomial res;
        for (const auto& [x, v] : p.terms) {
            const auto [y, w] = (*this)(x);
            res.terms[y] += w * v;
        }
        return res;
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString()] = 0.5;
        poly.terms[PauliString(std::vector<int>{i}, std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{j}, std::vector<char>{'Z'})] = 0.5;
        poly.terms[PauliString(std::vector<int>{i, j}, std::vector<char>{'Z', 'Z'})] = -0.5;
        return poly;
    }
    std::string to_string() const {
        return "CZ(" + std::to_string(i) + "," + std::to_string(j) + ")";
    }
};

// Pauli rotation gate: U = exp(-i theta/2 P) where P is a PauliString.
struct ROT {
    PauliString ps;
    ff_float theta;
    ROT(const PauliString& ps, const ff_float& theta) : ps(ps), theta(theta) {}
    ROT(const std::string& str, const ff_float& theta) : ps(PauliString(str)), theta(theta) {}
    ROT(const std::string& str, const std::vector<int>& loc, const ff_float& theta)
        : ps(PauliString(loc, std::vector<char>(str.begin(), str.end()))), theta(theta) {}
    std::string to_string() const {
        return "ROT(" + ps.to_compact_string() + "," + std::to_string(theta) + ")";
    }
    PauliPolynomial aspoly() const {
        PauliPolynomial poly;
        poly.terms[PauliString()] = std::cos(theta / 2);
        poly.terms[ps] += ff_complex(0, -std::sin(theta / 2));
        return poly;
    }

    PauliPolynomial operator()(const PauliPolynomial& o) const {
        PauliPolynomial obs(o);
        std::vector<std::pair<PauliString, ff_complex>> new_terms;
        new_terms.reserve(o.terms.size());
        const ff_float costheta = cos(theta);
        const ff_complex isintheta = ff_complex(0, sin(theta));
        for (auto& [x, v] : obs.terms) {
            if (!x.commutes(ps)) {
                PauliMonomial partner = ps * x;
                new_terms.emplace_back(partner.pauli_string(),
                                       v * isintheta * partner.coefficient());
                v *= costheta;
            }
        }
        for (const auto& [x, v] : new_terms) {
            obs.terms[x] += v;
        }
        return obs;
    }

    PauliPolynomial operator()(const PauliString& o) const { return (*this)(PauliPolynomial(o)); }
};

// ------------------------------------------------------------------------------------------------

// A CliffordGate is either a H, or S, CNOT, SWAP, CZ
using CliffordGate = std::variant<H, S, CNOT, SWAP, CZ>;

// A Clifford circuit is a sequence of Clifford gates
using CliffordCircuit = std::vector<CliffordGate>;

// A gate is either a CliffordGate or a Pauli Rotation
using Gate = std::variant<CliffordGate, ROT>;

// A circuit is a sequence of gates
using Circuit = std::vector<Gate>;

// ------------------------------------------------------------------------------------------------

inline std::pair<PauliString, ff_complex> propagate_clifford(const CliffordCircuit& circuit,
                                                             const PauliString& a) {
    ff_complex coeff = 1;
    PauliString res = a;
    for (int i = circuit.size() - 1; i >= 0; i--) {
        std::visit([&res, &coeff](const auto& gate) { gate.apply_inplace(res, coeff); },
                   circuit[i]);
    }
    return std::make_pair(res, coeff);
}

inline void _apply_clifford_circuit(PauliPolynomial& obs, const Circuit& circuit, int begin,
                                    int end) {
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

}  // namespace pauli_gates

}  // namespace fastfermion
