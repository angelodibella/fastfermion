// Sorted-vector Pauli propagation: sort-and-merge alternative to hash-map insertion.
// All sorted internals are in this file: conversion, merge, radix sort, conjugation.

#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common.h"
#include "pauli/algebra.h"
#include "pauli/gates.h"

namespace fastfermion {
namespace pauli_gates {

// Flat sorted representation of a Pauli polynomial, used during sorted propagation.
// Entries are sorted by PauliString key (via operator<). This is an ephemeral
// working format; the persistent representation is PauliPolynomial (hash map).
using SortedTerms = std::vector<std::pair<PauliString, ff_complex>>;

// Convert PauliPolynomial (hash map) to a sorted flat vector.
inline SortedTerms to_sorted(const PauliPolynomial& p) {
    SortedTerms out(p.terms.begin(), p.terms.end());
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// Convert sorted flat vector back to PauliPolynomial (hash map).
inline PauliPolynomial to_poly(const SortedTerms& s) {
    PauliPolynomial p;
    for (const auto& [k, c] : s) p.terms[k] += c;
    return p;
}

// Merge two sorted vectors into one, summing coefficients on matching keys.
// This is the sorted-path analogue of hash-map insertion: O(K) sequential access
// vs O(K) random access for the hash map.
inline SortedTerms sorted_merge(const SortedTerms& a, const SortedTerms& b) {
    SortedTerms result;
    result.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first < b[j].first)
            result.push_back(a[i++]);
        else if (b[j].first < a[i].first)
            result.push_back(b[j++]);
        else {
            result.emplace_back(a[i].first, a[i].second + b[j].second);
            i++;
            j++;
        }
    }
    for (; i < a.size(); i++) result.push_back(a[i]);
    for (; j < b.size(); j++) result.push_back(b[j]);
    return result;
}

// --- Radix sort for SortedTerms ---
//
// LSD (least-significant-digit) radix sort on the PauliString key.
// The sort key matches PauliString::operator<, which compares the tuple
// (support, yorz, xory) lexicographically, each component a BitSet<N_WORDS>.
// We decompose this into 3 * N_WORDS * 8 bytes, processed from least to most
// significant. Bytes that are all-zero across the input are skipped.

namespace detail {

constexpr int RADIX = 256;
constexpr int N_WORDS = SYS_NUM_ULONG;
// Total bytes in the sort key: 3 components x N_WORDS words x 8 bytes/word.
constexpr int SORT_KEY_BYTES = 3 * N_WORDS * 8;

// Extract the word_idx-th 64-bit word from the composite sort key.
// Words 0..N-1: xory, N..2N-1: yorz, 2N..3N-1: support (= xory | yorz).
inline std::uint64_t sort_key_word(const PauliString& ps, int word_idx) {
    if (word_idx < N_WORDS)
        return ps.xory.words[word_idx];
    else if (word_idx < 2 * N_WORDS)
        return ps.yorz.words[word_idx - N_WORDS];
    else
        return (ps.xory | ps.yorz).words[word_idx - 2 * N_WORDS];
}

// Extract one byte from the composite sort key.
inline std::uint8_t sort_key_byte(const PauliString& ps, int byte_idx) {
    return static_cast<std::uint8_t>(sort_key_word(ps, byte_idx / 8) >> ((byte_idx % 8) * 8));
}

// True if byte_idx is zero for every entry (skip this byte in the radix sort).
inline bool byte_is_zero(const SortedTerms& v, int byte_idx) {
    for (const auto& [ps, _] : v)
        if (sort_key_byte(ps, byte_idx) != 0) return false;
    return true;
}

}  // namespace detail

// Sort a SortedTerms vector in place by PauliString key using LSD radix sort.
// O(K) per byte of key, vs O(K log K) for std::sort. The caller may supply
// a scratch buffer `buf` to avoid reallocation across repeated calls.
inline void radix_sort(SortedTerms& terms, SortedTerms& buf) {
    const size_t n = terms.size();
    if (n <= 1) return;
    buf.resize(n);

    SortedTerms* src = &terms;
    SortedTerms* dst = &buf;
    int passes = 0;

    for (int b = 0; b < detail::SORT_KEY_BYTES; b++) {
        if (detail::byte_is_zero(*src, b)) continue;

        size_t counts[detail::RADIX] = {};
        for (size_t i = 0; i < n; i++) counts[detail::sort_key_byte((*src)[i].first, b)]++;

        size_t offsets[detail::RADIX];
        offsets[0] = 0;
        for (int r = 1; r < detail::RADIX; r++) offsets[r] = offsets[r - 1] + counts[r - 1];

        for (size_t i = 0; i < n; i++) {
            auto digit = detail::sort_key_byte((*src)[i].first, b);
            (*dst)[offsets[digit]] = (*src)[i];
            offsets[digit]++;
        }
        std::swap(src, dst);
        passes++;
    }

    if (passes % 2 != 0) terms = buf;
}

// Convenience overload that allocates its own scratch buffer.
inline void radix_sort(SortedTerms& terms) {
    SortedTerms buf;
    radix_sort(terms, buf);
}

// --- Per-gate sorted conjugation ---

enum class SortMethod { comparison, radix };

// Apply one ROT gate to a sorted polynomial: emit kept/partner streams,
// sort the partner stream, then merge. This is the sorted-path analogue
// of conjugate() in propagate.h (which uses hash-map insertion instead).
// `sort_buf` is reused across gates to avoid repeated allocation.
inline void conjugate_sorted(SortedTerms& obs, const ROT& gate, int maxdegree, SortMethod method,
                             SortedTerms& sort_buf) {
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    // Emit: commuting terms pass through; anticommuting terms produce
    // a scaled copy (kept) and a partner string (needs sorting).
    SortedTerms kept, partners;
    kept.reserve(obs.size());
    partners.reserve(obs.size());

    for (const auto& [q, c] : obs) {
        if (q.commutes(gate.ps)) {
            kept.emplace_back(q, c);
        } else {
            kept.emplace_back(q, c * cos_t);
            PauliMonomial pq = gate.ps * q;
            if (pq.degree_total() <= maxdegree)
                partners.emplace_back(pq.pauli_string(), c * isin_t * pq.coefficient());
        }
    }

    // Sort partners (the kept stream is already sorted since the input was).
    if (method == SortMethod::radix)
        radix_sort(partners, sort_buf);
    else
        std::sort(partners.begin(), partners.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

    obs = sorted_merge(kept, partners);
}

}  // namespace pauli_gates
}  // namespace fastfermion
