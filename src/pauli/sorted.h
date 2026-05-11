// Sorted parallel-array Pauli polynomial: struct, merge, radix sort, truncation, conjugation.

#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "common.h"
#include "pauli/algebra.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {


// --- SortedPauliPoly ---

// Sorted parallel arrays (keys, coeffs), no duplicate keys.
struct SortedPauliPoly {
    std::vector<PauliString> keys;
    std::vector<ff_complex> coeffs;

    SortedPauliPoly() = default;

    static SortedPauliPoly from_poly(const PauliPolynomial& p) {
        std::vector<std::pair<PauliString, ff_complex>> entries(p.terms.begin(), p.terms.end());
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        SortedPauliPoly result;
        result.keys.reserve(entries.size());
        result.coeffs.reserve(entries.size());
        for (const auto& [k, c] : entries) {
            result.keys.push_back(k);
            result.coeffs.push_back(c);
        }
        return result;
    }

    PauliPolynomial to_poly() const { return PauliPolynomial(keys, coeffs); }

    size_t size() const { return keys.size(); }

    // Merge two sorted polynomials, summing coefficients on matching keys.
    static SortedPauliPoly merge(const SortedPauliPoly& a, const SortedPauliPoly& b) {
        SortedPauliPoly result;
        result.keys.reserve(a.size() + b.size());
        result.coeffs.reserve(a.size() + b.size());

        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a.keys[i] < b.keys[j]) {
                result.keys.push_back(a.keys[i]);
                result.coeffs.push_back(a.coeffs[i++]);
            } else if (b.keys[j] < a.keys[i]) {
                result.keys.push_back(b.keys[j]);
                result.coeffs.push_back(b.coeffs[j++]);
            } else {
                result.keys.push_back(a.keys[i]);
                result.coeffs.push_back(a.coeffs[i++] + b.coeffs[j++]);
            }
        }
        for (; i < a.size(); i++) {
            result.keys.push_back(a.keys[i]);
            result.coeffs.push_back(a.coeffs[i]);
        }
        for (; j < b.size(); j++) {
            result.keys.push_back(b.keys[j]);
            result.coeffs.push_back(b.coeffs[j]);
        }
        return result;
    }
};

#ifdef FF_OPENMP

// Parallel merge: split both arrays into aligned chunks via binary search,
// merge each chunk independently.
inline SortedPauliPoly sorted_merge_omp(const SortedPauliPoly& a, const SortedPauliPoly& b,
                                        int n_threads) {
    if (a.size() == 0) return b;
    if (b.size() == 0) return a;

    // Choose split points from the larger array, find matching positions in the smaller
    const SortedPauliPoly& big = (a.size() >= b.size()) ? a : b;
    const SortedPauliPoly& small = (a.size() >= b.size()) ? b : a;

    // Divide the big array into n_threads chunks
    std::vector<size_t> big_splits(n_threads + 1);
    std::vector<size_t> small_splits(n_threads + 1);
    big_splits[0] = 0;
    big_splits[n_threads] = big.size();
    small_splits[0] = 0;
    small_splits[n_threads] = small.size();

    for (int t = 1; t < n_threads; t++) {
        big_splits[t] = (big.size() * t) / n_threads;
        // Find where big's split key falls in small's array
        const PauliString& split_key = big.keys[big_splits[t]];
        small_splits[t] =
            std::lower_bound(small.keys.begin(), small.keys.end(), split_key) - small.keys.begin();
    }

    // Each thread merges its chunk into a local result
    std::vector<SortedPauliPoly> chunks(n_threads);

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int t = 0; t < n_threads; t++) {
        size_t bi = big_splits[t], b_end = big_splits[t + 1];
        size_t si = small_splits[t], s_end = small_splits[t + 1];

        auto& chunk = chunks[t];
        chunk.keys.reserve((b_end - bi) + (s_end - si));
        chunk.coeffs.reserve((b_end - bi) + (s_end - si));

        while (bi < b_end && si < s_end) {
            if (big.keys[bi] < small.keys[si]) {
                chunk.keys.push_back(big.keys[bi]);
                chunk.coeffs.push_back(big.coeffs[bi++]);
            } else if (small.keys[si] < big.keys[bi]) {
                chunk.keys.push_back(small.keys[si]);
                chunk.coeffs.push_back(small.coeffs[si++]);
            } else {
                chunk.keys.push_back(big.keys[bi]);
                chunk.coeffs.push_back(big.coeffs[bi++] + small.coeffs[si++]);
            }
        }
        for (; bi < b_end; bi++) {
            chunk.keys.push_back(big.keys[bi]);
            chunk.coeffs.push_back(big.coeffs[bi]);
        }
        for (; si < s_end; si++) {
            chunk.keys.push_back(small.keys[si]);
            chunk.coeffs.push_back(small.coeffs[si]);
        }
    }

    // Concatenate chunks (already in order)
    size_t total = 0;
    for (const auto& chunk : chunks) total += chunk.size();

    SortedPauliPoly result;
    result.keys.reserve(total);
    result.coeffs.reserve(total);
    for (const auto& chunk : chunks) {
        result.keys.insert(result.keys.end(), chunk.keys.begin(), chunk.keys.end());
        result.coeffs.insert(result.coeffs.end(), chunk.coeffs.begin(), chunk.coeffs.end());
    }
    return result;
}

#endif  // FF_OPENMP

// --- Radix sort ---

namespace detail {

constexpr int RADIX = 256;
constexpr int N_WORDS = SYS_NUM_ULONG;
// 3 components (support, yorz, xory) x N_WORDS words x 8 bytes/word
constexpr int SORT_KEY_BYTES = 3 * N_WORDS * 8;

// Extract the sort key word at position `word_idx` (0 = least significant).
inline std::uint64_t sort_key_word(const PauliString& ps, int word_idx) {
    if (word_idx < N_WORDS)
        return ps.xory.words[word_idx];
    else if (word_idx < 2 * N_WORDS)
        return ps.yorz.words[word_idx - N_WORDS];
    else
        return (ps.xory | ps.yorz).words[word_idx - 2 * N_WORDS];
}

// Extract byte `byte_idx` (0 = least significant byte of LSW) from a sort key.
inline std::uint8_t sort_key_byte(const PauliString& ps, int byte_idx) {
    int word = byte_idx / 8;
    int shift = (byte_idx % 8) * 8;
    return static_cast<std::uint8_t>(sort_key_word(ps, word) >> shift);
}

// Check whether byte position `byte_idx` is all-zero across the array.
inline bool byte_is_zero(const PauliString* keys, size_t n, int byte_idx) {
    for (size_t i = 0; i < n; i++)
        if (sort_key_byte(keys[i], byte_idx) != 0) return false;
    return true;
}

}  // namespace detail

// Sort parallel (keys, coeffs) arrays in-place using LSD radix sort.
// Requires scratch buffers of the same size; caller may supply them
// to amortise allocation across gates.
inline void radix_sort_pauli(std::vector<PauliString>& keys, std::vector<ff_complex>& coeffs,
                             std::vector<PauliString>& keys_buf,
                             std::vector<ff_complex>& coeffs_buf) {
    const size_t n = keys.size();
    if (n <= 1) return;

    keys_buf.resize(n);
    coeffs_buf.resize(n);

    PauliString* src_k = keys.data();
    ff_complex* src_c = coeffs.data();
    PauliString* dst_k = keys_buf.data();
    ff_complex* dst_c = coeffs_buf.data();

    int passes = 0;
    for (int b = 0; b < detail::SORT_KEY_BYTES; b++) {
        if (detail::byte_is_zero(src_k, n, b)) continue;

        size_t counts[detail::RADIX] = {};
        for (size_t i = 0; i < n; i++) counts[detail::sort_key_byte(src_k[i], b)]++;

        size_t offsets[detail::RADIX];
        offsets[0] = 0;
        for (int r = 1; r < detail::RADIX; r++) offsets[r] = offsets[r - 1] + counts[r - 1];

        for (size_t i = 0; i < n; i++) {
            std::uint8_t digit = detail::sort_key_byte(src_k[i], b);
            dst_k[offsets[digit]] = src_k[i];
            dst_c[offsets[digit]] = src_c[i];
            offsets[digit]++;
        }

        std::swap(src_k, dst_k);
        std::swap(src_c, dst_c);
        passes++;
    }

    // If odd number of passes, result is in the buffer; copy back.
    if (passes % 2 != 0) {
        std::memcpy(keys.data(), keys_buf.data(), n * sizeof(PauliString));
        std::memcpy(coeffs.data(), coeffs_buf.data(), n * sizeof(ff_complex));
    }
}

// Convenience overload that allocates its own scratch buffers.
inline void radix_sort_pauli(std::vector<PauliString>& keys, std::vector<ff_complex>& coeffs) {
    std::vector<PauliString> kb;
    std::vector<ff_complex> cb;
    radix_sort_pauli(keys, coeffs, kb, cb);
}

// --- Sorted truncation ---

namespace pauli_gates {

inline void truncate_threshold(SortedPauliPoly& p, ff_float mincoeff) {
    size_t j = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (std::abs(p.coeffs[i]) > mincoeff) {
            if (j != i) {
                p.keys[j] = p.keys[i];
                p.coeffs[j] = p.coeffs[i];
            }
            j++;
        }
    }
    p.keys.resize(j);
    p.coeffs.resize(j);
}

inline void truncate_top_k(SortedPauliPoly& p, int k) {
    if (k <= 0 || (int)p.size() <= k) return;

    std::vector<size_t> idx(p.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](size_t a, size_t b) {
        return std::abs(p.coeffs[a]) > std::abs(p.coeffs[b]);
    });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());

    std::vector<PauliString> new_keys;
    std::vector<ff_complex> new_coeffs;
    new_keys.reserve(k);
    new_coeffs.reserve(k);
    for (size_t i : idx) {
        new_keys.push_back(p.keys[i]);
        new_coeffs.push_back(p.coeffs[i]);
    }
    p.keys = std::move(new_keys);
    p.coeffs = std::move(new_coeffs);
}

inline void truncate_x_weight(SortedPauliPoly& p, int max_xweight) {
    if (max_xweight < 0) return;
    size_t j = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (p.keys[i].degree_x() + p.keys[i].degree_y() <= max_xweight) {
            if (j != i) {
                p.keys[j] = p.keys[i];
                p.coeffs[j] = p.coeffs[i];
            }
            j++;
        }
    }
    p.keys.resize(j);
    p.coeffs.resize(j);
}

inline void truncate_all(SortedPauliPoly& obs, ff_float mincoeff, int topk, int max_xweight,
                         int xtrunc_period, int rot_count) {
    if (mincoeff > 0) truncate_threshold(obs, mincoeff);
    if (topk > 0) truncate_top_k(obs, topk);
    if (max_xweight >= 0 && xtrunc_period > 0 && (rot_count + 1) % xtrunc_period == 0)
        truncate_x_weight(obs, max_xweight);
}

// --- Observable-type dispatch helpers ---

template <typename Obs>
Obs obs_from_poly(const PauliPolynomial& p) {
    if constexpr (std::is_same_v<Obs, PauliPolynomial>)
        return p;
    else
        return SortedPauliPoly::from_poly(p);
}

inline PauliPolynomial obs_to_poly(const PauliPolynomial& obs) { return obs; }
inline PauliPolynomial obs_to_poly(const SortedPauliPoly& obs) { return obs.to_poly(); }

// --- Sorted conjugation helpers ---

// Shared emit helper for sorted conjugation paths
inline void emit_kept_partners(const SortedPauliPoly& in, const PauliString& ps, ff_float cos_t,
                               ff_complex isin_t, int maxdegree, SortedPauliPoly& kept,
                               SortedPauliPoly& partners) {
    kept.keys.clear();
    kept.coeffs.clear();
    partners.keys.clear();
    partners.coeffs.clear();
    kept.keys.reserve(in.size());
    kept.coeffs.reserve(in.size());
    partners.keys.reserve(in.size());
    partners.coeffs.reserve(in.size());

    for (size_t i = 0; i < in.size(); i++) {
        const PauliString& q = in.keys[i];
        const ff_complex& c = in.coeffs[i];

        if (q.commutes(ps)) {
            kept.keys.push_back(q);
            kept.coeffs.push_back(c);
        } else {
            kept.keys.push_back(q);
            kept.coeffs.push_back(c * cos_t);

            PauliMonomial pq = ps * q;
            if (pq.degree_total() <= maxdegree) {
                partners.keys.push_back(pq.pauli_string());
                partners.coeffs.push_back(c * isin_t * pq.coefficient());
            }
        }
    }
}

// --- Per-gate conjugation: serial sorted (comparison sort) ---

inline void conjugate(SortedPauliPoly& obs, const ROT& gate, int maxdegree) {
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    SortedPauliPoly kept, partners;
    emit_kept_partners(obs, gate.ps, cos_t, isin_t, maxdegree, kept, partners);

    std::vector<size_t> idx(partners.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return partners.keys[a] < partners.keys[b]; });

    SortedPauliPoly sorted_partners;
    sorted_partners.keys.reserve(partners.size());
    sorted_partners.coeffs.reserve(partners.size());
    for (size_t i : idx) {
        sorted_partners.keys.push_back(partners.keys[i]);
        sorted_partners.coeffs.push_back(partners.coeffs[i]);
    }

    obs = SortedPauliPoly::merge(kept, sorted_partners);
}

// --- Per-gate conjugation: serial sorted (radix sort, caller supplies scratch) ---

inline void conjugate_radix(SortedPauliPoly& obs, const ROT& gate, int maxdegree,
                            std::vector<PauliString>& keys_buf,
                            std::vector<ff_complex>& coeffs_buf) {
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    SortedPauliPoly kept, partners;
    emit_kept_partners(obs, gate.ps, cos_t, isin_t, maxdegree, kept, partners);

    radix_sort_pauli(partners.keys, partners.coeffs, keys_buf, coeffs_buf);

    obs = SortedPauliPoly::merge(kept, partners);
}

#ifdef FF_OPENMP

// --- Per-gate conjugation: OMP sorted (parallel emit + sort + parallel merge) ---

inline void conjugate_omp(SortedPauliPoly& obs, const ROT& gate, int maxdegree, int n_threads) {
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));
    const int K = (int)obs.size();

    std::vector<SortedPauliPoly> kept_parts(n_threads);
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> partner_parts(n_threads);

    int chunk = (K + n_threads - 1) / n_threads;

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& my_kept = kept_parts[tid];
        auto& my_partners = partner_parts[tid];
        my_kept.keys.reserve(chunk);
        my_kept.coeffs.reserve(chunk);
        my_partners.reserve(chunk);

#pragma omp for schedule(static)
        for (int j = 0; j < K; j++) {
            const PauliString& q = obs.keys[j];
            const ff_complex& c = obs.coeffs[j];

            if (q.commutes(gate.ps)) {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c);
            } else {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c * cos_t);

                PauliMonomial pq = gate.ps * q;
                if (pq.degree_total() <= maxdegree)
                    my_partners.emplace_back(pq.pauli_string(), c * isin_t * pq.coefficient());
            }
        }
    }

    SortedPauliPoly kept;
    {
        size_t total = 0;
        for (const auto& part : kept_parts) total += part.size();
        kept.keys.reserve(total);
        kept.coeffs.reserve(total);
        for (const auto& part : kept_parts) {
            kept.keys.insert(kept.keys.end(), part.keys.begin(), part.keys.end());
            kept.coeffs.insert(kept.coeffs.end(), part.coeffs.begin(), part.coeffs.end());
        }
    }

    std::vector<std::pair<PauliString, ff_complex>> all_partners;
    {
        size_t total = 0;
        for (const auto& part : partner_parts) total += part.size();
        all_partners.reserve(total);
        for (auto& part : partner_parts)
            all_partners.insert(all_partners.end(), part.begin(), part.end());
    }

    std::sort(all_partners.begin(), all_partners.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    SortedPauliPoly partners;
    partners.keys.reserve(all_partners.size());
    partners.coeffs.reserve(all_partners.size());
    for (const auto& [k, c] : all_partners) {
        partners.keys.push_back(k);
        partners.coeffs.push_back(c);
    }

    obs = sorted_merge_omp(kept, partners, n_threads);
}

#endif  // FF_OPENMP

}  // namespace pauli_gates

}  // namespace fastfermion
