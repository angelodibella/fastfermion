// Sorted-array Pauli propagation.
// Two-pointer merge over sorted arrays instead of hash-map insertion.

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "common.h"
#include "pauli_algebra.h"
#include "pauli_propagation.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {

// ---------------------------------------------------------------------------
// LSD radix sort for PauliString keys + associated coefficients.
//
// PauliString comparison is lexicographic on the tuple
//   (support, yorz, xory)  where  support = xory | yorz,
// and each component is a BitSet<SYS_NUM_ULONG> (array of uint64_t words,
// compared MSW-first).  The sort key is therefore a sequence of
// 3 * SYS_NUM_ULONG uint64_t words, ordered from most significant to least:
//   support[N-1], ..., support[0], yorz[N-1], ..., yorz[0], xory[N-1], ..., xory[0].
//
// LSD radix sort processes bytes from least significant to most significant.
// Each pass is a stable counting sort on one byte (256 buckets).
// ---------------------------------------------------------------------------

namespace detail {

constexpr int RADIX = 256;
constexpr int N_WORDS = SYS_NUM_ULONG;
// 3 components (support, yorz, xory) × N_WORDS words × 8 bytes/word
constexpr int SORT_KEY_BYTES = 3 * N_WORDS * 8;

// Extract the sort key word at position `word_idx` (0 = least significant).
// Word layout (LSW to MSW):
//   0..N-1     : xory.words[0..N-1]
//   N..2N-1    : yorz.words[0..N-1]
//   2N..3N-1   : support.words[0..N-1]
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

// Sort parallel (keys, coeffs) arrays in-place using LSD radix sort,
// producing the same ordering as std::sort with PauliString::operator<.
// Requires scratch buffers of the same size; caller may supply them
// to amortise allocation across gates.
inline void radix_sort_pauli(std::vector<PauliString>& keys,
                             std::vector<ff_complex>& coeffs,
                             std::vector<PauliString>& keys_buf,
                             std::vector<ff_complex>& coeffs_buf) {
    const size_t n = keys.size();
    if (n <= 1) return;

    keys_buf.resize(n);
    coeffs_buf.resize(n);

    PauliString* src_k = keys.data();
    ff_complex*  src_c = coeffs.data();
    PauliString* dst_k = keys_buf.data();
    ff_complex*  dst_c = coeffs_buf.data();

    int passes = 0;
    for (int b = 0; b < detail::SORT_KEY_BYTES; b++) {
        if (detail::byte_is_zero(src_k, n, b)) continue;

        size_t counts[detail::RADIX] = {};
        for (size_t i = 0; i < n; i++)
            counts[detail::sort_key_byte(src_k[i], b)]++;

        size_t offsets[detail::RADIX];
        offsets[0] = 0;
        for (int r = 1; r < detail::RADIX; r++)
            offsets[r] = offsets[r - 1] + counts[r - 1];

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
inline void radix_sort_pauli(std::vector<PauliString>& keys,
                             std::vector<ff_complex>& coeffs) {
    std::vector<PauliString> kb;
    std::vector<ff_complex> cb;
    radix_sort_pauli(keys, coeffs, kb, cb);
}

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

// Emit kept (sorted) and partner (unsorted) streams from a single gate.
// Factored out so both comparison-sort and radix-sort paths can share it.
inline void emit_kept_partners(const SortedPauliPoly& in, const PauliString& ps,
                               ff_float cos_t, ff_complex isin_t, int maxdegree,
                               SortedPauliPoly& kept, SortedPauliPoly& partners) {
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

// Conjugate sorted polynomial by exp(-i t/2 P) using comparison sort.
SortedPauliPoly sorted_conjugate(const SortedPauliPoly& in, const PauliString& ps, ff_float theta,
                                 int maxdegree) {
    const ff_float cos_t = cos(theta);
    const ff_complex isin_t = ff_complex(0, sin(theta));

    SortedPauliPoly kept, partners;
    emit_kept_partners(in, ps, cos_t, isin_t, maxdegree, kept, partners);

    // Sort partners by key (comparison sort: O(K_a log K_a))
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

    return SortedPauliPoly::merge(kept, sorted_partners);
}

// Conjugate sorted polynomial by exp(-i t/2 P) using radix sort.
SortedPauliPoly sorted_conjugate_radix(const SortedPauliPoly& in, const PauliString& ps,
                                       ff_float theta, int maxdegree,
                                       std::vector<PauliString>& keys_buf,
                                       std::vector<ff_complex>& coeffs_buf) {
    const ff_float cos_t = cos(theta);
    const ff_complex isin_t = ff_complex(0, sin(theta));

    SortedPauliPoly kept, partners;
    emit_kept_partners(in, ps, cos_t, isin_t, maxdegree, kept, partners);

    // Sort partners by key (radix sort: O(K_a))
    radix_sort_pauli(partners.keys, partners.coeffs, keys_buf, coeffs_buf);

    return SortedPauliPoly::merge(kept, partners);
}

#ifdef FF_OPENMP

// Parallel merge: split both arrays into aligned chunks via binary search,
// merge each chunk independently.
SortedPauliPoly sorted_merge_omp(const SortedPauliPoly& a, const SortedPauliPoly& b,
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

// Parallel conjugation: parallel emit + sort + parallel merge.
SortedPauliPoly sorted_conjugate_omp(const SortedPauliPoly& in, const PauliString& ps,
                                     ff_float theta, int maxdegree, int n_threads) {
    const ff_float cos_t = cos(theta);
    const ff_complex isin_t = ff_complex(0, sin(theta));
    const int K = (int)in.size();

    // Parallel emit into thread-local buffers
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
            const PauliString& q = in.keys[j];
            const ff_complex& c = in.coeffs[j];

            if (q.commutes(ps)) {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c);
            } else {
                my_kept.keys.push_back(q);
                my_kept.coeffs.push_back(c * cos_t);

                PauliMonomial pq = ps * q;
                if (pq.degree_total() <= maxdegree)
                    my_partners.emplace_back(pq.pauli_string(), c * isin_t * pq.coefficient());
            }
        }
    }

    // Concatenate kept parts (each is sorted since input was sorted and
    // static scheduling preserves order within each chunk)
    SortedPauliPoly kept;
    {
        size_t total_kept = 0;
        for (const auto& part : kept_parts) total_kept += part.size();
        kept.keys.reserve(total_kept);
        kept.coeffs.reserve(total_kept);
        for (const auto& part : kept_parts) {
            kept.keys.insert(kept.keys.end(), part.keys.begin(), part.keys.end());
            kept.coeffs.insert(kept.coeffs.end(), part.coeffs.begin(), part.coeffs.end());
        }
    }

    // Concatenate partner pairs, then sort
    std::vector<std::pair<PauliString, ff_complex>> all_partners;
    {
        size_t total_partners = 0;
        for (const auto& part : partner_parts) total_partners += part.size();
        all_partners.reserve(total_partners);
        for (auto& part : partner_parts)
            all_partners.insert(all_partners.end(), part.begin(), part.end());
    }

    // Sort partners
    std::sort(all_partners.begin(), all_partners.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    SortedPauliPoly partners;
    partners.keys.reserve(all_partners.size());
    partners.coeffs.reserve(all_partners.size());
    for (const auto& [k, c] : all_partners) {
        partners.keys.push_back(k);
        partners.coeffs.push_back(c);
    }

    return sorted_merge_omp(kept, partners, n_threads);
}

#endif  // FF_OPENMP

// Top-K truncation: keep the K largest |coefficient| entries, preserving sort order.
SortedPauliPoly sorted_truncate_top_k(const SortedPauliPoly& in, int k) {
    if (k <= 0 || (int)in.size() <= k) return in;

    // Build index array, partial-sort by descending magnitude
    std::vector<size_t> idx(in.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](size_t a, size_t b) {
        return std::abs(in.coeffs[a]) > std::abs(in.coeffs[b]);
    });

    // Take the top-K indices, sort them to preserve key order
    idx.resize(k);
    std::sort(idx.begin(), idx.end());

    SortedPauliPoly result;
    result.keys.reserve(k);
    result.coeffs.reserve(k);
    for (size_t i : idx) {
        result.keys.push_back(in.keys[i]);
        result.coeffs.push_back(in.coeffs[i]);
    }
    return result;
}

// X-truncation (xSPD): discard strings with more than max_xweight X/Y factors.
SortedPauliPoly sorted_truncate_x_weight(const SortedPauliPoly& in, int max_xweight) {
    if (max_xweight < 0) return in;
    SortedPauliPoly result;
    result.keys.reserve(in.size());
    result.coeffs.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        if (in.keys[i].degree_x() + in.keys[i].degree_y() <= max_xweight) {
            result.keys.push_back(in.keys[i]);
            result.coeffs.push_back(in.coeffs[i]);
        }
    }
    return result;
}

// Threshold truncation: remove entries with |coeff| <= threshold.
SortedPauliPoly sorted_truncate(const SortedPauliPoly& in, ff_float mincoeff) {
    SortedPauliPoly result;
    result.keys.reserve(in.size());
    result.coeffs.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        if (std::abs(in.coeffs[i]) > mincoeff) {
            result.keys.push_back(in.keys[i]);
            result.coeffs.push_back(in.coeffs[i]);
        }
    }
    return result;
}

namespace pauli_gates {

// Serial propagation using sorted arrays with radix sort.
PauliPolynomial propagate_sorted_radix(const Circuit& circuit, const PauliPolynomial& a,
                                       int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                       int max_xweight = -1, int xtrunc_period = 1) {
    SortedPauliPoly obs = SortedPauliPoly::from_poly(a);

    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
    PauliPolynomial obs_hm;

    // Scratch buffers reused across gates
    std::vector<PauliString> keys_buf;
    std::vector<ff_complex> coeffs_buf;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                obs_hm = obs.to_poly();
                _apply_clifford_circuit(obs_hm, circuit, i + 1, clifford_begin + 1);
                obs = SortedPauliPoly::from_poly(obs_hm);
                pending_clifford = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);
            obs = sorted_conjugate_radix(obs, gate.ps, gate.theta, maxdegree, keys_buf, coeffs_buf);

            if (mincoeff > 0) obs = sorted_truncate(obs, mincoeff);
            if (topk > 0) obs = sorted_truncate_top_k(obs, topk);
            rot_count++;
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                obs = sorted_truncate_x_weight(obs, max_xweight);
        }
    }
    if (pending_clifford) {
        obs_hm = obs.to_poly();
        _apply_clifford_circuit(obs_hm, circuit, 0, clifford_begin + 1);
        return obs_hm;
    }
    return obs.to_poly();
}

// Serial propagation using sorted arrays (comparison sort).
PauliPolynomial propagate_sorted(const Circuit& circuit, const PauliPolynomial& a,
                                 int maxdegree = 128, ff_float mincoeff = 0, int topk = 0,
                                 int max_xweight = -1, int xtrunc_period = 1) {
    SortedPauliPoly obs = SortedPauliPoly::from_poly(a);

    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
    PauliPolynomial obs_hm;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                obs_hm = obs.to_poly();
                _apply_clifford_circuit(obs_hm, circuit, i + 1, clifford_begin + 1);
                obs = SortedPauliPoly::from_poly(obs_hm);
                pending_clifford = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);
            obs = sorted_conjugate(obs, gate.ps, gate.theta, maxdegree);

            if (mincoeff > 0) obs = sorted_truncate(obs, mincoeff);
            if (topk > 0) obs = sorted_truncate_top_k(obs, topk);
            rot_count++;
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                obs = sorted_truncate_x_weight(obs, max_xweight);
        }
    }
    if (pending_clifford) {
        obs_hm = obs.to_poly();
        _apply_clifford_circuit(obs_hm, circuit, 0, clifford_begin + 1);
        return obs_hm;
    }
    return obs.to_poly();
}

// Parallel propagation using sorted arrays.
// Falls back to serial when n_threads <= 1 or FF_OPENMP is not defined.
PauliPolynomial propagate_sorted_omp(const Circuit& circuit, const PauliPolynomial& a,
                                     int n_threads = 1, int maxdegree = 128, ff_float mincoeff = 0,
                                     int topk = 0, int max_xweight = -1, int xtrunc_period = 1) {
    if (n_threads <= 1)
        return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);

#ifdef FF_OPENMP
    SortedPauliPoly obs = SortedPauliPoly::from_poly(a);

    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;
    PauliPolynomial obs_hm;

    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {
                obs_hm = obs.to_poly();
                _apply_clifford_circuit(obs_hm, circuit, i + 1, clifford_begin + 1);
                obs = SortedPauliPoly::from_poly(obs_hm);
                pending_clifford = false;
            }

            const ROT& gate = std::get<ROT>(circuit[i]);
            obs = sorted_conjugate_omp(obs, gate.ps, gate.theta, maxdegree, n_threads);

            if (mincoeff > 0) obs = sorted_truncate(obs, mincoeff);
            if (topk > 0) obs = sorted_truncate_top_k(obs, topk);
            rot_count++;
            if (max_xweight >= 0 && xtrunc_period > 0 && rot_count % xtrunc_period == 0)
                obs = sorted_truncate_x_weight(obs, max_xweight);
        }
    }
    if (pending_clifford) {
        obs_hm = obs.to_poly();
        _apply_clifford_circuit(obs_hm, circuit, 0, clifford_begin + 1);
        return obs_hm;
    }
    return obs.to_poly();
#else
    return propagate_sorted(circuit, a, maxdegree, mincoeff, topk, max_xweight, xtrunc_period);
#endif
}

}  // namespace pauli_gates
}  // namespace fastfermion
