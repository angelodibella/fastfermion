/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "truncate.h"

#ifdef FF_OPENMP
#include <omp.h>
#endif

#ifdef FF_GPU
#include "gpu/engine.h"
#endif

namespace fastfermion {

// Conjugation of a term map by a rotation gate e^{-i theta/2 A}: a term x anticommuting with
// the generator A becomes cos(theta) x + (i sin(theta)) A x, where the new term A x is kept if
// the truncation rules admit it. rot supplies the generator and the two constants (see Rotation
// in pauli/propagate.h and majorana/propagate.h).
template <class Map, class Rotation, class Rules>
void conjugate(Map& terms, const Rotation& rot, const Rules& rules) {
    std::vector<std::pair<typename Map::key_type, ff_complex>> partners;
    partners.reserve(terms.size());
    for (auto& [x, c] : terms) {
        if (x.commutes(rot.axis)) continue;
        const auto ax = rot.axis * x;  // monomial: the basis element ax.s with phase ax.coeff
        if (rules.admits(ax.s)) partners.emplace_back(ax.s, c * rot.isin_t * ax.coeff);
        c *= rot.cos_t;
    }
    for (const auto& [x, c] : partners) terms[x] += c;
}

// A backend holds the terms of the observable between gates, and applies gates and the truncation
// rules due after a window to them. Backends implement conjugate(rotation), truncate(first, last),
// take() and load(polynomial). This one is the single-threaded reference.
template <class Poly, class Rules>
struct SerialBackend {
    Poly obs;
    const Rules& rules;
    SerialBackend(const Poly& a, const Rules& rules) : obs(a), rules(rules) {}
    template <class Rotation>
    void conjugate(const Rotation& rot) { fastfermion::conjugate(obs.terms, rot, rules); }
    void truncate(int first, int last) { truncate_window(obs.terms, rules, first, last); }
    Poly take() { return std::move(obs); }
    void load(Poly&& a) { obs = std::move(a); }
};

#ifdef FF_OPENMP

// Sharded hash map: the terms are partitioned into n_threads maps by the hash of their basis
// element, one per thread. A thread rotates its own shard, adds the new terms that hash to its
// shard and sends the others to their shard through per-pair buffers, which are merged in a
// second parallel pass, so that no phase is single-threaded. Buffers persist across gates.
template <class Poly, class Rules>
struct ShardedBackend {
    using Map = decltype(Poly::terms);
    using Key = typename Map::key_type;
    using Term = std::pair<Key, ff_complex>;
    struct alignas(64) Buffers {  // one per thread, on its own cache lines
        std::vector<Term> local;
        std::vector<std::vector<Term>> outgoing;  // outgoing[to]
    };
    const Rules& rules;
    int n_threads;
    std::vector<Map> shards;
    std::vector<Buffers> buffers;
    ShardedBackend(const Poly& a, const Rules& rules, int n_threads)
        : rules(rules), n_threads(n_threads), buffers(n_threads) {
        for (Buffers& b : buffers) b.outgoing.resize(n_threads);
        shard(a);
    }

    int shard_of(const Key& x) const { return int(x.hash() % std::uint64_t(n_threads)); }

    template <class Rotation>
    void conjugate(const Rotation& rot) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_threads; t++) {
            std::vector<Term>& mine = buffers[t].local;
            mine.clear();
            for (std::vector<Term>& out : buffers[t].outgoing) out.clear();
            for (auto& [x, c] : shards[t]) {
                if (x.commutes(rot.axis)) continue;
                const auto ax = rot.axis * x;
                if (rules.admits(ax.s)) {
                    const int to = shard_of(ax.s);
                    (to == t ? mine : buffers[t].outgoing[to]).emplace_back(ax.s, c * rot.isin_t * ax.coeff);
                }
                c *= rot.cos_t;
            }
            for (const auto& [x, c] : mine) shards[t][x] += c;
        }
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_threads; t++) {
            for (int from = 0; from < n_threads; from++) {
                for (const auto& [x, c] : buffers[from].outgoing[t]) shards[t][x] += c;
            }
        }
    }

    void truncate(int first, int last) {
        if (rules.topk > 0) {  // top-k compares all the terms: merge, truncate, shard again
            Poly obs = take();
            truncate_window(obs.terms, rules, first, last);
            shard(obs);
            return;
        }
        std::size_t n_terms = 0;
        double discarded = 0;
#pragma omp parallel for schedule(static) num_threads(n_threads) reduction(+ : n_terms, discarded)
        for (int t = 0; t < n_threads; t++) {
            n_terms += shards[t].size();
            discarded += truncate_terms(shards[t], rules, first, last);
        }
        rules.record(first, last, n_terms, discarded);
    }

    // Merges the shards into one polynomial, emptying them
    Poly take() {
        Poly obs;
        for (Map& shard : shards) {
            for (const auto& [x, c] : shard) obs.terms[x] += c;
            shard = Map();
        }
        return obs;
    }
    void load(Poly&& a) { shard(a); }
    void shard(const Poly& a) {
        shards.assign(n_threads, Map());
        for (const auto& [x, c] : a.terms) shards[shard_of(x)][x] += c;
    }
};

#endif  // FF_OPENMP

enum class Backend { serial, sharded, gpu };

// The backend named by parallel: "serial", "sharded", "gpu", or "auto" for sharded when
// n_threads > 1 and serial otherwise. Without OpenMP, or with n_threads <= 1, the CPU backend is
// serial.
inline Backend select_backend(const std::string& parallel, int n_threads) {
    if (n_threads < 1) throw_error("n_threads must be >= 1");
    Backend backend;
    if (parallel == "gpu") {
#ifdef FF_GPU
        if (gpu::device_count() == 0) throw_error("No CUDA device available");
        return Backend::gpu;
#else
        throw_error("fastfermion was built without GPU support (meson option -Dgpu=enabled)");
#endif
    } else if (parallel == "auto") {
        backend = n_threads > 1 ? Backend::sharded : Backend::serial;
    } else if (parallel == "serial") {
        backend = Backend::serial;
    } else if (parallel == "sharded") {
        backend = Backend::sharded;
    } else {
        throw_error("Unknown parallel strategy \"" << parallel << "\" (valid: auto, serial, sharded, gpu)");
    }
#ifndef FF_OPENMP
    n_threads = 1;
#endif
    return n_threads > 1 ? backend : Backend::serial;
}

}  // namespace fastfermion
