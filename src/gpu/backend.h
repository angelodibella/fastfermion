/*
    Copyright (c) 2025-2026 Hamza Fawzi (hamzafawzi@gmail.com)
    All rights reserved. Use of this source code is governed
    by a license that can be found in the LICENSE file.
*/

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "truncate.h"

#ifdef FF_GPU
#include "gpu/engine.h"
#endif

namespace fastfermion {

// Options of the GPU backend (see gpu/engine.h)
struct GpuOptions {
    // The buffers hold twice this many terms (the terms plus one gate's worth of new ones): < 0
    // allocates them once for the a priori bound on the number of terms when that fits in the
    // device memory, 0 grows them on demand, > 0 allocates as requested and fails if that does not fit
    long long reserve_terms = -1;
    // "dense", "support", or "auto" for the support-list key whenever it is valid and smaller
    std::string key = "auto";
    // Deduplicate the new terms once they exceed beta times the deduplicated ones (0: after every
    // gate, < 0: only when the buffers are full); this does not change the results
    double beta = 0.35;
};

#ifdef FF_GPU

// Backend keeping the terms on the device (see gpu/engine.h). Codec converts basis elements to and
// from the flat keys of the engine, and coefficients to and from the real coefficients of the
// Hermitian basis elements; it also picks the key format and the preallocation (see GpuCodec in
// pauli/propagate.h and majorana/propagate.h).
template <class Poly, class Rules, class Codec>
struct GpuBackend {
    using Key = typename decltype(Poly::terms)::key_type;
    const Rules& rules;
    Codec codec;
    gpu::Engine engine;
    std::vector<std::uint64_t> gate_key;

    GpuBackend(const Poly& a, const Rules& rules, const Codec& codec)
        : rules(rules), codec(codec),
          engine(codec.words, a.terms.size(), codec.reserve, codec.reserve_hard, codec.key_format, codec.beta),
          gate_key(2 * codec.words) {
        upload(a);
    }

    template <class Rotation>
    void conjugate(const Rotation& rot) {
        codec.flatten(rot.axis, gate_key.data());
        engine.apply_rot(gate_key.data(), rot.theta, rules.emission_degree());
    }

    // The rules due after the window fire in one compaction (the engine deduplicates on its own
    // schedule otherwise, which does not change the results)
    void truncate(int first, int last) {
        const bool degree = rules.degree_due(first, last);
        const bool threshold = rules.threshold_due(first, last);
        const std::size_t n_terms = engine.size();
        double discarded = 0;
        if (degree || threshold) {
            discarded = engine.compact(threshold ? rules.mincoeff : 0, degree ? rules.maxdegree : INT_MAX);
        }
        rules.record(first, last, n_terms, discarded);
    }

    Poly take() {
        std::vector<std::uint64_t> keys;
        std::vector<double> coeffs;
        engine.download(keys, coeffs);
        Poly obs;
        obs.terms.reserve(coeffs.size());
        for (std::size_t i = 0; i < coeffs.size(); i++) {
            const Key x = codec.unflatten(&keys[2 * std::size_t(codec.words) * i]);
            obs.terms[x] = codec.coeff_from_device(x, coeffs[i]);
        }
        TruncStats& stats = trunc_stats();
        stats.peak_device_bytes = std::max(stats.peak_device_bytes, engine.peak_device_bytes());
        return obs;
    }

    void load(Poly&& a) { upload(a); }

    void upload(const Poly& a) {
        std::vector<std::uint64_t> keys(2 * std::size_t(codec.words) * a.terms.size());
        std::vector<double> coeffs(a.terms.size());
        std::size_t i = 0;
        for (const auto& [x, c] : a.terms) {
            codec.flatten(x, &keys[2 * std::size_t(codec.words) * i]);
            coeffs[i++] = codec.coeff_to_device(x, c);
        }
        engine.upload(keys.data(), coeffs.data(), a.terms.size());
    }
};

#endif  // FF_GPU

}  // namespace fastfermion
