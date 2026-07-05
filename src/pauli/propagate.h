// Pauli propagation: truncated Heisenberg-picture evolution.
//
// Single entry point: propagate(). The `parallel` parameter selects the
// parallelisation strategy:
//
//   "serial"   — baseline, one thread
//   "omp"      — parallel emission, serial hash-map rebuild
//   "sharded"  — sharded hash-map, all-parallel merge
//   "gpu"      — CUDA sorted-array engine (built with -Dgpu=enabled; see GPU_PLAN.md)
//   "auto"     — serial when n_threads=1, sharded when n_threads>1 (never gpu)
//
// Gate batching is on by default.

#pragma once
#include <chrono>
#include <string>
#include <vector>

#include "common.h"
#include "pauli/gates.h"
#include "pauli/truncate.h"

#ifdef FF_GPU
#include <random>

#include "pauli/propagate_gpu.h"
#endif

#ifdef FF_OPENMP
#include <omp.h>
#endif

namespace fastfermion {
namespace pauli_gates {

// =========================================================================
// Per-phase wall-time profiling (opt-in; near-zero cost when disabled).
// Scalar accumulations happen in serial regions; the per-thread slots are
// written only by their owning thread. Both are race-free.
// =========================================================================
struct PropProfile {
    double snapshot = 0, emit = 0, merge = 0;
    long long n_gates = 0;
    bool enabled = false;
    // Sharded backend only: per-thread work time inside the emit/merge
    // regions (exposes load imbalance and barrier wait), and how many
    // emitted partners stayed in their shard vs crossed shards.
    std::vector<double> emit_thread, merge_thread;
    long long n_local = 0, n_remote = 0;
    void reset() {
        snapshot = emit = merge = 0;
        n_gates = 0;
        emit_thread.clear();
        merge_thread.clear();
        n_local = n_remote = 0;
    }
    void ensure_threads(int n) {
        if (static_cast<int>(emit_thread.size()) < n) {
            emit_thread.resize(n, 0.0);
            merge_thread.resize(n, 0.0);
        }
    }
};
inline PropProfile& prop_profile() {
    static PropProfile p;
    return p;
}
inline double prof_now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// =========================================================================
// Per-gate conjugation: serial
// =========================================================================

inline void conjugate(PauliPolynomial& obs, const ROT& gate, int maxdegree) {
    std::vector<std::pair<PauliString, ff_complex>> new_terms;
    new_terms.reserve(obs.terms.size());

    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    for (auto& [x, coeff] : obs.terms) {
        if (!x.commutes(ps)) {
            PauliMonomial partner = ps * x;
            if (partner.degree_total() <= maxdegree)
                new_terms.emplace_back(partner.pauli_string(),
                                       coeff * isin_t * partner.coefficient());
            coeff *= cos_t;
        }
    }
    for (const auto& [x, coeff] : new_terms) obs.terms[x] += coeff;
}

// =========================================================================
// Per-gate conjugation: OMP (parallel emission, serial rebuild)
// =========================================================================

#ifdef FF_OPENMP

inline void conjugate_omp(PauliPolynomial& obs, const ROT& gate, int maxdegree, int n_threads,
                          std::vector<std::pair<PauliString, ff_complex>>& snap) {
    const bool prof = prop_profile().enabled;
    double _t = prof ? prof_now() : 0.0;
    snap.clear();
    snap.insert(snap.end(), obs.terms.begin(), obs.terms.end());
    if (prof) prop_profile().snapshot += prof_now() - _t;

    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));

    int chunk = (snap.size() + n_threads - 1) / n_threads;
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_kept(n_threads);
    std::vector<std::vector<std::pair<PauliString, ff_complex>>> all_partners(n_threads);

    _t = prof ? prof_now() : 0.0;
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        all_kept[tid].reserve(chunk);
        all_partners[tid].reserve(chunk);

#pragma omp for schedule(static)
        for (int j = 0; j < (int)snap.size(); j++) {
            const auto& [ps_q, c] = snap[j];
            if (ps_q.commutes(ps))
                all_kept[tid].emplace_back(ps_q, c);
            else {
                all_kept[tid].emplace_back(ps_q, c * cos_t);
                PauliMonomial partner = ps * ps_q;
                if (partner.degree_total() <= maxdegree)
                    all_partners[tid].emplace_back(partner.pauli_string(),
                                                   c * isin_t * partner.coefficient());
            }
        }
    }
    if (prof) prop_profile().emit += prof_now() - _t;

    _t = prof ? prof_now() : 0.0;
    obs.terms.clear();
    for (int t = 0; t < n_threads; t++) {
        for (const auto& [x, c] : all_kept[t]) obs.terms[x] += c;
        for (const auto& [x, c] : all_partners[t]) obs.terms[x] += c;
    }
    if (prof) {
        prop_profile().merge += prof_now() - _t;
        prop_profile().n_gates++;
    }
}

// =========================================================================
// Per-gate conjugation: sharded hash-map (all-parallel merge)
//
// The polynomial is partitioned into n_threads shards by hashing the
// PauliString key. Each thread owns one shard. After emission, partners
// destined for other shards are exchanged via thread-local buffers.
// No serial phase: merging is embarrassingly parallel.
// =========================================================================

using ShardedPoly = std::vector<PauliPolynomialMap>;
using SendBuf = std::vector<std::pair<PauliString, ff_complex>>;

inline int shard_of(const PauliString& ps, int n_shards) {
    return static_cast<int>(ps.hash() % static_cast<std::uint64_t>(n_shards));
}

inline ShardedPoly to_sharded(const PauliPolynomial& poly, int n_shards) {
    ShardedPoly shards(n_shards);
    for (const auto& [ps, c] : poly.terms) shards[shard_of(ps, n_shards)][ps] += c;
    return shards;
}

inline PauliPolynomial from_sharded(const ShardedPoly& shards) {
    PauliPolynomial out;
    for (const auto& shard : shards)
        for (const auto& [ps, c] : shard) out.terms[ps] += c;
    return out;
}

// Persistent cross-shard routing buffers, allocated once and reused across gates.
// outgoing[src][dst] holds partners from thread src bound for thread dst; local[tid]
// holds partners that stay in shard tid. Reusing them (clear, retain capacity) removes
// the per-gate t*t reconstruction that capped sharded scaling above 32 threads.
struct ShardBuffers {
    std::vector<std::vector<SendBuf>> outgoing;
    std::vector<SendBuf> local;
    explicit ShardBuffers(int n_threads)
        : outgoing(n_threads, std::vector<SendBuf>(n_threads)), local(n_threads) {}
};

inline void conjugate_sharded(ShardedPoly& shards, const ROT& gate, int maxdegree, int n_threads,
                              ShardBuffers& buf) {
    const bool prof = prop_profile().enabled;
    const PauliString& ps = gate.ps;
    const ff_float cos_t = cos(gate.theta);
    const ff_complex isin_t = ff_complex(0, sin(gate.theta));
    if (prof) prop_profile().ensure_threads(n_threads);

    double _t = prof ? prof_now() : 0.0;
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        SendBuf& local_new = buf.local[tid];
        const double _tt = prof ? prof_now() : 0.0;

        // Reuse this thread's buffers (retain capacity); no per-gate reallocation.
        local_new.clear();
        for (auto& out : buf.outgoing[tid]) out.clear();

        for (auto& [x, coeff] : shard) {
            if (!x.commutes(ps)) {
                PauliMonomial partner = ps * x;
                if (partner.degree_total() <= maxdegree) {
                    auto pk = partner.pauli_string();
                    auto pc = coeff * isin_t * partner.coefficient();
                    int dest = shard_of(pk, n_threads);
                    if (dest == tid)
                        local_new.emplace_back(pk, pc);
                    else
                        buf.outgoing[tid][dest].emplace_back(pk, pc);
                }
                coeff *= cos_t;
            }
        }

        // Insert partners that stayed in this shard
        for (const auto& [k, c] : local_new) shard[k] += c;

        if (prof) {
            long long nr = 0;
            for (int d = 0; d < n_threads; d++)
                nr += static_cast<long long>(buf.outgoing[tid][d].size());
            prop_profile().emit_thread[tid] += prof_now() - _tt;
#pragma omp atomic
            prop_profile().n_local += static_cast<long long>(local_new.size());
#pragma omp atomic
            prop_profile().n_remote += nr;
        }
    }
    // Implicit barrier
    if (prof) prop_profile().emit += prof_now() - _t;

    // Each thread merges incoming partners from all other threads
    _t = prof ? prof_now() : 0.0;
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards[tid];
        const double _tt = prof ? prof_now() : 0.0;
        for (int src = 0; src < n_threads; src++) {
            for (const auto& [k, c] : buf.outgoing[src][tid]) shard[k] += c;
        }
        if (prof) prop_profile().merge_thread[tid] += prof_now() - _tt;
    }
    if (prof) {
        prop_profile().merge += prof_now() - _t;
        prop_profile().n_gates++;
    }
}

inline void truncate_sharded(ShardedPoly& shards, int n_threads, ff_float mincoeff, int topk,
                             int max_xweight, int xtrunc_period, int rot_count) {
    if (topk > 0) {
        // Top-K requires a global view — merge, truncate, re-shard
        PauliPolynomial merged = from_sharded(shards);
        truncate_all(merged, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
        shards = to_sharded(merged, n_threads);
        return;
    }
    // Per-shard truncation (threshold and x-weight are separable)
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        PauliPolynomial tmp;
        tmp.terms = std::move(shards[tid]);
        truncate_all(tmp, mincoeff, 0, max_xweight, xtrunc_period, rot_count);
        shards[tid] = std::move(tmp.terms);
    }
}

#endif  // FF_OPENMP

// =========================================================================
// Gate batching
// =========================================================================

inline std::vector<std::vector<ROT>> batch_commuting_gates(const std::vector<ROT>& gates) {
    std::vector<std::vector<ROT>> batches;
    std::vector<ROT> current;
    for (const auto& gate : gates) {
        bool ok = true;
        for (const auto& g : current)
            if (!gate.ps.commutes(g.ps)) {
                ok = false;
                break;
            }
        if (ok)
            current.push_back(gate);
        else {
            if (!current.empty()) batches.push_back(std::move(current));
            current = {gate};
        }
    }
    if (!current.empty()) batches.push_back(std::move(current));
    return batches;
}

#ifdef FF_GPU
// =========================================================================
// GPU path (CUDA sorted-array engine in propagate_gpu.cu)
//
// The host owns everything algebraic: key-width choice, flattening to raw
// words, Clifford segments, gate order/batching, and the truncation
// schedule. The engine owns only the device term array.
// =========================================================================

// Narrowest per-plane word count covering the observable and every gate;
// XOR closure keeps partners inside it.
inline int _gpu_key_words(const Circuit& circuit, const PauliPolynomial& a) {
    int extent = 1;
    for (const auto& [x, c] : a.terms) extent = MAX(extent, x.extent());
    for (const auto& g : circuit) {
        if (g.index() == 1) {
            extent = MAX(extent, std::get<ROT>(g).ps.extent());
        } else {
            std::visit(
                [&extent](const auto& v) {
                    extent = MAX(extent, v.i + 1);
                    if constexpr (requires { v.j; }) extent = MAX(extent, v.j + 1);
                },
                std::get<CliffordGate>(g));
        }
    }
    return (extent + WORD_LENGTH - 1) / WORD_LENGTH;
}

inline void _gpu_flatten_key(const PauliString& s, int words, std::uint64_t* out) {
    for (int i = 0; i < words; i++) out[i] = (i < SYS_NUM_ULONG) ? s.xory.words[i] : 0;
    for (int i = 0; i < words; i++) out[words + i] = (i < SYS_NUM_ULONG) ? s.yorz.words[i] : 0;
}

inline void _gpu_upload(gpu::Engine& eng, const PauliPolynomial& p, int words) {
    std::vector<std::uint64_t> keys(2 * std::size_t(words) * p.terms.size());
    std::vector<double> coeffs(p.terms.size());
    std::size_t i = 0;
    for (const auto& [x, c] : p.terms) {
        // ROT conjugation of a Hermitian observable keeps coefficients real;
        // the engine stores real doubles, so anything else is a hard error.
        if (std::abs(c.imag()) > 1e-12 * (1.0 + std::abs(c.real())))
            throw_error("gpu backend requires a real observable (imaginary coefficient found)");
        _gpu_flatten_key(x, words, &keys[2 * std::size_t(words) * i]);
        coeffs[i] = c.real();
        i++;
    }
    eng.upload(keys.data(), coeffs.data(), p.terms.size());
}

inline PauliPolynomial _gpu_download(gpu::Engine& eng, int words) {
    std::vector<std::uint64_t> keys;
    std::vector<double> coeffs;
    eng.download(keys, coeffs);
    PauliPolynomial out;
    out.terms.reserve(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); i++) {
        ff_ulong xory(0), yorz(0);
        for (int w = 0; w < MIN(words, SYS_NUM_ULONG); w++) {
            xory.words[w] = keys[2 * std::size_t(words) * i + w];
            yorz.words[w] = keys[2 * std::size_t(words) * i + words + w];
        }
        out.terms[PauliString(xory, yorz)] = coeffs[i];
    }
    return out;
}

inline PauliPolynomial propagate_gpu_path(const Circuit& circuit, const PauliPolynomial& a,
                                          int maxdegree, ff_float mincoeff, bool batched) {
    if (gpu::device_count() == 0) throw_error("gpu backend: no CUDA device available");
    const int words = _gpu_key_words(circuit, a);
    if (words > 2) throw_error("gpu backend supports at most 128 qubits");

    gpu::Engine eng(words, a.terms.size());
    _gpu_upload(eng, a, words);
    std::vector<std::uint64_t> pkey(2 * words);

    // Threshold cadence mirrors truncate_all: per gate unbatched, per
    // commuting batch batched. The pure weight cutoff (mincoeff = 0) needs no
    // scheduled compaction at all — the retained set is compaction-invariant,
    // so deduplication is deferred to the engine's budget (GPU_PLAN.md, D1).
    auto apply_rots = [&](const std::vector<ROT>& rots) {
        auto rot = [&](const ROT& g) {
            _gpu_flatten_key(g.ps, words, pkey.data());
            eng.apply_rot(pkey.data(), g.theta, maxdegree);
        };
        if (!batched) {
            for (const auto& g : rots) {
                rot(g);
                if (mincoeff > 0) eng.compact(mincoeff);
            }
        } else {
            for (const auto& batch : batch_commuting_gates(rots)) {
                for (const auto& g : batch) rot(g);
                if (mincoeff > 0) eng.compact(mincoeff);
            }
        }
    };

    std::vector<ROT> rot_buffer;
    auto flush = [&]() {
        if (rot_buffer.empty()) return;
        apply_rots(rot_buffer);
        rot_buffer.clear();
    };

    int clifford_begin = 0;
    bool pending_clifford = false;
    for (int i = circuit.size() - 1; i >= 0; i--) {
        if (circuit[i].index() == 0) {
            flush();
            if (!pending_clifford) {
                clifford_begin = i;
                pending_clifford = true;
            }
        } else if (circuit[i].index() == 1) {
            if (pending_clifford) {  // Clifford segments round-trip through the host (rare)
                PauliPolynomial obs = _gpu_download(eng, words);
                _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                _gpu_upload(eng, obs, words);
                pending_clifford = false;
            }
            rot_buffer.push_back(std::get<ROT>(circuit[i]));
        }
    }
    flush();
    if (pending_clifford) {
        PauliPolynomial obs = _gpu_download(eng, words);
        _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        return obs;
    }
    return _gpu_download(eng, words);
}

// Random-pair check of the device phase rule against the host oracle
// (pauli_string_multiply) — the one identity the engine duplicates.
inline bool gpu_phase_selftest(int n_pairs, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int words = SYS_NUM_ULONG;
    std::vector<std::uint64_t> a(2 * std::size_t(words) * n_pairs);
    std::vector<std::uint64_t> b(2 * std::size_t(words) * n_pairs);
    std::vector<int> expected(n_pairs);
    for (int i = 0; i < n_pairs; i++) {
        ff_ulong ax(0), az(0), bx(0), bz(0);
        for (int w = 0; w < words; w++) {
            ax.words[w] = rng(), az.words[w] = rng();
            bx.words[w] = rng(), bz.words[w] = rng();
        }
        PauliString p(ax, az), q(bx, bz);
        _gpu_flatten_key(p, words, &a[2 * std::size_t(words) * i]);
        _gpu_flatten_key(q, words, &b[2 * std::size_t(words) * i]);
        int jpow = 0;
        pauli_string_multiply(p, jpow, q);
        expected[i] = ((jpow % 4) + 4) % 4;
    }
    return gpu::phase_check(a.data(), b.data(), expected.data(), n_pairs, words);
}
#endif  // FF_GPU

// =========================================================================
// Public API
// =========================================================================

inline PauliPolynomial propagate(const Circuit& circuit, const PauliPolynomial& a,
                                 int n_threads = 1, int maxdegree = 128, ff_float mincoeff = 0,
                                 int topk = 0, int max_xweight = -1, int xtrunc_period = 1,
                                 bool batched = true, const std::string& parallel = "auto") {
#ifndef FF_OPENMP
    n_threads = 1;
#endif

    // Resolve "auto": serial when single-threaded, sharded otherwise
    std::string strategy = parallel;
    if (strategy == "auto") strategy = (n_threads > 1) ? "sharded" : "serial";
    if (strategy == "gpu") {
#ifdef FF_GPU
        if (topk > 0 || max_xweight >= 0)
            throw_error("gpu backend does not support topk or max_xweight yet");
        return propagate_gpu_path(circuit, a, maxdegree, mincoeff, batched);
#else
        throw_error("fastfermion was built without GPU support (rebuild with -Dgpu=enabled)");
#endif
    }
    if (n_threads <= 1) strategy = "serial";

    PauliPolynomial obs(a);
    int clifford_begin;
    bool pending_clifford = false;
    int rot_count = 0;

    // -----------------------------------------------------------------
    // Serial path
    // -----------------------------------------------------------------
    if (strategy == "serial") {
        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    conjugate(obs, std::get<ROT>(circuit[i]), maxdegree);
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch) conjugate(obs, gate, maxdegree);
                    rot_count += batch.size();
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }
        if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        return obs;
    }

#ifdef FF_OPENMP
    // -----------------------------------------------------------------
    // OMP path (parallel emission, serial rebuild)
    // -----------------------------------------------------------------
    if (strategy == "omp") {
        std::vector<std::pair<PauliString, ff_complex>> snap;
        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    conjugate_omp(obs, std::get<ROT>(circuit[i]), maxdegree, n_threads, snap);
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch)
                        conjugate_omp(obs, gate, maxdegree, n_threads, snap);
                    rot_count += batch.size();
                    truncate_all(obs, mincoeff, topk, max_xweight, xtrunc_period, rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }
        if (pending_clifford) _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
        return obs;
    }

    // -----------------------------------------------------------------
    // Sharded path (all-parallel merge)
    // -----------------------------------------------------------------
    if (strategy == "sharded") {
        ShardedPoly shards = to_sharded(obs, n_threads);
        ShardBuffers buf(n_threads);  // persistent routing buffers, reused every gate

        if (!batched) {
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        obs = from_sharded(shards);
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        shards = to_sharded(obs, n_threads);
                        pending_clifford = false;
                    }
                    conjugate_sharded(shards, std::get<ROT>(circuit[i]), maxdegree, n_threads, buf);
                    truncate_sharded(shards, n_threads, mincoeff, topk, max_xweight, xtrunc_period,
                                     rot_count);
                    rot_count++;
                }
            }
        } else {
            std::vector<ROT> rot_buffer;
            auto flush = [&]() {
                if (rot_buffer.empty()) return;
                for (const auto& batch : batch_commuting_gates(rot_buffer)) {
                    for (const auto& gate : batch)
                        conjugate_sharded(shards, gate, maxdegree, n_threads, buf);
                    rot_count += batch.size();
                    truncate_sharded(shards, n_threads, mincoeff, topk, max_xweight, xtrunc_period,
                                     rot_count - 1);
                }
                rot_buffer.clear();
            };
            for (int i = circuit.size() - 1; i >= 0; i--) {
                if (circuit[i].index() == 0) {
                    flush();
                    if (!pending_clifford) {
                        clifford_begin = i;
                        pending_clifford = true;
                    }
                } else if (circuit[i].index() == 1) {
                    if (pending_clifford) {
                        obs = from_sharded(shards);
                        _apply_clifford_circuit(obs, circuit, i + 1, clifford_begin + 1);
                        shards = to_sharded(obs, n_threads);
                        pending_clifford = false;
                    }
                    rot_buffer.push_back(std::get<ROT>(circuit[i]));
                }
            }
            flush();
        }

        if (pending_clifford) {
            obs = from_sharded(shards);
            _apply_clifford_circuit(obs, circuit, 0, clifford_begin + 1);
            return obs;
        }
        return from_sharded(shards);
    }
#endif  // FF_OPENMP

    return obs;
}

inline PauliPolynomial propagate(const Circuit& circuit, const PauliString& a,
                                 int maxdegree = 128) {
    return propagate(circuit, PauliPolynomial(a), 1, maxdegree);
}

}  // namespace pauli_gates
}  // namespace fastfermion
