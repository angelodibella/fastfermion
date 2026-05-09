# fastfermion development log

## 2026-05-08: OpenMP term-parallel emission

Added `propagate_omp` in `src/pauli_propagation_omp.h`. The propagation loop for each Pauli rotation gate has three phases:

1. **Snapshot:** copy the hash map into a flat vector (needed because hash maps aren't indexable and can't be safely read by multiple threads while being modified).
2. **Parallel emit:** each thread processes a chunk of the snapshot independently. For each Pauli string Q, check if it commutes with the gate axis P. If so, keep it unchanged. If not, scale its coefficient by cos(theta) and compute the partner string P*Q with coefficient i*sin(theta). Results go to thread-local buffers — no synchronisation needed.
3. **Serial rebuild:** clear the hash map and reinsert all results from all thread-local buffers.

**Result:** correctness verified (17 tests, serial/OMP agreement to 1e-12 on TFIM, Heisenberg, mixed Clifford+ROT circuits, edge cases). However, the OMP version is *slower* than serial at current problem sizes (K_m ~ 10^5). The bottleneck is phase 3: reinserting ~2*K_m entries into the hash map involves random memory access (cache misses), which dominates the time saved by parallel emission.

**Conclusion:** term-parallel emission alone is not sufficient. The hash-map rebuild must be replaced.

## 2026-05-09: Sorted-array merge path (design)

To eliminate the hash-map bottleneck, replace it with a sorted-array representation during propagation. The polynomial is stored as two parallel arrays `keys[]` and `coeffs[]`, sorted lexicographically by PauliString. All operations become sequential scans over contiguous memory.

**Conjugation algorithm for one gate (axis P, angle theta):**

1. **Classify and emit.** Single pass over the sorted input. Each entry (Q, c) produces:
   - A `kept` entry: (Q, c) if commuting, or (Q, c*cos_theta) if anticommuting.
   - A `partner` entry (anticommuting only): (P*Q, c * i*sin_theta * phase).
   
   The `kept` stream is automatically sorted (same keys as input, unchanged). The `partners` stream is NOT sorted (XOR does not preserve lexicographic order in general).

2. **Sort partners.** O(K_a log K_a) where K_a <= K_m is the number of anticommuting entries. Parallelisable.

3. **Merge.** Two-pointer merge of the sorted `kept` and sorted `partners` streams into the output. Sum coefficients on key collision. O(K_m), parallelisable by splitting into chunks via binary search.

4. **Truncate.** Single pass, remove |c| < tau. Already sorted, stays sorted.

**Why this is faster:** every step is a sequential scan over contiguous memory. No hash lookups, no random access. The dominant cost is the sort in step 2, which at K_m = 10^6 takes ~0.05s (vs ~0.1s for the hash-map rebuild, which also cannot be parallelised).

**Implementation plan:** define `SortedPauliPoly` (parallel sorted arrays), implement `from_poly`/`to_poly` (convert to/from hash map), `sorted_merge` (two-pointer merge), `sorted_conjugate` (steps 1-3 above). Serial first, then add OpenMP to emit and merge.

## 2026-05-09: Sorted-array implementation (serial + parallel)

Implemented in `src/pauli_sorted.h`:

- `SortedPauliPoly`: parallel sorted arrays (`keys[]`, `coeffs[]`) with `from_poly`/`to_poly` conversion and a static `merge` (two-pointer sweep summing on key collision).
- `sorted_conjugate`: serial emit → sort partners → merge. The kept stream preserves input order; the partner stream (XORed keys) is sorted separately then merged.
- `sorted_conjugate_omp`: parallel emit (thread-local buffers, same pattern as `propagate_omp`), serial sort (TODO: parallel sort), parallel merge via `sorted_merge_omp` (splits both arrays into aligned chunks via binary search, merges each chunk on a separate thread, concatenates).
- `sorted_truncate`: single-pass threshold filter preserving sorted order.
- `propagate_sorted` / `propagate_sorted_omp`: full propagation loops using the sorted path, with Clifford batching fallback to hash map. Exposed to Python.

**Tests:** 20 tests (10 serial sorted + 10 parallel sorted), all passing. Total: 71 ff tests + 42 majorana = 113.

**Current state:** four propagation backends available (`propagate`, `propagate_omp`, `propagate_sorted`, `propagate_sorted_omp`). Next step: CPU benchmarking to determine crossover points and whether the sorted parallel path actually outperforms serial at realistic K_m.

## 2026-05-09: Top-K truncation, X-truncation, gate batching

Added three features to all backends:

- **Top-K truncation** (`truncate_top_k`): retain the K largest-magnitude terms. Uses `std::nth_element` (O(K) average). Gives a fixed memory budget, important for GPU stability. Source: Shao, Cheng & Liu 2025.
- **X-truncation** (`truncate_x_weight`): discard strings with X-weight exceeding a threshold. Applied every `xtrunc_period` gates. Source: Begušić & Chan 2025.
- **Gate batching** (`propagate_batched` in `pauli_batched.h`): partition consecutive ROT gates into maximal batches of mutually commuting gates, apply each batch in a single pass. Reduces the number of emit-merge-truncate cycles. Truncation is once per batch (not per gate), so results differ from per-gate truncation by O(tau).

All five backends now support: `mincoeff`, `maxdegree`, `topk`, `max_xweight`, `xtrunc_period`.

**Tests:** 87 ff tests + 42 majorana = 129 passing.

## 2026-05-09: Per-gate cost analysis and sorted-path bottleneck

### Profiling results (TFIM, n=24, order 2, N=50)

K_m depends on tau (not n) — operator spreading is contained within ~20 qubits at T=1:

| tau    | K      | hash (s) | sorted (s) | sorted/hash |
|--------|--------|----------|------------|-------------|
| 1e-08  |  4,092 | 0.051    | 0.099      | 1.94x       |
| 1e-10  | 11,245 | 0.122    | 0.214      | 1.75x       |
| 1e-12  | 27,065 | 0.313    | 0.561      | 1.79x       |
| 1e-14  | 60,115 | 0.718    | 1.329      | 1.85x       |

Hash serial outperforms sorted serial at all tested K values. OMP versions are slower than serial (thread overhead dominates at K < 10^5).

### Analysis

Each gate processes K active terms through: emit O(K) → merge → truncate. The backends differ only in the merge:

- **Hash-map merge:** K_a random insertions at cost C_h each.
- **Sorted merge:** sort partners O(K_a log K_a) + two-pointer sweep O(K), all sequential at cost C_m each.

Hash wins when K_a * C_h < K_a * log(K_a) * C_m, i.e., when C_h/C_m < log(K_a). With alpha = K_a/K ~ 0.5:

| K       | Hash table size | Cache level | C_h/C_m | log(K_a) | Winner  |
|---------|-----------------|-------------|---------|----------|---------|
| 10^3    | 40 KB           | L2          | ~3      | ~9       | hash    |
| 6*10^4  | 2.4 MB          | L3          | ~5-10   | ~15      | hash    |
| 4*10^5  | 16 MB           | spills DRAM | ~30-100 | ~18      | sorted  |

The crossover is at K ~ 4*10^5 where the hash table exceeds L3.

### The sort is the bottleneck — radix sort as the fix

The O(log K_a) factor comes from `std::sort` (comparison sort). Pauli string keys are pairs of `uint64_t` — fixed-width integers that admit radix sort at O(K_a) instead of O(K_a log K_a).

A 2-pass radix sort (one pass per 64-bit word, 256-bucket counting sort) costs ~4 * K_a * C_m. The crossover condition becomes:

    C_h/C_m > (1 + 4*alpha) / alpha = 6    (for alpha = 0.5)

This shifts the crossover from K ~ 4*10^5 (comparison sort) to K ~ 10^4 (radix sort), making sorted competitive at all K values currently reachable on CPU.

cuPauliProp uses exactly this approach on GPU (`cub::DeviceRadixSort`), where C_h/C_m ~ 100+ makes sorted+radix unconditionally faster.

**TODO:** implement radix sort in `sorted_conjugate` as a drop-in replacement for `std::sort`, benchmark against comparison sort to verify the crossover shift.
