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
