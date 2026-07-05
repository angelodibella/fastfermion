"""Tests for the CUDA GPU Pauli propagation backend.

Verifies propagate(parallel="gpu") against propagate(parallel="serial"):
bit-identical retained sets under the pure weight cutoff, coefficient
agreement, matched threshold schedules, and the device phase rule against
the host oracle. Skipped when fastfermion was built without GPU support or
no CUDA device is present.
"""

import pytest
import fastfermion as ff

pytestmark = pytest.mark.skipif(
    not ff.has_gpu or ff.gpu_device_count() == 0,
    reason="fastfermion built without GPU support or no CUDA device",
)


def assert_equal(circuit, n_steps, obs, **kwargs):
    if isinstance(obs, str):
        obs = ff.PauliString(obs)
    full = circuit * n_steps
    ref = ff.propagate(full, obs, parallel="serial", **kwargs)
    gpu = ff.propagate(full, obs, parallel="gpu", **kwargs)
    assert len(ref.terms) == len(gpu.terms), (
        f"term count: serial={len(ref.terms)}, gpu={len(gpu.terms)}"
    )
    for ps, c in ref.terms.items():
        assert abs(c - gpu.terms.get(ps, 0)) < 1e-12, (
            f"coeff at {ps}: serial={c}, gpu={gpu.terms.get(ps, 0)}"
        )
    assert abs(ref.overlapwithzero() - gpu.overlapwithzero()) < 1e-12


def tfim(n, dt):
    g = []
    for i in range(n):
        g.append(ff.ROT("ZZ", [i, (i + 1) % n], -2 * dt))
    for i in range(n):
        g.append(ff.ROT("X", [i], -2 * dt))
    return g


def heisenberg(n, dt):
    g = []
    for i in range(n):
        j = (i + 1) % n
        g += [
            ff.ROT("XX", [i, j], 2 * dt),
            ff.ROT("YY", [i, j], 2 * dt),
            ff.ROT("ZZ", [i, j], 2 * dt),
        ]
    return g


def test_phase_rule():
    # underscore names are not re-exported by the package's `import *`
    from fastfermion import ffcore

    assert ffcore._gpu_phase_check(100000, 1234)
    assert ffcore._gpu_phase_check(100000, 5678)


@pytest.mark.parametrize("w", [2, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_tfim_weight_cutoff(w, batched):
    assert_equal(tfim(10, 0.05), 8, "Z0", maxdegree=w, batched=batched)


@pytest.mark.parametrize("w", [3, 4])
def test_heisenberg_weight_cutoff(w):
    assert_equal(heisenberg(8, 0.05), 6, "Z0", maxdegree=w)


@pytest.mark.parametrize("batched", [False, True])
def test_threshold_schedule(batched):
    # mincoeff > 0 must fire at the CPU cadence (per gate / per commuting batch)
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=4, mincoeff=1e-6, batched=batched)


def test_no_truncation_exact():
    assert_equal(heisenberg(6, 0.05), 4, "Z0")


def test_wide_keys():
    # Force the 2-word key path: a qubit index above 64.
    g = [ff.ROT("XX", [0, 65], 0.1), ff.ROT("ZZ", [0, 65], 0.1), ff.ROT("X", [0], 0.1)]
    assert_equal(g, 5, ff.PauliString("Z0 Z65"), maxdegree=4)


def test_polynomial_observable():
    obs = ff.PauliPolynomial(ff.PauliString("Z0"))
    obs += ff.PauliPolynomial(ff.PauliString("Z1")) * 0.5
    assert_equal(tfim(8, 0.05), 4, obs, maxdegree=3)


def test_rejects_unsupported():
    with pytest.raises(Exception):
        ff.propagate(tfim(4, 0.05), ff.PauliString("Z0"), parallel="gpu", topk=10)
    with pytest.raises(Exception):
        ff.propagate(tfim(4, 0.05), ff.PauliString("Z0"), parallel="gpu", max_xweight=2)
