"""Tests for OpenMP-parallel Pauli propagation.

Verifies propagate_omp matches serial propagate across thread counts,
models, and truncation settings.
"""

import pytest
import fastfermion as ff


def assert_equal(circuit, n_steps, obs, n_threads, **kwargs):
    """Assert serial and OMP results agree (expectation, terms, coefficients)."""
    if isinstance(obs, str):
        obs = ff.PauliString(obs)
    full = circuit * n_steps
    ref = ff.propagate(full, obs, **kwargs)
    omp = ff.propagate_omp(full, obs, n_threads=n_threads, **kwargs)
    assert abs(ref.overlapwithzero() - omp.overlapwithzero()) < 1e-12
    assert len(ref.terms) == len(omp.terms)
    for ps, c in ref.terms.items():
        assert abs(c - omp.terms.get(ps, 0)) < 1e-12


def tfim(n, dt):
    """First-order TFIM step (J=h=1)."""
    g = []
    for i in range(n):
        g.append(ff.ROT("ZZ", [i, (i + 1) % n], -2 * dt))
    for i in range(n):
        g.append(ff.ROT("X", [i], -2 * dt))
    return g


def heisenberg(n, dt):
    """First-order Heisenberg step (J=1)."""
    g = []
    for i in range(n):
        j = (i + 1) % n
        g += [
            ff.ROT("XX", [i, j], 2 * dt),
            ff.ROT("YY", [i, j], 2 * dt),
            ff.ROT("ZZ", [i, j], 2 * dt),
        ]
    return g


# -- agreement -----------------------------------------------------------------


@pytest.mark.parametrize("n_threads", [2, 4, 7])
def test_tfim(n_threads):
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5, n_threads)


@pytest.mark.parametrize("n_threads", [2, 4])
def test_tfim_threshold(n_threads):
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5, n_threads, mincoeff=1e-3)


@pytest.mark.parametrize("n_threads", [2, 4])
def test_tfim_maxdegree(n_threads):
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5, n_threads, maxdegree=3)


@pytest.mark.parametrize("n_threads", [2, 4])
def test_heisenberg(n_threads):
    assert_equal(heisenberg(4, 0.03), 5, "Z" + "I" * 3, n_threads)


def test_mixed_clifford_rot():
    c = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)]
    assert_equal(c, 5, "ZI", 4)


def test_polynomial_observable():
    obs = ff.PauliPolynomial(ff.PauliString("ZI")) + ff.PauliPolynomial(
        ff.PauliString("IZ")
    )
    full = tfim(4, 0.05) * 10
    ref = ff.propagate(full, obs)
    omp = ff.propagate_omp(full, obs, n_threads=4)
    assert abs(ref.overlapwithzero() - omp.overlapwithzero()) < 1e-12


# -- edge cases ----------------------------------------------------------------


def test_fallback():
    assert_equal([ff.ROT("X", (0,), 0.3)], 1, "Z", 1)


def test_empty_circuit():
    assert_equal([], 1, "Z", 4)


def test_identity_rotation():
    obs = ff.PauliString("Z")
    omp = ff.propagate_omp([ff.ROT("X", (0,), 0.0)] * 10, obs, n_threads=4)
    assert abs(omp.overlapwithzero() - 1.0) < 1e-14


def test_more_threads_than_terms():
    assert_equal([ff.ROT("X", (0,), 0.3)], 1, "Z", 32)


def test_all_commuting():
    assert_equal([ff.ROT("ZZ", [0, 1], 0.3)], 10, "ZI", 4)


def test_clifford_only():
    assert_equal([ff.H(0), ff.CNOT(0, 1)], 1, "ZI", 4)
