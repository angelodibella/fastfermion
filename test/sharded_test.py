"""Tests for sharded hash-map Pauli propagation.

Verifies propagate(parallel="sharded") matches propagate(parallel="serial").
"""

import pytest
import fastfermion as ff


def assert_equal(circuit, n_steps, obs, n_threads, **kwargs):
    if isinstance(obs, str):
        obs = ff.PauliString(obs)
    full = circuit * n_steps
    ref = ff.propagate(full, obs, parallel="serial", **kwargs)
    shd = ff.propagate(full, obs, n_threads=n_threads, parallel="sharded", **kwargs)
    assert abs(ref.overlapwithzero() - shd.overlapwithzero()) < 1e-12, (
        f"expectation: serial={ref.overlapwithzero()}, sharded={shd.overlapwithzero()}"
    )
    assert len(ref.terms) == len(shd.terms), (
        f"term count: serial={len(ref.terms)}, sharded={len(shd.terms)}"
    )
    for ps, c in ref.terms.items():
        assert abs(c - shd.terms.get(ps, 0)) < 1e-12, (
            f"coeff at {ps}: serial={c}, sharded={shd.terms.get(ps, 0)}"
        )


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
    ref = ff.propagate(full, obs, parallel="serial")
    shd = ff.propagate(full, obs, n_threads=4, parallel="sharded")
    assert abs(ref.overlapwithzero() - shd.overlapwithzero()) < 1e-12


# -- edge cases ----------------------------------------------------------------


def test_two_threads():
    assert_equal(tfim(4, 0.1), 3, "Z" + "I" * 3, 2)


def test_more_threads_than_terms():
    assert_equal([ff.ROT("X", (0,), 0.3)], 1, "Z", 32)


def test_all_commuting():
    assert_equal([ff.ROT("ZZ", [0, 1], 0.3)], 10, "ZI", 4)


def test_clifford_only():
    assert_equal([ff.H(0), ff.CNOT(0, 1)], 1, "ZI", 4)


def test_unbatched():
    """Sharded with batched=False should match serial unbatched."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    ref = ff.propagate(full, obs, batched=False, parallel="serial")
    shd = ff.propagate(full, obs, n_threads=4, batched=False, parallel="sharded")
    assert abs(ref.overlapwithzero() - shd.overlapwithzero()) < 1e-12
    assert len(ref.terms) == len(shd.terms)
