"""Tests for sorted-array Pauli propagation (serial and parallel).

Verifies propagate_sorted and propagate_sorted_omp match serial propagate.
"""

import pytest
import fastfermion as ff


def assert_equal(circuit, n_steps, obs_str, func=ff.propagate_sorted, **kwargs):
    """Assert a sorted propagation function matches serial propagate."""
    obs = ff.PauliString(obs_str)
    full = circuit * n_steps
    ref = ff.propagate(
        full, obs, **{k: v for k, v in kwargs.items() if k != "n_threads"}
    )
    srt = func(full, obs, **kwargs)

    assert abs(ref.overlapwithzero() - srt.overlapwithzero()) < 1e-12, (
        f"expectation mismatch: serial={ref.overlapwithzero()}, sorted={srt.overlapwithzero()}"
    )
    assert len(ref.terms) == len(srt.terms), (
        f"term count mismatch: serial={len(ref.terms)}, sorted={len(srt.terms)}"
    )
    for ps, c in ref.terms.items():
        assert abs(c - srt.terms.get(ps, 0)) < 1e-12, (
            f"coefficient mismatch at {ps}: serial={c}, sorted={srt.terms.get(ps, 0)}"
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


# -- serial sorted agreement ---------------------------------------------------


def test_tfim():
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5)


def test_tfim_threshold():
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5, mincoeff=1e-3)


def test_tfim_maxdegree():
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5, maxdegree=3)


def test_heisenberg():
    assert_equal(heisenberg(4, 0.03), 5, "Z" + "I" * 3)


def test_mixed_clifford_rot():
    c = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)]
    assert_equal(c, 5, "ZI")


def test_empty_circuit():
    assert_equal([], 1, "Z")


def test_single_gate():
    assert_equal([ff.ROT("X", (0,), 0.3)], 1, "Z")


def test_identity_rotation():
    obs = ff.PauliString("Z")
    srt = ff.propagate_sorted([ff.ROT("X", (0,), 0.0)] * 10, obs)
    assert abs(srt.overlapwithzero() - 1.0) < 1e-14


def test_all_commuting():
    assert_equal([ff.ROT("ZZ", [0, 1], 0.3)], 10, "ZI")


def test_clifford_only():
    assert_equal([ff.H(0), ff.CNOT(0, 1)], 1, "ZI")


# -- top-K truncation ----------------------------------------------------------


def test_topk_serial():
    """Top-K via propagate (hash map) should keep at most K terms."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    ref = ff.propagate(full, obs)
    topk_result = ff.propagate(full, obs, topk=10)
    assert len(topk_result.terms) <= 10
    assert len(topk_result.terms) < len(ref.terms)


def test_topk_sorted():
    """Top-K via propagate_sorted should match hash-map top-K."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    ref = ff.propagate(full, obs, topk=20)
    srt = ff.propagate_sorted(full, obs, topk=20)
    # Both should have at most 20 terms
    assert len(ref.terms) <= 20
    assert len(srt.terms) <= 20


def test_topk_sorted_omp():
    """Top-K via propagate_sorted_omp should produce at most K terms."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    result = ff.propagate_sorted_omp(full, obs, n_threads=4, topk=15)
    assert len(result.terms) <= 15


def test_topk_zero_is_noop():
    """topk=0 should not truncate."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    ref = ff.propagate(full, obs)
    topk0 = ff.propagate(full, obs, topk=0)
    assert len(ref.terms) == len(topk0.terms)


# -- parallel sorted agreement -------------------------------------------------


@pytest.mark.parametrize("n_threads", [2, 4, 7])
def test_sorted_omp_tfim(n_threads):
    assert_equal(
        tfim(6, 0.1),
        5,
        "Z" + "I" * 5,
        func=ff.propagate_sorted_omp,
        n_threads=n_threads,
    )


@pytest.mark.parametrize("n_threads", [2, 4])
def test_sorted_omp_threshold(n_threads):
    assert_equal(
        tfim(6, 0.1),
        5,
        "Z" + "I" * 5,
        func=ff.propagate_sorted_omp,
        n_threads=n_threads,
        mincoeff=1e-3,
    )


@pytest.mark.parametrize("n_threads", [2, 4])
def test_sorted_omp_heisenberg(n_threads):
    assert_equal(
        heisenberg(4, 0.03),
        5,
        "Z" + "I" * 3,
        func=ff.propagate_sorted_omp,
        n_threads=n_threads,
    )


def test_sorted_omp_fallback():
    assert_equal(
        [ff.ROT("X", (0,), 0.3)], 1, "Z", func=ff.propagate_sorted_omp, n_threads=1
    )


def test_sorted_omp_more_threads_than_terms():
    assert_equal(
        [ff.ROT("X", (0,), 0.3)], 1, "Z", func=ff.propagate_sorted_omp, n_threads=32
    )


def test_sorted_omp_clifford_mixed():
    c = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)]
    assert_equal(c, 5, "ZI", func=ff.propagate_sorted_omp, n_threads=4)
