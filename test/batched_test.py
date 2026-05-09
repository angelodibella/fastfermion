"""Tests for gate-batched Pauli propagation.

Verifies propagate_batched matches serial propagate.
"""

import pytest
import fastfermion as ff


def assert_equal(circuit, n_steps, obs_str, **kwargs):
    """Assert batched propagation matches serial."""
    obs = ff.PauliString(obs_str)
    full = circuit * n_steps
    ref = ff.propagate(full, obs, **kwargs)
    bat = ff.propagate_batched(full, obs, **kwargs)

    assert abs(ref.overlapwithzero() - bat.overlapwithzero()) < 1e-12, \
        f"expectation mismatch: serial={ref.overlapwithzero()}, batched={bat.overlapwithzero()}"
    assert len(ref.terms) == len(bat.terms), \
        f"term count mismatch: serial={len(ref.terms)}, batched={len(bat.terms)}"
    for ps, c in ref.terms.items():
        assert abs(c - bat.terms.get(ps, 0)) < 1e-12, \
            f"coefficient mismatch at {ps}: serial={c}, batched={bat.terms.get(ps, 0)}"


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
        g += [ff.ROT("XX", [i, j], 2 * dt), ff.ROT("YY", [i, j], 2 * dt),
              ff.ROT("ZZ", [i, j], 2 * dt)]
    return g


# -- agreement ----------------------------------------------------------------

def test_tfim():
    assert_equal(tfim(6, 0.1), 5, "Z" + "I" * 5)

def test_tfim_threshold():
    """Batched truncation schedule differs from serial (once per batch vs per gate).
    Expectation values should agree within truncation error."""
    obs = ff.PauliString("Z" + "I" * 5)
    full = tfim(6, 0.1) * 5
    ref = ff.propagate(full, obs, mincoeff=1e-3)
    bat = ff.propagate_batched(full, obs, mincoeff=1e-3)
    assert abs(ref.overlapwithzero() - bat.overlapwithzero()) < 1e-3

def test_heisenberg():
    assert_equal(heisenberg(4, 0.03), 5, "Z" + "I" * 3)

def test_clifford_mixed():
    c = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)]
    assert_equal(c, 5, "ZI")

# -- edge cases ---------------------------------------------------------------

def test_empty_circuit():
    assert_equal([], 1, "Z")

def test_single_gate():
    assert_equal([ff.ROT("X", (0,), 0.3)], 1, "Z")

def test_all_commuting_gates():
    """All X gates commute — should form one batch."""
    n = 6
    circuit = [ff.ROT("X", [i], 0.1) for i in range(n)]
    assert_equal(circuit, 3, "Z" + "I" * (n - 1))

def test_no_commuting_gates():
    """Adjacent ZZ gates don't commute — each is its own batch."""
    n = 4
    circuit = [ff.ROT("ZZ", [i, (i + 1) % n], 0.1) for i in range(n)]
    assert_equal(circuit, 3, "Z" + "I" * (n - 1))
