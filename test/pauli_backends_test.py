"""
Tests of the OpenMP backend of Pauli propagation: "sharded" must agree with "serial" (up to rounding)
under every truncation rule and schedule, with and without batching.
"""

import pytest

from common import tfim_circuit, heisenberg_circuit, assert_poly_close
import fastfermion as ff
from fastfermion import propagate, PauliString, PauliPolynomial

pytestmark = pytest.mark.skipif(not ff.has_openmp, reason="fastfermion built without OpenMP")

def z0(n):
    return PauliString("Z" + "I" * (n - 1))

def assert_sharded_agrees(circ, obs, n_threads=4, **kwargs):
    ref = propagate(circ, obs, parallel="serial", **kwargs)
    out = propagate(circ, obs, parallel="sharded", n_threads=n_threads, **kwargs)
    assert_poly_close(ref, out)
    assert abs(ref.overlapwithzero() - out.overlapwithzero()) < 1e-12


@pytest.mark.parametrize("n_threads", [2, 4, 7])
def test_tfim(n_threads):
    assert_sharded_agrees(tfim_circuit(6, 0.1) * 5, z0(6), n_threads)

@pytest.mark.parametrize("kwargs", [
    dict(maxdegree=3),
    dict(mincoeff=1e-3),
    dict(topk=30),
    dict(max_xweight=2),
    dict(maxdegree=3, maxdegree_period=3),
    dict(maxdegree=3, mincoeff=1e-5, mincoeff_period=4),
    dict(maxdegree=3, maxdegree_period=2, mincoeff=1e-5, mincoeff_period=3),
    dict(max_xweight=2, xweight_period=3),
])
@pytest.mark.parametrize("batched", [False, True])
def test_truncation_rules(kwargs, batched):
    # disordered angles: top-k is sensitive to ties, which the backends' summation orders break differently
    assert_sharded_agrees(tfim_circuit(6, 0.1, seed=1) * 5, z0(6), batched=batched, **kwargs)

def test_heisenberg():
    assert_sharded_agrees(heisenberg_circuit(4, 0.03) * 5, z0(4))

def test_clifford_and_rotations():
    circ = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)] * 5
    assert_sharded_agrees(circ, z0(2))

def test_polynomial_observable():
    obs = PauliPolynomial(PauliString("ZIII")) + PauliPolynomial(PauliString("IZII")) * 0.5
    assert_sharded_agrees(tfim_circuit(4, 0.05) * 10, obs)

@pytest.mark.parametrize("circ,obs", [
    ([], z0(1)),
    ([ff.ROT("X", (0,), 0.3)], z0(1)),
    ([ff.ROT("ZZ", [0, 1], 0.3)] * 10, z0(2)),  # all gates commute with the observable
    ([ff.H(0), ff.CNOT(0, 1)], z0(2)),          # Clifford gates only
])
def test_small_circuits(circ, obs):
    assert_sharded_agrees(circ, obs, n_threads=32)  # more threads than terms

def test_certificate_agrees():
    circ = tfim_circuit(6, 0.1) * 5
    propagate(circ, z0(6), maxdegree=3, mincoeff=1e-4, mincoeff_period=2)
    ref = ff.trunc_stats()
    propagate(circ, z0(6), maxdegree=3, mincoeff=1e-4, mincoeff_period=2, parallel="sharded", n_threads=4)
    out = ff.trunc_stats()
    assert out["n_tau_events"] == ref["n_tau_events"] and out["peak_terms"] == ref["peak_terms"]
    assert out["cert_tau"] == pytest.approx(ref["cert_tau"], rel=1e-12)

def test_auto_selects_sharded_and_n_threads_one_is_serial():
    circ = tfim_circuit(6, 0.1) * 5
    assert_poly_close(propagate(circ, z0(6), n_threads=4), propagate(circ, z0(6), parallel="sharded", n_threads=4), tol=0)
    assert_poly_close(propagate(circ, z0(6), parallel="sharded"), propagate(circ, z0(6), parallel="serial"), tol=0)

def test_unknown_backend_raises():
    with pytest.raises(Exception):
        propagate(tfim_circuit(4, 0.1), z0(4), parallel="shard", n_threads=4)
