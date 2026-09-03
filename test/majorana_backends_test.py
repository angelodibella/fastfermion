"""
Tests of the OpenMP backend of Majorana propagation, which must agree with "serial" under every
truncation rule and schedule.
"""

import pytest

from common import tv_circuit, assert_poly_close
import fastfermion as ff
from fastfermion import propagate, MajoranaString, MajoranaPolynomial

pytestmark = pytest.mark.skipif(not ff.has_openmp, reason="fastfermion built without OpenMP")

def assert_sharded_agrees(circ, obs, n_threads=4, **kwargs):
    assert_poly_close(propagate(circ, obs, parallel="serial", **kwargs),
                      propagate(circ, obs, parallel="sharded", n_threads=n_threads, **kwargs))


@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(maxdegree=4),
    dict(mincoeff=1e-4),
    dict(topk=30),
    dict(max_unpaired=2),
    dict(maxdegree=4, mincoeff=1e-5, mincoeff_period=3),
    dict(maxdegree=4, maxdegree_period=2, mincoeff=1e-5),
    dict(max_unpaired=2, unpaired_period=3),
])
@pytest.mark.parametrize("batched", [False, True])
def test_truncation_rules(kwargs, batched):
    # disordered couplings: top-k is sensitive to ties, which the backends' summation orders break differently
    assert_sharded_agrees(tv_circuit(5, 1.0, 2.0, 0.09, 5, seed=1), MajoranaString([1, 2]), batched=batched, **kwargs)

def test_polynomial_observable():
    obs = MajoranaPolynomial(MajoranaString([1, 2])) * 1j + MajoranaPolynomial(MajoranaString([3, 4])) * 0.5j
    assert_sharded_agrees(tv_circuit(4, 1.0, 2.0, 0.09, 4), obs, n_threads=32)

def test_certificate_agrees():
    circ = tv_circuit(5, 1.0, 2.0, 0.09, 5)
    obs = MajoranaString([1, 2])
    propagate(circ, obs, maxdegree=4, mincoeff=1e-4, mincoeff_period=2)
    ref = ff.trunc_stats()
    propagate(circ, obs, maxdegree=4, mincoeff=1e-4, mincoeff_period=2, parallel="sharded", n_threads=4)
    out = ff.trunc_stats()
    assert out["n_tau_events"] == ref["n_tau_events"] and out["peak_terms"] == ref["peak_terms"]
    assert out["cert_tau"] == pytest.approx(ref["cert_tau"], rel=1e-12)

def test_unknown_backend_raises():
    with pytest.raises(Exception):
        propagate(tv_circuit(3, 1.0, 1.0, 0.1, 2), MajoranaString([1, 2]), parallel="shard", n_threads=4)
