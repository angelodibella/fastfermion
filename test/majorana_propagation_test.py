"""
Tests of Majorana propagation: exactness on free circuits, parity conservation, agreement with the
Pauli propagation through the Jordan-Wigner transform, the unpaired-mode cutoff, and the truncation
schedules and certificate as in pauli_propagation_test.py.
"""

import math
import re
import numpy as np
import pytest

from common import tv_circuit, poly_terms, assert_poly_close
import fastfermion as ff
from fastfermion import propagate, MajoranaString, MajoranaPolynomial


def jw_circuit(circ):
    """The Pauli rotations implementing a circuit of Majorana rotations under Jordan-Wigner"""
    gates = []
    for gate in circ:
        ms = gate.axis
        r = 0 if ms.degree() % 4 in (0, 1) else 1  # i^r ms is Hermitian
        ((ps, phase),) = list(ff.jw(MajoranaPolynomial(ms)).terms.items())
        sign = ((1j ** r) * phase).real  # jw(i^r ms) = sign * ps with sign = +-1
        assert abs(abs(sign) - 1) < 1e-12
        labels, sites = zip(*[(label, int(q)) for label, q in re.findall(r"([XYZ])(\d+)", str(ps))])
        gates.append(ff.ROT("".join(labels), list(sites), sign * gate.theta))
    return gates


@pytest.mark.parametrize("deg", [2, 4])
def test_free_circuit_conserves_degree(deg):
    # quadratic gates conserve the degree, so a cutoff at the degree of the observable is exact
    circ = tv_circuit(5, 1.0, 0.0, 0.07, 6)
    obs = MajoranaString([1, 2] if deg == 2 else [1, 2, 3, 4])
    cut = propagate(circ, obs, maxdegree=deg)
    assert_poly_close(propagate(circ, obs), cut, tol=1e-14)
    assert all(k.degree() <= deg for k in cut.terms)

def test_interacting_circuit_grows_degree():
    circ = tv_circuit(4, 1.0, 2.0, 0.1, 6)
    assert any(k.degree() > 2 for k in propagate(circ, MajoranaString([1, 2])).terms)

def test_parity_conserved():
    circ = tv_circuit(4, 1.0, 2.0, 0.1, 8)
    for obs, parity in ((MajoranaString([1, 2]), 0), (MajoranaString([1]), 1)):
        assert all(k.degree() % 2 == parity for k in propagate(circ, obs, maxdegree=6).terms)

def test_agrees_with_pauli_propagation():
    circ = tv_circuit(4, 1.0, 2.0, 0.09, 4)
    obs = MajoranaString([1, 2])
    assert_poly_close(ff.jw(propagate(circ, obs)), propagate(jw_circuit(circ), ff.jw(MajoranaPolynomial(obs))))


# Unpaired modes

def test_unpaired():
    # mode j owns the Majorana operators 2j and 2j+1
    assert MajoranaString([0, 1]).unpaired() == 0
    assert MajoranaString([1, 2]).unpaired() == 2
    assert MajoranaString([0, 1, 2]).unpaired() == 1
    assert MajoranaString([64, 65]).unpaired() == 0
    assert MajoranaString([63, 64]).unpaired() == 2

def test_max_unpaired_is_max_xweight_under_jw():
    # the unpaired count of a monomial is the x-weight of its Jordan-Wigner image, so the two
    # cutoffs must agree gate for gate
    circ = tv_circuit(4, 1.0, 2.0, 0.09, 4)
    obs = MajoranaString([0, 1])  # mode 0 paired: survives max_unpaired=0
    pcirc, pobs = jw_circuit(circ), ff.jw(MajoranaPolynomial(obs))
    for u in (0, 2, 4):
        for batched in (False, True):
            maj = propagate(circ, obs, max_unpaired=u, batched=batched)
            assert len(maj) > 0
            assert_poly_close(ff.jw(maj), propagate(pcirc, pobs, max_xweight=u, batched=batched))

def test_max_unpaired_inert_when_large():
    circ = tv_circuit(4, 1.0, 2.0, 0.07, 4)
    obs = MajoranaString([2, 3])
    assert_poly_close(propagate(circ, obs), propagate(circ, obs, max_unpaired=10 ** 6), tol=0)


# Schedules and certificate

def test_threshold_within_certificate():
    circ = tv_circuit(4, 1.0, 2.0, 0.08, 6)
    obs = MajoranaString([1, 2])
    ref = poly_terms(propagate(circ, obs, maxdegree=4, batched=False))
    out = poly_terms(propagate(circ, obs, maxdegree=4, mincoeff=1e-4, mincoeff_period=3, batched=False))
    stats = ff.trunc_stats()
    assert stats["n_tau_events"] > 0
    drift = math.sqrt(sum(abs(ref.get(k, 0) - out.get(k, 0)) ** 2 for k in ref.keys() | out.keys()))
    assert drift <= stats["cert_tau"] + 1e-15

def test_deferred_degree_differs_from_immediate():
    circ = tv_circuit(4, 1.0, 3.0, 0.35, 8)
    obs = MajoranaString([1, 2])
    immediate = propagate(circ, obs, maxdegree=2, batched=False)
    deferred = propagate(circ, obs, maxdegree=2, maxdegree_period=7, batched=False)
    assert poly_terms(immediate) != poly_terms(deferred)

def test_batched_equals_unbatched():
    circ = tv_circuit(5, 1.0, 2.0, 0.09, 4)
    obs = MajoranaString([1, 2])
    assert_poly_close(propagate(circ, obs, batched=False), propagate(circ, obs, batched=True))

def test_stats_reset_per_call():
    circ = tv_circuit(3, 1.0, 1.0, 0.1, 4)
    obs = MajoranaString([1, 2])
    propagate(circ, obs, maxdegree=4, mincoeff=1e-4)
    assert ff.trunc_stats()["n_tau_events"] > 0
    propagate(circ, obs, maxdegree=4)
    stats = ff.trunc_stats()
    assert stats["n_tau_events"] == 0 and stats["cert_tau"] == 0
