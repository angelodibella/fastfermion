"""
Tests of the truncation rules, their schedules and gate batching in Pauli propagation.

The scheduled semantics are spelled out in reference_scheduled(), a gate-by-gate propagation written
with the Pauli string algebra only; the certificate cert_tau of trunc_stats() is checked against it
and against the error it bounds.
"""

import math
import numpy as np
import pytest

from common import tfim_circuit, heisenberg_circuit, poly_terms, assert_poly_close
import fastfermion as ff
from fastfermion import propagate, PauliString


def z0(n):
    return PauliString("Z" + "I" * (n - 1))

def tfim_specs(n, steps=4, dt=0.05, seed=1):
    """A TFIM circuit with slightly disordered fields, as (label, sites, angle) triples"""
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(steps):
        specs += [("XX", (i, i + 1), 2 * dt) for i in range(n - 1)]
        specs += [("Z", (i,), 1.4 * dt * (1 + 0.1 * rng.random())) for i in range(n)]
    return specs

def circuit(specs):
    return [ff.ROT(label, list(sites), theta) for label, sites, theta in specs]

def reference_scheduled(specs, n, obs, maxdegree, period, mincoeff=0.0, mincoeff_period=1):
    """Gate-by-gate propagation without batching: after gate g (0-based, last gate first), when
    (g+1) % period == 0 drop the terms of degree > maxdegree, then when (g+1) % mincoeff_period == 0
    drop the terms of magnitude <= mincoeff, adding the norm of the dropped terms to the certificate.
    Returns (dict of terms, certificate)."""
    state = {obs: 1.0 + 0j}
    cert = 0.0
    for g, (label, sites, theta) in enumerate(reversed(specs)):
        chars = ["I"] * n
        for ch, q in zip(label, sites):
            chars[q] = ch
        p = PauliString("".join(chars))
        new = {}
        for q, c in state.items():
            if q.commutes(p):
                new[q] = new.get(q, 0) + c
            else:
                ((pq, phase),) = list((p * q).terms.items())
                new[q] = new.get(q, 0) + c * math.cos(theta)
                new[pq] = new.get(pq, 0) + c * 1j * math.sin(theta) * complex(phase)
        state = new
        if (g + 1) % period == 0:
            state = {q: v for q, v in state.items() if q.degree() <= maxdegree}
        if mincoeff > 0 and (g + 1) % mincoeff_period == 0:
            cert += math.sqrt(sum(abs(v) ** 2 for v in state.values() if abs(v) <= mincoeff))
            state = {q: v for q, v in state.items() if abs(v) > mincoeff}
    return {str(q): v for q, v in state.items() if v != 0}, cert


# Gate batching

@pytest.mark.parametrize("circ,steps,n", [
    (tfim_circuit(6, 0.1), 5, 6),
    (heisenberg_circuit(4, 0.03), 5, 4),
    ([ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3)], 5, 2),
    ([ff.ROT("X", [i], 0.1) for i in range(6)], 3, 6),                # all gates commute
    ([ff.ROT("ZZ", [i, (i + 1) % 4], 0.1) for i in range(4)], 3, 4),  # no two adjacent gates commute
    ([], 1, 1),
])
def test_batched_equals_unbatched(circ, steps, n):
    assert_poly_close(propagate(circ * steps, z0(n), batched=False),
                      propagate(circ * steps, z0(n), batched=True))

def test_batched_threshold_within_certificate():
    # batching changes when the threshold fires, so the results differ, but each is within its
    # certificate of the threshold-free propagation
    circ = tfim_circuit(6, 0.1) * 5
    exact = poly_terms(propagate(circ, z0(6)))
    for batched in (False, True):
        out = poly_terms(propagate(circ, z0(6), mincoeff=1e-3, batched=batched))
        drift = math.sqrt(sum(abs(exact.get(k, 0) - out.get(k, 0)) ** 2 for k in exact.keys() | out.keys()))
        assert drift <= ff.trunc_stats()["cert_tau"] + 1e-15


# Rules

def test_topk():
    circ = tfim_circuit(6, 0.1) * 5
    full = propagate(circ, z0(6))
    assert len(full) > 20
    out = propagate(circ, z0(6), topk=20)
    assert len(out) == 20

def test_max_xweight():
    circ = heisenberg_circuit(5, 0.1) * 4
    for k in (1, 2):
        out = propagate(circ, z0(5), max_xweight=k)
        assert all(s.degree("X") + s.degree("Y") <= k for s in out.terms)
        # enforced as the terms are created, so batching does not change it
        assert_poly_close(propagate(circ, z0(5), max_xweight=k, batched=True), out, tol=0)
    assert_poly_close(propagate(circ, z0(5), max_xweight=5), propagate(circ, z0(5)))

def test_topk_ties_are_deterministic():
    # 2k terms of equal magnitude, a gate commuting with all of them: the k kept terms must not
    # depend on the order of the terms in the map (nor on the backend)
    k = 6
    strings = [PauliString(s) for s in ("XIII", "YIII", "ZIII", "IXII", "IYII", "IZII", "IIXI", "IIYI", "IIZI", "IIIX", "IIIY", "IIIZ")]
    gate = [ff.ROT("Z", [7], 0.3)]
    kept = set()
    for order in (strings, strings[::-1], strings[3:] + strings[:3]):
        poly = ff.PauliPolynomial()
        for s in order:
            poly += ff.PauliPolynomial(s) * 0.5
        out = propagate(gate, poly, topk=k)
        assert len(out) == k
        kept.add(frozenset(str(s) for s in out.terms))
        if ff.has_openmp:
            kept.add(frozenset(str(s) for s in propagate(gate, poly, topk=k, n_threads=4).terms))
    assert len(kept) == 1

def test_mincoeff_period_inert_without_threshold():
    specs = tfim_specs(6)
    ref = propagate(circuit(specs), z0(6), maxdegree=3)
    for period in (1, 3, 10):
        assert_poly_close(propagate(circuit(specs), z0(6), maxdegree=3, mincoeff_period=period), ref, tol=0)

@pytest.mark.parametrize("batched", [False, True])
def test_period_one_is_default(batched):
    specs = tfim_specs(6)
    ref = propagate(circuit(specs), z0(6), maxdegree=3, mincoeff=1e-6, batched=batched)
    out = propagate(circuit(specs), z0(6), maxdegree=3, mincoeff=1e-6, batched=batched,
                    maxdegree_period=1, mincoeff_period=1)
    assert_poly_close(ref, out, tol=0)

@pytest.mark.parametrize("period", [2, 5])
def test_deferred_degree_at_full_degree_is_exact(period):
    specs = tfim_specs(5, steps=3)
    assert_poly_close(propagate(circuit(specs), z0(5), maxdegree=5, maxdegree_period=period),
                      propagate(circuit(specs), z0(5)), tol=0)

def test_deferred_degree_matches_reference():
    n, w, period = 4, 2, 3
    specs = tfim_specs(n, steps=2, dt=0.13)
    out = poly_terms(propagate(circuit(specs), z0(n), maxdegree=w, maxdegree_period=period, batched=False))
    ref, _ = reference_scheduled(specs, n, z0(n), w, period)
    assert out.keys() == ref.keys()
    assert all(abs(out[k] - ref[k]) < 1e-12 for k in ref)

def test_deferred_degree_differs_from_immediate():
    # a deferred cutoff is a different rule: terms above the cutoff can be rotated back below it
    n, w = 4, 2
    specs = tfim_specs(n, steps=3, dt=0.2)
    immediate = propagate(circuit(specs), z0(n), maxdegree=w, batched=False)
    deferred = propagate(circuit(specs), z0(n), maxdegree=w, maxdegree_period=4, batched=False)
    assert poly_terms(immediate) != poly_terms(deferred)


# Certificate and statistics

def test_certificate_matches_reference_and_bounds_error():
    n, w, tau, period = 4, 3, 5e-4, 2
    specs = tfim_specs(n, steps=3, dt=0.11)
    ref = poly_terms(propagate(circuit(specs), z0(n), maxdegree=w, batched=False))
    out = poly_terms(propagate(circuit(specs), z0(n), maxdegree=w, mincoeff=tau, mincoeff_period=period, batched=False))
    stats = ff.trunc_stats()
    _, cert = reference_scheduled(specs, n, z0(n), w, 1, mincoeff=tau, mincoeff_period=period)
    assert stats["cert_tau"] == pytest.approx(cert, rel=1e-12)
    assert stats["n_tau_events"] > 0
    drift = math.sqrt(sum(abs(ref.get(k, 0) - out.get(k, 0)) ** 2 for k in ref.keys() | out.keys()))
    assert drift <= stats["cert_tau"] + 1e-15

def test_stats_reset_per_call():
    specs = tfim_specs(4)
    propagate(circuit(specs), z0(4), maxdegree=3, mincoeff=1e-4)
    assert ff.trunc_stats()["n_tau_events"] > 0
    propagate(circuit(specs), z0(4), maxdegree=3)
    stats = ff.trunc_stats()
    assert stats["n_tau_events"] == 0 and stats["cert_tau"] == 0 and stats["peak_terms"] > 0

def test_periods_validated():
    with pytest.raises(Exception):
        propagate(circuit(tfim_specs(4)), z0(4), maxdegree_period=0)
