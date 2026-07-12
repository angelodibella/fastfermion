"""Truncation-schedule tests (maxdegree_period / mincoeff_period / trunc_stats).

Validation gates from the frozen A.2 spec (LAB 8i):
(i)   any mincoeff_period with mincoeff=0 is bit-identical to the pure weight run;
(ii)  period 1 is bit-identical to the historical behavior;
(iii) deferred weight with maxdegree=n equals the untruncated run;
(iv)  a nontrivial deferred-weight case against a schedule reference written
      out gate-by-gate in this file (it reuses ff's string algebra, which the
      cross-validation suite already gates against the Python oracle; what is
      independently spelled out here is the SCHEDULING semantics);
(v)   the certificate: cert_tau equals the hand-accumulated event norms and
      bounds the drift from the weight-only reference.
"""

import math

import numpy as np
import pytest

import fastfermion as ff


def tfim_specs(n, steps=4, dt=0.05, seed=1):
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(steps):
        for i in range(n - 1):
            specs.append(("XX", (i, i + 1), 2 * dt))
        for i in range(n):
            specs.append(("Z", (i,), 2 * 0.7 * dt * (1 + 0.1 * rng.random())))
    return specs


def circuit(specs):
    return [ff.ROT(lab, list(sites), theta) for lab, sites, theta in specs]


def gate_string(lab, sites, n):
    chars = ["I"] * n
    for ch, q in zip(lab, sites):
        chars[q] = ch
    return ff.PauliString("".join(chars))


def obs_z(n, site=0):
    return ff.PauliString("".join("Z" if q == site else "I" for q in range(n)))


def poly_dict(p):
    return {str(k): complex(v) for k, v in p.terms.items()}


def assert_identical(a, b):
    da, db = poly_dict(a), poly_dict(b)
    assert da.keys() == db.keys()
    for k in da:
        assert da[k] == db[k], f"coefficient differs at {k}: {da[k]} vs {db[k]}"


# --- gate (i): threshold period is inert when the threshold is off ----------


@pytest.mark.parametrize("period", [1, 3, 10])
def test_mincoeff_period_inert_at_zero_threshold(period):
    specs = tfim_specs(6)
    ref = ff.propagate(circuit(specs), obs_z(6), maxdegree=3)
    out = ff.propagate(circuit(specs), obs_z(6), maxdegree=3, mincoeff_period=period)
    assert_identical(ref, out)


# --- gate (ii): period 1 is the historical behavior -------------------------


@pytest.mark.parametrize("batched", [False, True])
def test_period_one_is_default(batched):
    specs = tfim_specs(6)
    ref = ff.propagate(circuit(specs), obs_z(6), maxdegree=3, mincoeff=1e-6, batched=batched)
    out = ff.propagate(circuit(specs), obs_z(6), maxdegree=3, mincoeff=1e-6, batched=batched,
                       maxdegree_period=1, mincoeff_period=1)
    assert_identical(ref, out)


# --- gate (iii): deferred weight at full degree = untruncated ---------------


@pytest.mark.parametrize("period", [2, 5])
def test_deferred_weight_full_degree_untruncated(period):
    n = 5
    specs = tfim_specs(n, steps=3)
    ref = ff.propagate(circuit(specs), obs_z(n))
    out = ff.propagate(circuit(specs), obs_z(n), maxdegree=n, maxdegree_period=period)
    assert_identical(ref, out)


# --- gate (iv): deferred weight against a spelled-out schedule reference ----


def reference_scheduled(specs, n, obs, maxdegree, period, mincoeff=0.0, mincoeff_period=1):
    """The scheduled semantics, spelled out: gates processed in Heisenberg
    order (last first) with NO emission filter; after processed gate g
    (0-indexed), when (g+1) % period == 0 drop weight > maxdegree, then when
    (g+1) % mincoeff_period == 0 and mincoeff > 0 drop |c| <= mincoeff,
    accumulating the certificate sqrt(sum of discarded |c|^2) per event.
    Returns (dict keyed by compact string, certificate)."""
    state = {obs: 1.0 + 0j}
    cert = 0.0
    for g, (lab, sites, theta) in enumerate(reversed(specs)):
        p = gate_string(lab, sites, n)
        new = {}
        for q, c in state.items():
            if q.commutes(p):
                new[q] = new.get(q, 0) + c
            else:
                prod = p * q  # single-term polynomial: the partner with its phase
                ((pk, phase),) = list(prod.terms.items())
                new[q] = new.get(q, 0) + c * math.cos(theta)
                new[pk] = new.get(pk, 0) + c * 1j * math.sin(theta) * complex(phase)
        state = new
        if (g + 1) % period == 0:  # period 1 = after every gate = emission filter
            state = {q: v for q, v in state.items() if q.degree() <= maxdegree}
        if mincoeff > 0 and (g + 1) % mincoeff_period == 0:
            disc2 = sum(abs(v) ** 2 for v in state.values() if abs(v) <= mincoeff)
            cert += math.sqrt(disc2)
            state = {q: v for q, v in state.items() if abs(v) > mincoeff}
    return {str(q): v for q, v in state.items() if v != 0}, cert


def test_deferred_weight_matches_reference():
    n, w, period = 4, 2, 3
    specs = tfim_specs(n, steps=2, dt=0.13)
    out = ff.propagate(circuit(specs), obs_z(n), maxdegree=w, maxdegree_period=period,
                       batched=False)
    ref, _ = reference_scheduled(specs, n, obs_z(n), w, period)
    dout = poly_dict(out)
    assert set(dout) == set(ref)
    for k in ref:
        assert abs(dout[k] - ref[k]) < 1e-12


def test_deferred_weight_differs_from_emission_filter():
    # The deferred cutoff is a different rule: above-w strings rotating back
    # must change the result on a generic circuit (grow-back flux).
    n, w = 4, 2
    specs = tfim_specs(n, steps=3, dt=0.2)
    per_emission = ff.propagate(circuit(specs), obs_z(n), maxdegree=w, batched=False)
    deferred = ff.propagate(circuit(specs), obs_z(n), maxdegree=w, maxdegree_period=4,
                            batched=False)
    assert poly_dict(per_emission) != poly_dict(deferred)


# --- gate (v): the certificate ----------------------------------------------


def test_certificate_matches_reference_and_bounds_drift():
    n, w, tau, p_tau = 4, 3, 5e-4, 2
    specs = tfim_specs(n, steps=3, dt=0.11)
    ref_poly = ff.propagate(circuit(specs), obs_z(n), maxdegree=w, batched=False)
    out = ff.propagate(circuit(specs), obs_z(n), maxdegree=w, mincoeff=tau,
                       mincoeff_period=p_tau, batched=False)
    stats = ff.trunc_stats()
    # per-emission weight filtering equals a weight event after every gate
    # (no above-w string ever exists, so post-gate dropping = not emitting)
    _, cert_ref = reference_scheduled(specs, n, obs_z(n), w, 1, mincoeff=tau,
                                      mincoeff_period=p_tau)
    assert stats["cert_tau"] == pytest.approx(cert_ref, rel=1e-12)
    assert stats["n_tau_events"] > 0
    # drift from the weight-only reference bounded by the certificate
    da, db = poly_dict(ref_poly), poly_dict(out)
    drift = math.sqrt(sum(abs(da.get(k, 0) - db.get(k, 0)) ** 2 for k in set(da) | set(db)))
    assert drift <= stats["cert_tau"] + 1e-15


def test_stats_reset_per_run():
    specs = tfim_specs(4)
    ff.propagate(circuit(specs), obs_z(4), maxdegree=3, mincoeff=1e-4)
    first = ff.trunc_stats()
    ff.propagate(circuit(specs), obs_z(4), maxdegree=3)
    second = ff.trunc_stats()
    assert first["n_tau_events"] > 0
    assert second["n_tau_events"] == 0 and second["cert_tau"] == 0
    assert second["peak_terms"] > 0


def test_periods_validated():
    specs = tfim_specs(4)
    with pytest.raises(Exception):
        ff.propagate(circuit(specs), obs_z(4), maxdegree_period=0)


# --- backend agreement across schedules --------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(maxdegree=3, maxdegree_period=3),
    dict(maxdegree=3, mincoeff=1e-5, mincoeff_period=4),
    dict(maxdegree=3, maxdegree_period=2, mincoeff=1e-5, mincoeff_period=3),
])
def test_backends_agree_on_schedules(kwargs):
    if not ff.has_openmp:
        pytest.skip("no OpenMP")
    specs = tfim_specs(6, steps=3)
    serial = ff.propagate(circuit(specs), obs_z(6), batched=False, **kwargs)
    sharded = ff.propagate(circuit(specs), obs_z(6), n_threads=4, parallel="sharded",
                           batched=False, **kwargs)
    ds, dh = poly_dict(serial), poly_dict(sharded)
    assert set(ds) == set(dh)
    for k in ds:
        assert abs(ds[k] - dh[k]) < 1e-13


def test_unknown_parallel_strategy_throws():
    # Rebuild-gate F2: a typo'd strategy must fail loudly, never return the
    # un-evolved observable.
    with pytest.raises(Exception):
        ff.propagate(circuit(tfim_specs(4)), obs_z(4), n_threads=4, parallel="shard")
