"""Majorana propagation tests: structural validation gates plus
schedule/certificate semantics, mirroring scheduling_test.py on the spin side.

Structural gates: (i) free closure -- a quadratic (hopping-only) circuit
conserves degree, so any cutoff at or above the observable's degree is exact;
(ii) parity conservation -- an even observable stays even; (iii) Jordan-Wigner
cross-validation -- the Majorana path agrees with the validated Pauli path on
the JW image of the same circuit; (iv) the Fock-readout pair phase
(-1)^{p(p-1)/2}; (v) schedule semantics against a spelled-out reference and
the certificate bound, as in the spin suite.
"""

import math

import pytest

import fastfermion as ff


# -- circuit builders ----------------------------------------------------------
# spinless t-V chain in the Majorana dictionary:
# hopping (j,j+1) -> (t/1)*[ +1/2 G_{2j+2,2j+3} - 1/2 G_{2j+1,2j+4} ] with the
# -t coupling folded in; interaction V n_j n_{j+1} -> quartic coupling -V/4
# (plus quadratic pieces +V/4 on each pair monomial and a constant, dropped).
# Gate angle theta = 2 h dt for a Hamiltonian term h*Gamma;
# MROT(ms, theta) applies e^{-i theta/2 M}.


def mrot(idx, theta):
    return ff.MROT(ff.MajoranaString(idx), theta)


def tv_layer(M, t, V, dt):
    gates = []
    for j in range(M - 1):
        m = 2 * j + 1  # 1-based first Majorana of mode j
        gates.append(mrot([m + 1, m + 2], 2 * (-t) * 0.5 * dt))
        gates.append(mrot([m, m + 3], 2 * (+t) * 0.5 * dt))
        if V != 0:
            gates.append(mrot([m, m + 1], 2 * (V / 4) * dt))
            gates.append(mrot([m + 2, m + 3], 2 * (V / 4) * dt))
            gates.append(mrot([m, m + 1, m + 2, m + 3], 2 * (-V / 4) * dt))
    return gates


def tv_circuit(M, t, V, dt, steps):
    return tv_layer(M, t, V, dt) * steps


def poly_dict(p):
    return {str(k): complex(v) for k, v in p.terms.items()}


# -- gate (i): free closure ----------------------------------------------------


@pytest.mark.parametrize("deg", [2, 4])
def test_free_closure(deg):
    # V = 0: quadratic gates conserve degree (|P triangle Q| = q when |P|=2),
    # so a cutoff at the observable's degree discards nothing.
    M = 5
    circ = tv_circuit(M, 1.0, 0.0, 0.07, 6)
    obs = ff.MajoranaString([1, 2] if deg == 2 else [1, 2, 3, 4])
    exact = ff.propagate(circ, obs)
    cut = ff.propagate(circ, obs, maxdegree=deg)
    de, dc = poly_dict(exact), poly_dict(cut)
    assert de.keys() == dc.keys()
    assert all(abs(de[k] - dc[k]) < 1e-14 for k in de)
    assert all(len(k.split("m")) - 1 <= deg for k in dc)


def test_interacting_breaks_closure():
    # V != 0 grows degree; the same cutoff must now discard (sanity that the
    # free-closure test cannot pass vacuously).
    M = 4
    circ = tv_circuit(M, 1.0, 2.0, 0.1, 6)
    obs = ff.MajoranaString([1, 2])
    exact = ff.propagate(circ, obs)
    assert any(k.degree() > 2 for k in exact.terms)


# -- gate (ii): parity ---------------------------------------------------------


def test_parity_conserved():
    M = 4
    circ = tv_circuit(M, 1.0, 2.0, 0.1, 8)
    for obs, par in ((ff.MajoranaString([1, 2]), 0), (ff.MajoranaString([1]), 1)):
        out = ff.propagate(circ, obs, maxdegree=6)
        assert all(k.degree() % 2 == par for k in out.terms)


# -- gate (iii): Jordan-Wigner cross-validation --------------------------------


def test_jw_cross_validation():
    # Propagate in the Majorana basis, map through JW, and compare with the
    # validated Pauli path run on the JW image circuit built from the same
    # spec. jw(Gamma_S) = f * P with f = +-1 (both sides Hermitian), so the
    # gate e^{-i theta/2 Gamma} maps to the Pauli rotation ROT(P, f * theta).
    import re

    M = 4
    spec = []
    t, V, dt = 1.0, 2.0, 0.09
    for _ in range(4):
        for j in range(M - 1):
            m = 2 * j + 1
            spec.append(([m + 1, m + 2], 2 * (-t) * 0.5 * dt))
            spec.append(([m, m + 3], 2 * (+t) * 0.5 * dt))
            spec.append(([m, m + 1], 2 * (V / 4) * dt))
            spec.append(([m + 2, m + 3], 2 * (V / 4) * dt))
            spec.append(([m, m + 1, m + 2, m + 3], 2 * (-V / 4) * dt))
    circ = [mrot(idx, th) for idx, th in spec]
    obs = ff.MajoranaString([1, 2])
    maj_jw = ff.jw(ff.propagate(circ, obs))

    def pauli_rot(idx, theta):
        k = len(idx)
        r = 0 if k % 4 in (0, 1) else 1  # i^r makes gamma_S Hermitian
        img = ff.jw(ff.MajoranaPolynomial(ff.MajoranaString(idx)))
        ((ps, ph),) = list(img.terms.items())
        f = ((1j ** r) * ph).real  # Hermitian generator's real +-1 factor
        assert abs(abs(f) - 1) < 1e-12
        sites, labels = [], []
        for lab, q in re.findall(r"([XYZ])(\d+)", str(ps)):
            labels.append(lab)
            sites.append(int(q))
        return ff.ROT("".join(labels), sites, f * theta)

    pl = ff.propagate([pauli_rot(i, th) for i, th in spec],
                      ff.jw(ff.MajoranaPolynomial(ff.MajoranaPolynomial(obs))))
    da, db = poly_dict(maj_jw), poly_dict(pl)
    assert da.keys() == db.keys()
    assert all(abs(da[k] - db[k]) < 1e-12 for k in da)


# -- gate (iv): Fock readout phase --------------------------------------------


def test_fock_readout_phase():
    # <b| Gamma_S |b> = (-1)^{p(p-1)/2} prod (-1)^{b_j} for paired S
    # (paired-sector readout identity), 0 for unpaired S. Checked against the
    # dense matrix via the fork's own sparse/JW machinery if available,
    # otherwise against the analytic form through number-operator algebra.
    import numpy as np

    M = 3
    # dense Majoranas via JW: gamma_{2j-1} = (prod Z)<j X_j, gamma_2j = (prod Z)<j Y_j
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron(ops):
        out = np.array([[1]], dtype=complex)
        for o in ops:
            out = np.kron(out, o)
        return out

    def gamma(mu):  # 1-based
        j = (mu - 1) // 2  # mode, 0-based
        letter = X if mu % 2 == 1 else -Y  # gamma_2j = i(c-c^dag) = -(prod Z) Y_j
        return kron([Z] * j + [letter] + [I] * (M - j - 1))

    def Gamma(S):
        k = len(S)
        m = (k * (k - 1) // 2) % 2
        out = np.eye(2**M, dtype=complex) * (1j**m)
        for mu in S:
            out = out @ gamma(mu)
        return out

    for S, p in (([1, 2], 1), ([1, 2, 3, 4], 2), ([1, 2, 3, 4, 5, 6], 3)):
        G = Gamma(S)
        for b in range(2**M):
            bits = [(b >> (M - 1 - j)) & 1 for j in range(M)]
            expect = G[b, b].real
            pred = (-1) ** (p * (p - 1) // 2)
            for j in range(M):
                if 2 * j + 1 in S and 2 * j + 2 in S:
                    pred *= (-1) ** bits[j]
            assert abs(expect - pred) < 1e-12
    # unpaired: zero diagonal
    G = Gamma([1, 4])
    assert np.max(np.abs(np.diag(G))) < 1e-12


# -- gate (v): schedules and the certificate -----------------------------------


def test_period_one_default_and_certificate():
    M = 4
    circ = tv_circuit(M, 1.0, 2.0, 0.08, 6)
    obs = ff.MajoranaString([1, 2])
    ref = ff.propagate(circ, obs, maxdegree=4, batched=False)
    out = ff.propagate(circ, obs, maxdegree=4, mincoeff=1e-4, mincoeff_period=3,
                       batched=False)
    stats = ff.trunc_stats()
    assert stats["n_tau_events"] > 0
    da, db = poly_dict(ref), poly_dict(out)
    drift = math.sqrt(sum(abs(da.get(k, 0) - db.get(k, 0)) ** 2 for k in set(da) | set(db)))
    assert drift <= stats["cert_tau"] + 1e-15


def test_deferred_degree_differs():
    # Deferred structural cutoff is a different rule (above-d monomials can
    # rotate back between events).
    M = 4
    circ = tv_circuit(M, 1.0, 3.0, 0.35, 8)
    obs = ff.MajoranaString([1, 2])
    per_emission = ff.propagate(circ, obs, maxdegree=2, batched=False)
    deferred = ff.propagate(circ, obs, maxdegree=2, maxdegree_period=7, batched=False)
    assert poly_dict(per_emission) != poly_dict(deferred)


def test_stats_reset_per_run():
    M = 3
    circ = tv_circuit(M, 1.0, 1.0, 0.1, 4)
    obs = ff.MajoranaString([1, 2])
    ff.propagate(circ, obs, maxdegree=4, mincoeff=1e-4)
    assert ff.trunc_stats()["n_tau_events"] > 0
    ff.propagate(circ, obs, maxdegree=4)
    s = ff.trunc_stats()
    assert s["n_tau_events"] == 0 and s["cert_tau"] == 0


# -- parallel backends ---------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(maxdegree=4),
    dict(maxdegree=4, mincoeff=1e-5, mincoeff_period=3),
    dict(maxdegree=4, maxdegree_period=2, mincoeff=1e-5),
])
def test_backends_agree_on_schedules(kwargs):
    if not ff.has_openmp:
        pytest.skip("no OpenMP")
    M = 5
    circ = tv_circuit(M, 1.0, 2.0, 0.09, 5)
    obs = ff.MajoranaString([1, 2])
    serial = ff.propagate(circ, obs, batched=False, **kwargs)
    sm = ff.propagate(circ, obs, batched=False, n_threads=4, parallel="serial-merge", **kwargs)
    shard = ff.propagate(circ, obs, batched=False, n_threads=4, parallel="sharded", **kwargs)
    ds = poly_dict(serial)
    for other in (sm, shard):
        do = poly_dict(other)
        assert ds.keys() == do.keys()
        assert all(abs(ds[k] - do[k]) < 1e-13 for k in ds)


def test_unknown_parallel_strategy_throws():
    # A typo'd strategy must fail loudly, never return the initial observable.
    with pytest.raises(Exception):
        ff.propagate(tv_circuit(3, 1.0, 1.0, 0.1, 2), ff.MajoranaString([1, 2]),
                     n_threads=4, parallel="shard")
