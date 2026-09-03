"""
Tests of the CUDA backend: propagate(parallel="gpu") must retain the same terms as the serial
propagation under the degree cutoff and the coefficient threshold, with every schedule, key format
and allocation policy, and its certificate must agree. Skipped without a GPU build or a device.
"""

import pytest

from common import tfim_circuit, heisenberg_circuit, tv_circuit, assert_poly_close
import fastfermion as ff
from fastfermion import propagate, PauliString, PauliPolynomial, MajoranaString, MajoranaPolynomial

pytestmark = pytest.mark.skipif(not ff.has_gpu or ff.gpu_device_count() == 0,
                                reason="fastfermion built without GPU support, or no CUDA device")

def z0(n):
    return PauliString("Z" + "I" * (n - 1))

def assert_gpu_agrees(circ, obs, **kwargs):
    ref = propagate(circ, obs, parallel="serial", **kwargs)
    ref_stats = ff.trunc_stats()
    out = propagate(circ, obs, parallel="gpu", **kwargs)
    stats = ff.trunc_stats()
    assert_poly_close(ref, out)
    assert stats["n_tau_events"] == ref_stats["n_tau_events"]
    assert stats["cert_tau"] == pytest.approx(ref_stats["cert_tau"], rel=1e-12, abs=1e-15)


def test_phase_rule():
    from fastfermion import ffcore  # underscore names are not exported by the package
    assert ffcore._gpu_phase_check(100000, 1234)
    assert ffcore._gpu_phase_check(100000, 5678)

@pytest.mark.parametrize("w", [2, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_degree_cutoff(w, batched):
    assert_gpu_agrees(tfim_circuit(10, 0.05) * 8, z0(10), maxdegree=w, batched=batched)

@pytest.mark.parametrize("w", [3, 4])
def test_heisenberg(w):
    assert_gpu_agrees(heisenberg_circuit(8, 0.05) * 6, z0(8), maxdegree=w)

def test_no_truncation():
    assert_gpu_agrees(heisenberg_circuit(6, 0.05) * 4, z0(6))

@pytest.mark.parametrize("kwargs", [
    dict(maxdegree=4, mincoeff=1e-6),
    dict(maxdegree=4, mincoeff=1e-6, batched=False),
    dict(maxdegree=3, maxdegree_period=2),
    dict(maxdegree=3, maxdegree_period=5, batched=False),
    dict(maxdegree=4, mincoeff=1e-6, mincoeff_period=3, batched=False),
    dict(maxdegree=4, mincoeff=1e-6, mincoeff_period=7, batched=False),
    dict(maxdegree=3, maxdegree_period=3, mincoeff=1e-6, mincoeff_period=2, batched=False),
])
def test_schedules(kwargs):
    assert_gpu_agrees(tfim_circuit(8, 0.05) * 6, z0(8), **kwargs)

@pytest.mark.parametrize("key", ["dense", "support"])
@pytest.mark.parametrize("w", [4, 8])  # one or two words of support-list slots
def test_two_word_keys(key, w):
    circ = [ff.ROT("XX", [0, 65], 0.1), ff.ROT("ZZ", [0, 65], 0.1), ff.ROT("X", [0], 0.1), ff.ROT("YY", [65, 70], 0.2)] * 5
    assert_gpu_agrees(circ, PauliString("Z0 Z65"), maxdegree=w, gpu_key=key)

def test_polynomial_observable():
    obs = PauliPolynomial(PauliString("Z0")) + PauliPolynomial(PauliString("Z1")) * 0.5
    assert_gpu_agrees(tfim_circuit(8, 0.05) * 4, obs, maxdegree=3)

def test_clifford_gates():
    circ = [ff.H(0), ff.ROT("Z", (0,), 0.5), ff.CNOT(0, 1), ff.ROT("XX", [0, 1], 0.3), ff.S(1)] * 4
    assert_gpu_agrees(circ, z0(2))

def test_rejects_unsupported():
    with pytest.raises(Exception):
        propagate(tfim_circuit(4, 0.05), z0(4), parallel="gpu", topk=10)
    with pytest.raises(Exception):
        propagate(tfim_circuit(4, 0.05), z0(4), parallel="gpu", max_xweight=2)
    with pytest.raises(Exception):  # non-Hermitian observable
        propagate(tfim_circuit(4, 0.05), PauliPolynomial(z0(4)) * 1j, parallel="gpu")

def test_peak_device_bytes():
    # deterministic: same run, same peak; larger cutoff, no smaller peak; 0 on the CPU
    circ = tfim_circuit(8, 0.05) * 4
    propagate(circ, z0(8), maxdegree=3, parallel="gpu")
    a = ff.trunc_stats()["peak_device_bytes"]
    propagate(circ, z0(8), maxdegree=3, parallel="gpu")
    b = ff.trunc_stats()["peak_device_bytes"]
    propagate(circ, z0(8), maxdegree=5, parallel="gpu")
    c = ff.trunc_stats()["peak_device_bytes"]
    assert a > 0 and a == b and c >= a
    propagate(circ, z0(8), maxdegree=3)
    assert ff.trunc_stats()["peak_device_bytes"] == 0

@pytest.mark.parametrize("reserve", [0, 100000])
def test_reserve_terms(reserve):
    assert_gpu_agrees(tfim_circuit(8, 0.05) * 6, z0(8), maxdegree=3, reserve_terms=reserve)


# Support-list key

@pytest.mark.parametrize("key", ["dense", "support"])
@pytest.mark.parametrize("w", [2, 3, 4])
def test_key_formats(key, w):
    assert_gpu_agrees(tfim_circuit(10, 0.05) * 8, z0(10), maxdegree=w, gpu_key=key)

@pytest.mark.parametrize("key", ["dense", "support"])
def test_key_formats_with_threshold(key):
    assert_gpu_agrees(heisenberg_circuit(8, 0.05) * 6, z0(8), maxdegree=4, mincoeff=1e-6, mincoeff_period=3,
                      batched=False, gpu_key=key)

def test_support_key_wide_sites():
    circ = [ff.ROT("XX", [0, 100], 0.1), ff.ROT("ZZ", [0, 100], 0.1), ff.ROT("X", [0], 0.1)] * 5
    assert_gpu_agrees(circ, PauliString("Z0 Z100"), maxdegree=4, gpu_key="support")

def test_support_key_rejects_invalid():
    with pytest.raises(Exception):  # deferred degree cutoff
        propagate(tfim_circuit(4, 0.05), z0(4), parallel="gpu", maxdegree=2, maxdegree_period=3, gpu_key="support")
    with pytest.raises(Exception):  # Clifford gates can raise degrees past the slots
        propagate([ff.CNOT(0, 1), ff.ROT("X", [0], 0.1)], z0(2), parallel="gpu", maxdegree=2, gpu_key="support")
    with pytest.raises(Exception):  # site 127
        propagate([ff.ROT("XX", [0, 127], 0.1)], z0(1), parallel="gpu", maxdegree=2, gpu_key="support")
    with pytest.raises(Exception):  # rotation on 3 sites
        propagate([ff.ROT("XXX", [0, 1, 2], 0.1)], z0(1), parallel="gpu", maxdegree=3, gpu_key="support")
    assert_gpu_agrees([ff.ROT("XXX", [0, 1, 2], 0.1)] * 4, z0(1), maxdegree=3)  # auto falls back to dense

@pytest.mark.parametrize("beta", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("key", ["dense", "support"])
def test_beta_does_not_change_results(beta, key):
    assert_gpu_agrees(tfim_circuit(8, 0.05) * 6, z0(8), maxdegree=3, gpu_key=key, gpu_beta=beta)


# Majorana polynomials

def hermitian_pair(i, j):
    return MajoranaPolynomial(MajoranaString([i, j])) * 1j  # i m_i m_j is Hermitian

@pytest.mark.parametrize("kwargs", [
    dict(maxdegree=4),
    dict(maxdegree=4, mincoeff=1e-6),
    dict(maxdegree=4, mincoeff=1e-6, mincoeff_period=3, batched=False),
    dict(maxdegree=4, maxdegree_period=2, batched=False),
    dict(maxdegree=6, gpu_beta=0.0),
])
def test_majorana(kwargs):
    assert_gpu_agrees(tv_circuit(5, 1.0, 2.0, 0.09, 4), hermitian_pair(1, 2), **kwargs)

def test_majorana_free_circuit():
    circ = tv_circuit(5, 1.0, 0.0, 0.07, 6)
    assert_poly_close(propagate(circ, hermitian_pair(1, 2), parallel="gpu"),
                      propagate(circ, hermitian_pair(1, 2), parallel="gpu", maxdegree=2), tol=1e-13)
    assert ff.trunc_stats()["peak_device_bytes"] > 0

def test_majorana_two_word_keys():
    # more than 128 Majorana operators: 66 modes, free (the degree stays 2)
    assert_gpu_agrees(tv_circuit(66, 1.0, 0.0, 0.05, 3), hermitian_pair(1, 2), maxdegree=2)

def test_majorana_threshold_without_degree_cutoff():
    assert_gpu_agrees(tv_circuit(4, 1.0, 2.0, 0.1, 4), hermitian_pair(1, 2), mincoeff=1e-4)

def test_majorana_rejects_unsupported():
    circ = tv_circuit(4, 1.0, 2.0, 0.1, 3)
    with pytest.raises(Exception):
        propagate(circ, hermitian_pair(1, 2), parallel="gpu", gpu_key="support")
    with pytest.raises(Exception):
        propagate(circ, hermitian_pair(1, 2), parallel="gpu", max_unpaired=2)
    with pytest.raises(Exception):  # non-Hermitian observable
        propagate(circ, MajoranaPolynomial(MajoranaString([1, 2])), parallel="gpu")
