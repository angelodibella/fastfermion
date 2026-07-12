"""Tests for the CUDA GPU Pauli propagation backend.

Verifies propagate(parallel="gpu") against propagate(parallel="serial"):
bit-identical retained sets under the pure weight cutoff, coefficient
agreement, matched threshold schedules, and the device phase rule against
the host oracle. Skipped when fastfermion was built without GPU support or
no CUDA device is present.
"""

import pytest
import fastfermion as ff

pytestmark = pytest.mark.skipif(
    not ff.has_gpu or ff.gpu_device_count() == 0,
    reason="fastfermion built without GPU support or no CUDA device",
)


def assert_equal(circuit, n_steps, obs, **kwargs):
    if isinstance(obs, str):
        obs = ff.PauliString(obs)
    full = circuit * n_steps
    ref = ff.propagate(full, obs, parallel="serial", **kwargs)
    gpu = ff.propagate(full, obs, parallel="gpu", **kwargs)
    assert len(ref.terms) == len(gpu.terms), (
        f"term count: serial={len(ref.terms)}, gpu={len(gpu.terms)}"
    )
    for ps, c in ref.terms.items():
        assert abs(c - gpu.terms.get(ps, 0)) < 1e-12, (
            f"coeff at {ps}: serial={c}, gpu={gpu.terms.get(ps, 0)}"
        )
    assert abs(ref.overlapwithzero() - gpu.overlapwithzero()) < 1e-12


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


def test_phase_rule():
    # underscore names are not re-exported by the package's `import *`
    from fastfermion import ffcore

    assert ffcore._gpu_phase_check(100000, 1234)
    assert ffcore._gpu_phase_check(100000, 5678)


@pytest.mark.parametrize("w", [2, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_tfim_weight_cutoff(w, batched):
    assert_equal(tfim(10, 0.05), 8, "Z0", maxdegree=w, batched=batched)


@pytest.mark.parametrize("w", [3, 4])
def test_heisenberg_weight_cutoff(w):
    assert_equal(heisenberg(8, 0.05), 6, "Z0", maxdegree=w)


@pytest.mark.parametrize("batched", [False, True])
def test_threshold_schedule(batched):
    # mincoeff > 0 must fire at the CPU cadence (per gate / per commuting batch)
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=4, mincoeff=1e-6, batched=batched)


def test_no_truncation_exact():
    assert_equal(heisenberg(6, 0.05), 4, "Z0")


def test_wide_keys():
    # Force the 2-word key path: a qubit index above 64.
    g = [ff.ROT("XX", [0, 65], 0.1), ff.ROT("ZZ", [0, 65], 0.1), ff.ROT("X", [0], 0.1)]
    assert_equal(g, 5, ff.PauliString("Z0 Z65"), maxdegree=4)


def test_polynomial_observable():
    obs = ff.PauliPolynomial(ff.PauliString("Z0"))
    obs += ff.PauliPolynomial(ff.PauliString("Z1")) * 0.5
    assert_equal(tfim(8, 0.05), 4, obs, maxdegree=3)


def test_rejects_unsupported():
    with pytest.raises(Exception):
        ff.propagate(tfim(4, 0.05), ff.PauliString("Z0"), parallel="gpu", topk=10)
    with pytest.raises(Exception):
        ff.propagate(tfim(4, 0.05), ff.PauliString("Z0"), parallel="gpu", max_xweight=2)

# --- truncation schedules (maxdegree_period / mincoeff_period) ---------------
# GPU and serial CPU must retain identical sets under every schedule: the
# firing decisions use the same period_crossed test on both sides, and the
# GPU's scheduled compact() drops exactly what the CPU's truncate_all drops.


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("p_w", [2, 5])
def test_deferred_weight_schedule(p_w, batched):
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=3, maxdegree_period=p_w, batched=batched)


@pytest.mark.parametrize("p_tau", [3, 7])
def test_threshold_period(p_tau):
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=4, mincoeff=1e-6, mincoeff_period=p_tau,
                 batched=False)


def test_combined_schedules():
    assert_equal(heisenberg(8, 0.05), 6, "Z0", maxdegree=3, maxdegree_period=3, mincoeff=1e-6,
                 mincoeff_period=2, batched=False)


def test_certificate_backend_agreement():
    # The certificate is accumulated on the host from per-event device
    # reductions; it must match the CPU value for the same schedule to the
    # accuracy of a sum of ~1e2 event norms.
    circ = [g for _ in range(6) for g in tfim(8, 0.05)]
    obs = ff.PauliString("Z" + "I" * 7)
    ff.propagate(circ, obs, maxdegree=4, mincoeff=1e-5, mincoeff_period=2, batched=False)
    cpu = ff.trunc_stats()
    ff.propagate(circ, obs, maxdegree=4, mincoeff=1e-5, mincoeff_period=2, batched=False,
                 parallel="gpu")
    gpu = ff.trunc_stats()
    assert gpu["n_tau_events"] == cpu["n_tau_events"]
    assert abs(gpu["cert_tau"] - cpu["cert_tau"]) < 1e-12 * max(1.0, cpu["cert_tau"])


def test_peak_device_bytes_reported():
    # The allocator counter is deterministic: same run, same peak; larger
    # cutoff, no smaller peak. CPU runs report 0.
    circ = [g for _ in range(4) for g in tfim(8, 0.05)]
    obs = ff.PauliString("Z" + "I" * 7)
    ff.propagate(circ, obs, maxdegree=3, parallel="gpu")
    a = ff.trunc_stats()["peak_device_bytes"]
    ff.propagate(circ, obs, maxdegree=3, parallel="gpu")
    b = ff.trunc_stats()["peak_device_bytes"]
    ff.propagate(circ, obs, maxdegree=5, parallel="gpu")
    c = ff.trunc_stats()["peak_device_bytes"]
    assert a > 0 and a == b and c >= a
    ff.propagate(circ, obs, maxdegree=3)
    assert ff.trunc_stats()["peak_device_bytes"] == 0


@pytest.mark.parametrize("reserve", [0, 100000])
def test_reserve_knob_agreement(reserve):
    # 0 = pure growth path; explicit N = hard preallocation. Both must retain
    # exactly the auto-reserve run's terms -- sizing policy cannot touch results.
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=3, reserve_terms=reserve)


# --- support-list ("sparse") key ---------------------------------------------
# The key representation must be invisible in the results: identical retained
# sets and coefficients vs the serial CPU whatever the format.


@pytest.mark.parametrize("key", ["dense", "support"])
@pytest.mark.parametrize("w", [2, 3, 4])
def test_key_formats_agree(key, w):
    assert_equal(tfim(10, 0.05), 8, "Z0", maxdegree=w, gpu_key=key)


@pytest.mark.parametrize("key", ["dense", "support"])
def test_key_formats_with_threshold_and_periods(key):
    assert_equal(heisenberg(8, 0.05), 6, "Z0", maxdegree=4, mincoeff=1e-6,
                 mincoeff_period=3, batched=False, gpu_key=key)


def test_support_key_wide_sites():
    # sites up to 126 are representable; the win check needs 2-word dense
    g = [ff.ROT("XX", [0, 100], 0.1), ff.ROT("ZZ", [0, 100], 0.1), ff.ROT("X", [0], 0.1)]
    assert_equal(g, 5, ff.PauliString("Z0 Z100"), maxdegree=4, gpu_key="support")


def test_support_key_rejects_ineligible():
    with pytest.raises(Exception):  # deferred weight cutoff
        ff.propagate(tfim(4, 0.05), ff.PauliString("Z" + "I" * 3), parallel="gpu",
                     maxdegree=2, maxdegree_period=3, gpu_key="support")
    with pytest.raises(Exception):  # site 127 collides with the sentinel
        ff.propagate([ff.ROT("XX", [0, 127], 0.1)], ff.PauliString("Z0"), parallel="gpu",
                     maxdegree=2, gpu_key="support")


@pytest.mark.parametrize("beta", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("key", ["dense", "support"])
def test_beta_schedule_free(beta, key):
    # Compaction invariance made a test: the dedup cadence cannot change results.
    assert_equal(tfim(8, 0.05), 6, "Z0", maxdegree=3, gpu_key=key, gpu_beta=beta)
